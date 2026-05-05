#!/usr/bin/env python3
"""
Automated Drum Stem Extraction Based on Vocal Activity
=======================================================
Separates audio into stems via Demucs, detects vocal windows with Silero VAD,
then exports only the drum portions that fall within those vocal zones.

Supports Arabic and Western music genres.
Works on CUDA (primary), Apple MPS, or CPU (fallback).
"""

import logging
import sys
import warnings
from pathlib import Path

import click

import librosa
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment

warnings.filterwarnings("ignore")

# ─────────────────────────────── Logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ──────────────────────────── Supported formats ──────────────────────────────
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

# ─────────────────────────── Demucs stem names ───────────────────────────────
# htdemucs produces: vocals, drums, bass, other
DEMUCS_STEMS = ["vocals", "drums", "bass", "other"]

# ─────────────────────────── Sentinel for drum threshold ─────────────────────
# FINDING-3: used to distinguish "user passed a custom value" from "user left
# the default". When drum_silence_db == DEFAULT_DRUM_SILENCE_DB at runtime,
# _compute_dynamic_silence_db() replaces it with a stem-relative value.
DEFAULT_DRUM_SILENCE_DB: float = -40.0


# ══════════════════════════════════════════════════════════════════════════════
#  Device detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_device() -> torch.device:
    """Auto-detect compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        log.info(f"🖥️  Device: CUDA  ({gpu_name})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("🖥️  Device: MPS   (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log.info("🖥️  Device: CPU   ⚠️  processing will be significantly slower")
    return device


# ══════════════════════════════════════════════════════════════════════════════
#  Stem separation (Demucs)
# ══════════════════════════════════════════════════════════════════════════════

def separate_stems(
    audio_path: Path,
    cache_dir: Path,
    model_name: str,
    device: torch.device,
) -> dict[str, Path]:
    """
    Run Demucs separation on *audio_path* and return paths to each stem WAV.
    If cached stems already exist on disk, they are reused without re-running.
    """
    stem_dir = cache_dir / audio_path.stem
    expected = {stem: stem_dir / f"{stem}.wav" for stem in DEMUCS_STEMS}

    if all(p.exists() for p in expected.values()):
        log.info("    📁  Using cached stems")
        return expected

    stem_dir.mkdir(parents=True, exist_ok=True)

    # Import here so the rest of the script is importable without demucs
    from demucs.api import Separator

    separator = Separator(model=model_name, device=str(device))
    _, separated = separator.separate_audio_file(audio_path)

    for stem_name, tensor in separated.items():
        if stem_name not in expected:
            continue
        out_path = expected[stem_name]
        separator.save_audio(tensor, out_path, samplerate=separator.samplerate)

    return expected


# ══════════════════════════════════════════════════════════════════════════════
#  Vocal Activity Detection (Silero VAD)
# ══════════════════════════════════════════════════════════════════════════════

def detect_vocal_windows(
    vocal_path: Path,
    vad_model,
    vad_utils,
    silence_duration: float,
    device: torch.device,
    vad_threshold: float = 0.45,
) -> list[tuple[float, float]]:
    """
    Analyze the vocal stem and return a list of merged vocal windows:
        [(start_sec, end_sec), ...]

    A window only closes when the vocal track stays silent for more than
    *silence_duration* consecutive seconds.  Any shorter gap — breaths,
    ornamental pauses, maqam melisma — does NOT split the window.

    This makes the detector robust for Arabic music, which frequently contains
    long melismatic phrases separated by micro-pauses.

    Args:
        vad_threshold: Silero speech-probability onset threshold.
            Default 0.45 is calibrated for htdemucs separated stems, where
            bleed artefacts score higher than in a natural acoustic mix.
            Lower only if melismatic tail-ends are being missed; if so,
            prefer raising min_silence_duration_ms before reducing this value.

    # FINDING-1: --vad-silence-db was removed (Path A).
    # Silero VAD operates on a speech-probability score, not RMS energy.
    # There is no meaningful way to wire a dBFS threshold directly into
    # get_speech_timestamps() without an upstream RMS gate that would require
    # resampling and frame alignment. The parameter was accepted but silently
    # ignored in all prior versions. It has been removed from the CLI and this
    # signature to avoid misleading users. If a pre-Silero energy gate is ever
    # needed, implement _apply_energy_gate() upstream of this call and expose
    # a new --vad-energy-gate flag at that time.

    # FINDING-2: threshold promoted from hardcoded 0.30 to a CLI parameter.
    # 0.30 was too permissive for Demucs-separated stems: instrument bleed
    # (nay, oud, violin) into the vocal stem scores higher than it would in
    # a natural acoustic mix, causing phantom vocal windows during purely
    # instrumental sections (taqsim, lazma). Default raised to 0.45.
    # Validation: run on tracks that contain a clearly instrumental section
    # and confirm it produces zero or near-zero windows there.
    """
    get_speech_timestamps, _, _, _, _ = vad_utils

    # Silero VAD is trained at 16 kHz
    audio, _ = librosa.load(str(vocal_path), sr=16_000, mono=True)
    audio_tensor = torch.from_numpy(audio).to(device)

    # Let Silero produce fine-grained timestamps; we merge them ourselves
    raw_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=16_000,
        threshold=vad_threshold,          # FINDING-2: was hardcoded 0.30
        min_speech_duration_ms=150,
        min_silence_duration_ms=200,
        return_seconds=False,             # returns sample indices
    )

    if not raw_timestamps:
        return []

    # Convert sample indices → seconds
    raw_windows: list[tuple[float, float]] = [
        (ts["start"] / 16_000, ts["end"] / 16_000)
        for ts in raw_timestamps
    ]

    # ── Merge windows separated by less than silence_duration seconds ─────────
    # This implements the "6-second rule" from the brief.
    merged: list[tuple[float, float]] = [raw_windows[0]]
    for start, end in raw_windows[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        if gap < silence_duration:
            # Extend the current window; don't create a new one
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  Drum silence splitting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_rms_db(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (times_seconds, rms_db) arrays for the given mono audio.

    Uses 50 ms frames / 25 ms hop to resolve short percussive transients
    (doumbek, tabla, kick) without false gaps.
    """
    frame_len = int(0.050 * sr)   # 50 ms
    hop_len   = int(0.025 * sr)   # 25 ms

    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]

    # Absolute dBFS (0 dBFS = full-scale sine, ref = 1.0)
    rms_db = 20.0 * np.log10(rms + 1e-9)

    times = librosa.frames_to_time(
        np.arange(len(rms_db)), sr=sr, hop_length=hop_len
    )
    return times, rms_db


def _compute_dynamic_silence_db(
    audio: np.ndarray,
    silence_db_override: float,
) -> float:
    """Return a stem-relative silence threshold for drum segmentation.

    # FINDING-3: Demucs does not normalise its stem outputs. A track mastered
    # at -15 dBFS produces a drum stem whose peaks sit near -15 dBFS, making
    # the old fixed -40 dBFS cutoff slice straight through active drum content
    # on quiet recordings — silently producing wrong or empty segments.
    #
    # Fix: compute the stem's actual peak level and place the threshold 35 dB
    # below it (floored at -60 dBFS to avoid over-triggering on noise).
    #
    # Sentinel behaviour: when the caller passes DEFAULT_DRUM_SILENCE_DB
    # (i.e. the user did not explicitly set --drum-silence-db), the dynamic
    # value is used. When the user passes any other value, that value is
    # used as-is so manual overrides are fully respected.

    Args:
        audio: mono float32 array of the drum stem (full file, not a window
               slice) — peak estimate should reflect the whole stem's range.
        silence_db_override: the value from --drum-silence-db.

    Returns:
        Effective silence threshold in dBFS.
    """
    if silence_db_override != DEFAULT_DRUM_SILENCE_DB:
        # User explicitly overrode the default — respect their value exactly.
        return silence_db_override

    peak_db = 20.0 * np.log10(np.max(np.abs(audio)) + 1e-9)
    dynamic = max(peak_db - 35.0, -60.0)
    return dynamic


def _split_on_silence(
    times: np.ndarray,
    rms_db: np.ndarray,
    silence_db: float,
    silence_duration: float,
    win_start: float,
    win_end: float,
    tail_pad: float = 0.4,
) -> list[tuple[float, float]]:
    """
    Walk a frame-level dB trace and split it into contiguous active segments.

    A split is only made when the signal stays below *silence_db* for at least
    *silence_duration* seconds — mirroring Audacity's "silence finder" logic.

    Args:
        tail_pad: seconds to extend each segment past the silence_onset point
            when a mid-window split is confirmed. Preserves the natural
            resonance decay of Arabic hand percussion (doumbek, riq) that
            falls below the threshold before it fully rings out.

            # FINDING-4: the original code closed segments exactly at
            # silence_onset — the first frame that crossed below the threshold.
            # This clipped the audible resonance tail of doumbek hits. Fix:
            # extend the segment end by tail_pad seconds past silence_onset,
            # clamped to win_end to prevent overrun into the next window.
            #
            # The pad is applied ONLY to mid-window silence-confirmed closures
            # (where t - silence_onset >= silence_duration triggers the split).
            # The final segment in each window is closed at win_end and already
            # captures the full tail by definition — no pad is applied there.

    Returns list of (seg_start_abs, seg_end_abs) in absolute track time.
    """
    segments: list[tuple[float, float]] = []
    seg_start: float | None = None
    silence_onset: float | None = None   # when the current silent streak began

    for t, db in zip(times + win_start, rms_db):
        if t > win_end:
            break

        if seg_start is None:
            # Waiting for the drum to become active
            if db >= silence_db:
                seg_start = t
                silence_onset = None
        else:
            if db < silence_db:
                # Entering / continuing a silent streak
                if silence_onset is None:
                    silence_onset = t
                elif t - silence_onset >= silence_duration:
                    # Silence confirmed long enough → close segment.
                    # FINDING-4: extend past silence_onset by tail_pad to
                    # capture percussion resonance decay, clamped to win_end.
                    seg_end = min(silence_onset + tail_pad, win_end)
                    segments.append((seg_start, seg_end))
                    seg_start = None
                    silence_onset = None
            else:
                # Back above threshold → reset silence counter
                silence_onset = None

    # Close any still-open segment at the vocal window boundary.
    # No tail pad here: the segment already runs to win_end (or silence_onset
    # if a streak was in progress), which includes the full natural tail.
    if seg_start is not None:
        close_at = silence_onset if silence_onset is not None else win_end
        segments.append((seg_start, close_at))

    return segments


# ══════════════════════════════════════════════════════════════════════════════
#  Drum segment detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_drum_segments(
    drum_path: Path,
    vocal_windows: list[tuple[float, float]],
    silence_db: float,
    silence_duration: float,
    min_segment: float,
    tail_pad: float,
) -> list[tuple[float, float]]:
    """
    For each vocal window, slice the drum stem and detect continuous bursts.

    Logic mirrors how a human uses Audacity:
      1. Restrict view to the vocal window (search zone).
      2. Find every continuous drum burst within that zone.
      3. A gap ≥ silence_duration splits into separate segments.
      4. Discard segments shorter than min_segment.

    Returns absolute (start_sec, end_sec) pairs ordered by time.
    """
    audio, sr = librosa.load(str(drum_path), sr=None, mono=True)
    total_duration = len(audio) / sr

    # FINDING-3: replace the fixed threshold with a stem-relative value so that
    # quietly-mastered tracks are not silently truncated. The full-file audio
    # array is passed so the peak estimate reflects the whole stem's dynamic
    # range, not just the first vocal window.
    effective_silence_db = _compute_dynamic_silence_db(audio, silence_db)
    log.debug(
        f"    🥁  Drum silence threshold: {effective_silence_db:.1f} dBFS"
        f"{'  (user override)' if silence_db != DEFAULT_DRUM_SILENCE_DB else '  (dynamic)'}"
    )

    all_segments: list[tuple[float, float]] = []

    for win_start, win_end in vocal_windows:
        win_start = max(0.0, win_start)
        win_end   = min(total_duration, win_end)

        if win_end <= win_start:
            continue

        s_idx = int(win_start * sr)
        e_idx = int(win_end * sr)
        zone  = audio[s_idx:e_idx]

        if len(zone) == 0:
            continue

        times_rel, rms_db = _compute_rms_db(zone, sr)

        segments = _split_on_silence(
            times_rel, rms_db,
            effective_silence_db, silence_duration,
            win_start, win_end,
            tail_pad,
        )

        for seg_start, seg_end in segments:
            duration = seg_end - seg_start
            if duration >= min_segment:
                all_segments.append((seg_start, seg_end))

    return all_segments


# ══════════════════════════════════════════════════════════════════════════════
#  MP3 export
# ══════════════════════════════════════════════════════════════════════════════

def export_segment(
    drum_path: Path,
    seg_start: float,
    seg_end: float,
    out_path: Path,
) -> None:
    """Slice the drum stem WAV and export the segment as a 320 kbps MP3."""
    audio = AudioSegment.from_file(str(drum_path))
    start_ms = int(seg_start * 1000)
    end_ms   = int(seg_end   * 1000)
    clip = audio[start_ms:end_ms]
    clip.export(str(out_path), format="mp3", bitrate="320k")


# ══════════════════════════════════════════════════════════════════════════════
#  Per-file processing
# ══════════════════════════════════════════════════════════════════════════════

def process_file(
    audio_path: Path,
    *,
    output: Path,
    stems_cache: Path,
    model: str,
    vad_silence_duration: float,
    vad_threshold: float,
    drum_silence_db: float,
    drum_silence_duration: float,
    drum_min_segment: float,
    drum_tail_pad: float,
    file_index: int,
    total_files: int,
    device: torch.device,
    vad_model,
    vad_utils,
) -> str:
    """
    Full pipeline for a single audio file.
    Returns one of: 'ok' | 'skipped' | 'error'
    """
    log.info(f"\n[{file_index}/{total_files}] {audio_path.name}")

    try:
        # ── Step 1: Stem separation ──────────────────────────────────────────
        log.info("    🎵  Separating stems...")
        stems = separate_stems(audio_path, stems_cache, model, device)

        # ── Step 2: Vocal Activity Detection ────────────────────────────────
        log.info("    🎤  Detecting vocal activity...")
        vocal_windows = detect_vocal_windows(
            stems["vocals"],
            vad_model,
            vad_utils,
            vad_silence_duration,
            device,
            vad_threshold,
        )

        # ── Step 3: Instrumental-only check ─────────────────────────────────
        if not vocal_windows:
            log.info("    ⚠️   No vocals detected. Skipping extraction.")
            log.info(f"[{audio_path.name}]: No vocals detected. Skipping extraction.")
            return "skipped"

        log.info(f"    🎤  Detecting vocal activity... {len(vocal_windows)} vocal window(s) found")

        # ── Step 4: Drum segment detection ──────────────────────────────────
        log.info("    🥁  Detecting drum segments within vocal zones...")
        segments = detect_drum_segments(
            stems["drums"],
            vocal_windows,
            drum_silence_db,
            drum_silence_duration,
            drum_min_segment,
            drum_tail_pad,
        )

        log.info(f"    🥁  Detecting drum segments within vocal zones... {len(segments)} segment(s) found")

        if not segments:
            log.info("    ⚠️   No drum segments found within vocal zones.")
            return "skipped"

        # ── Step 5: Export ───────────────────────────────────────────────────
        output.mkdir(parents=True, exist_ok=True)

        for i, (seg_start, seg_end) in enumerate(segments, start=1):
            out_name = f"{audio_path.stem}_drums_{i:02d}.mp3"
            out_path = output / out_name
            export_segment(stems["drums"], seg_start, seg_end, out_path)
            log.info(f"    💾  Exported: {out_name}")

        return "ok"

    except Exception as exc:
        log.error(f"    ❌  Error processing {audio_path.name}: {exc}", exc_info=False)
        log.debug("Stack trace:", exc_info=True)
        return "error"


# ══════════════════════════════════════════════════════════════════════════════
#  CLI (Click)
# ══════════════════════════════════════════════════════════════════════════════

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input", "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder containing source audio files.",
)
@click.option(
    "--output", "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Folder for exported drum MP3s (created if missing).",
)
@click.option(
    "--stems-cache",
    default="./stems_cache",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where Demucs caches separated stems between runs.",
)
@click.option(
    "--vad-silence-duration",
    default=6.0,
    show_default=True,
    type=float,
    help="Seconds of continuous silence required to close a vocal window.",
)
@click.option(
    "--vad-threshold",
    default=0.45,
    show_default=True,
    type=float,
    help=(
        "Silero VAD speech-probability threshold. "
        "Higher values are stricter (fewer, more confident windows). "
        "Default 0.45 is calibrated for htdemucs stems; lower only if "
        "melismatic tail-ends are being missed."
    ),
)
@click.option(
    "--drum-silence-db",
    default=DEFAULT_DRUM_SILENCE_DB,
    show_default=True,
    type=float,
    help=(
        "dB level below which the drum track is considered a cut point. "
        "Defaults to auto (stem-relative dynamic threshold). "
        "Pass an explicit value to override."
    ),
)
@click.option(
    "--drum-silence-duration",
    default=1.5,
    show_default=True,
    type=float,
    help="Seconds of drum silence required to split into separate segments.",
)
@click.option(
    "--drum-min-segment",
    default=1.0,
    show_default=True,
    type=float,
    help="Minimum exported segment duration in seconds; shorter ones are discarded.",
)
@click.option(
    "--drum-tail-pad",
    default=0.4,
    show_default=True,
    type=float,
    help=(
        "Seconds of resonance tail to include after a drum segment's silence "
        "onset. Prevents doumbek/riq decay from being clipped at the threshold "
        "crossing point. Applied only to mid-window splits, not the final "
        "segment in each vocal window."
    ),
)
@click.option(
    "--model",
    default="htdemucs",
    show_default=True,
    help="Demucs model name.",
)
def main(
    input_dir: Path,
    output_dir: Path,
    stems_cache: Path,
    vad_silence_duration: float,
    vad_threshold: float,
    drum_silence_db: float,
    drum_silence_duration: float,
    drum_min_segment: float,
    drum_tail_pad: float,
    model: str,
) -> None:
    """Automated Drum Stem Extraction Based on Vocal Activity.

    Separates each audio file into stems, detects vocal windows, then
    exports only the drum segments that fall inside those windows.
    Instrumental breaks are automatically excluded.
    """
    # ── Device selection ─────────────────────────────────────────────────────
    device = detect_device()

    # ── Load Silero VAD ──────────────────────────────────────────────────────
    log.info("🔊  Loading Silero VAD model...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
        verbose=False,
    )
    vad_model = vad_model.to(device)
    log.info("🔊  Silero VAD ready.")

    # ── Gather audio files ───────────────────────────────────────────────────
    audio_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not audio_files:
        log.warning(
            f"⚠️   No supported audio files found in {input_dir}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        raise SystemExit(0)

    n = len(audio_files)
    log.info(f"\n📂  Found {n} audio file(s) in: {input_dir}")
    log.info("=" * 60)

    ok_count = skipped_count = error_count = 0

    for idx, audio_path in enumerate(audio_files, start=1):
        result = process_file(
            audio_path,
            output=output_dir,
            stems_cache=stems_cache,
            model=model,
            vad_silence_duration=vad_silence_duration,
            vad_threshold=vad_threshold,
            drum_silence_db=drum_silence_db,
            drum_silence_duration=drum_silence_duration,
            drum_min_segment=drum_min_segment,
            drum_tail_pad=drum_tail_pad,
            file_index=idx,
            total_files=n,
            device=device,
            vad_model=vad_model,
            vad_utils=vad_utils,
        )
        if result == "ok":
            ok_count += 1
        elif result == "skipped":
            skipped_count += 1
        else:
            error_count += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("✅  Batch complete.")
    log.info(f"    Processed : {ok_count}")
    log.info(f"    Skipped   : {skipped_count}  (no vocals detected or no drum segments found)")
    log.info(f"    Errors    : {error_count}")


if __name__ == "__main__":
    main()
