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
    silence_db: float,
    silence_duration: float,
    device: torch.device,
) -> list[tuple[float, float]]:
    """
    Analyze the vocal stem and return a list of merged vocal windows:
        [(start_sec, end_sec), ...]

    A window only closes when the vocal track stays below *silence_db* for
    more than *silence_duration* consecutive seconds.  Any shorter gap —
    breaths, ornamental pauses, maqam melisma — does NOT split the window.

    This makes the detector robust for Arabic music, which frequently contains
    long melismatic phrases separated by micro-pauses.
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
        threshold=0.30,           # slightly permissive for ornamental vocals
        min_speech_duration_ms=150,
        min_silence_duration_ms=200,
        return_seconds=False,     # returns sample indices
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


def _split_on_silence(
    times: np.ndarray,
    rms_db: np.ndarray,
    silence_db: float,
    silence_duration: float,
    win_start: float,
    win_end: float,
) -> list[tuple[float, float]]:
    """
    Walk a frame-level dB trace and split it into contiguous active segments.

    A split is only made when the signal stays below *silence_db* for at least
    *silence_duration* seconds — mirroring Audacity's "silence finder" logic.

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
                    # Silence confirmed long enough → close segment at onset
                    segments.append((seg_start, silence_onset))
                    seg_start = None
                    silence_onset = None
            else:
                # Back above threshold → reset silence counter
                silence_onset = None

    # Close any still-open segment at the vocal window boundary
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
            silence_db, silence_duration,
            win_start, win_end,
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
    vad_silence_db: float,
    vad_silence_duration: float,
    drum_silence_db: float,
    drum_silence_duration: float,
    drum_min_segment: float,
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
            vad_silence_db,
            vad_silence_duration,
            device,
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
    "--vad-silence-db",
    default=-40.0,
    show_default=True,
    type=float,
    help="dB level below which the vocal track is considered silent.",
)
@click.option(
    "--vad-silence-duration",
    default=6.0,
    show_default=True,
    type=float,
    help="Seconds of continuous silence required to close a vocal window.",
)
@click.option(
    "--drum-silence-db",
    default=-40.0,
    show_default=True,
    type=float,
    help="dB level below which the drum track is considered a cut point.",
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
    "--model",
    default="htdemucs",
    show_default=True,
    help="Demucs model name.",
)
def main(
    input_dir: Path,
    output_dir: Path,
    stems_cache: Path,
    vad_silence_db: float,
    vad_silence_duration: float,
    drum_silence_db: float,
    drum_silence_duration: float,
    drum_min_segment: float,
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
            vad_silence_db=vad_silence_db,
            vad_silence_duration=vad_silence_duration,
            drum_silence_db=drum_silence_db,
            drum_silence_duration=drum_silence_duration,
            drum_min_segment=drum_min_segment,
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
