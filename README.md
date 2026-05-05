# Automated Drum Stem Extractor

Extracts drum segments from a batch of audio files, keeping **only** the
portions where drums are playing *during* vocal performance. Instrumental
breaks are automatically excluded.

Built for Arabic and Western music. Primary runtime: **Google Colab (NVIDIA T4)**.
Falls back gracefully to Apple Silicon MPS or CPU.

---

## How it works

```
Audio file
   │
   ▼
[Demucs]  ──────────────────────── Separate into Vocals / Drums / Bass / Other
   │
   ▼
[Silero VAD]  ──────────────────── Detect vocal windows (6 s silence rule)
   │
   ▼
[Drum scanner]  ────────────────── Find continuous drum bursts inside each
   │                               vocal window; cut on ≥ 1.5 s of silence
   ▼
[MP3 export]  ──────────────────── Save numbered segments to output folder
```

---

## Requirements

- Python 3.10 or 3.11
- **ffmpeg** installed and on your `PATH` (required by pydub for MP3 encoding)
- PyTorch — installed separately based on your hardware (see below)

---

## Installation

### 1 — Install ffmpeg

**Ubuntu / Debian / Google Colab**
```bash
apt-get install -y ffmpeg
```

**macOS (Homebrew)**
```bash
brew install ffmpeg
```

**Windows**
Download from <https://ffmpeg.org/download.html> and add the `bin/` folder to
your system `PATH`.

---

### 2 — Install PyTorch

Install PyTorch **before** running `pip install -r requirements.txt`.

#### Google Colab (CUDA 12.x — T4, A100, L4)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Local machine with NVIDIA GPU (CUDA 11.8)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Apple Silicon (MPS)
```bash
pip install torch torchvision torchaudio
```

#### CPU only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> Check <https://pytorch.org/get-started/locally/> for the exact command that
> matches your CUDA version.

---

### 3 — Install Python dependencies
```bash
pip install -r requirements.txt
```

---

### 4 — (Google Colab only) Full setup cell

Paste this into a Colab code cell to go from zero to ready:

```python
# 1. ffmpeg
import subprocess
subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)

# 2. PyTorch (CUDA 12.1 — works on T4 / A100 / L4)
subprocess.run([
    "pip", "install", "-q",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu121",
], check=True)

# 3. Project dependencies
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

print("Setup complete!")
```

---

## Usage

```
python main.py --input <folder> --output <folder> [options]
```

### Minimal example
```bash
python main.py \
  --input  /music/arabic_tracks \
  --output /music/drum_exports
```

### Full example with custom parameters
```bash
python main.py \
  --input                /music/arabic_tracks \
  --output               /music/drum_exports \
  --stems-cache          /music/stems_cache \
  --vad-silence-duration 6.0 \
  --vad-threshold        0.45 \
  --drum-silence-db      -40 \
  --drum-silence-duration 1.5 \
  --drum-min-segment     1.0 \
  --drum-tail-pad        0.4 \
  --model                htdemucs
```

---

## CLI Reference

| Argument                    | Default          | Description |
|-----------------------------|------------------|-------------|
| `--input`                   | *(required)*     | Folder containing source audio files |
| `--output`                  | *(required)*     | Folder for exported drum MP3s (created if missing) |
| `--stems-cache`             | `./stems_cache`  | Where Demucs stores separated stems between runs |
| `--vad-silence-duration`    | `6.0`            | Seconds of continuous silence required to close a vocal window |
| `--vad-threshold`           | `0.45`           | Silero VAD speech-probability threshold. Higher = stricter. Calibrated for htdemucs stems; lower only if melismatic tail-ends are missed |
| `--drum-silence-db`         | `-40` *(auto)*   | dB cut point for drums. At the default, a stem-relative dynamic threshold is used (peak − 35 dB, floored at −60). Pass any explicit value to override |
| `--drum-silence-duration`   | `1.5`            | Seconds of drum silence required to split segments |
| `--drum-min-segment`        | `1.0`            | Minimum exported segment duration (seconds) |
| `--drum-tail-pad`           | `0.4`            | Seconds of resonance tail to preserve after a segment's silence onset. Prevents doumbek/riq decay from being clipped. Applied to mid-window splits only |
| `--model`                   | `htdemucs`       | Demucs model name |

---

## Output naming convention

For a source file `Fadhel_Shaker_Ya_Ghayeb.mp3` that yields three segments:

```
Fadhel_Shaker_Ya_Ghayeb_drums_01.mp3
Fadhel_Shaker_Ya_Ghayeb_drums_02.mp3
Fadhel_Shaker_Ya_Ghayeb_drums_03.mp3
```

---

## Supported input formats

`.mp3` · `.wav` · `.flac` · `.ogg` · `.m4a` · `.aac`

---

## Arabic music notes

Arabic music presents several characteristics that influenced the default
parameter choices:

**Vocal style**
- Melismatic (maqam) phrasing produces dense ornamental runs that Silero VAD
  may parse as many short bursts. The 6-second merge window (`--vad-silence-duration`)
  keeps entire sung phrases as a single vocal zone.
- The VAD threshold (`--vad-threshold`, default `0.45`) is calibrated for
  Demucs-separated stems. If melismatic tail-ends are being missed, lower this
  value cautiously (try `0.40`); do not go below `0.35` without also raising
  `--vad-silence-duration` to compensate for increased false positives.
- If your recordings have longer instrumental interludes between vocal passages,
  reduce `--vad-silence-duration` (e.g. `3.0`) to split them.

**Percussion**
- Egyptian doumbek, tabla baladi, and riq produce sharp high-frequency
  transients. The 50 ms RMS frame window used internally resolves these
  cleanly. The drum silence threshold defaults to a stem-relative dynamic
  value (peak − 35 dB) to handle tracks at any mastering level. Lower
  `--drum-silence-db` manually (e.g. `-50`) only if quieter hand-percussion
  fills are still being missed after the dynamic threshold is logged.
- The `--drum-tail-pad` (default `0.4 s`) preserves the natural ring-out of
  doumbek and riq strokes that decay below the silence threshold before they
  fully resolve. Increase to `0.6` if decay tails are still being clipped.
- If you hear a very long drum roll being split into multiple segments, increase
  `--drum-silence-duration` (e.g. `2.5`).

**Demucs model**
- `htdemucs` (default) is the recommended model for both Arabic and Western
  material. `htdemucs_ft` is a fine-tuned variant that is slower but can
  improve separation quality on some material; however, it does **not** fix
  the Arabic percussion routing issue (doumbek / riq energy routes to the
  `other` stem regardless of model variant).

---

## Stems caching

Demucs is the slowest step. Separated stems are written to `--stems-cache`
after the first run. Re-running the script on the same file reuses the cached
WAVs instantly, so you can safely experiment with VAD and drum parameters
without paying the Demucs cost again.

To force re-separation, delete the relevant subfolder inside the cache:
```bash
rm -rf ./stems_cache/Fadhel_Shaker_Ya_Ghayeb/
```

---

## Console output example

```
🖥️  Device: CUDA  (Tesla T4)
🔊  Loading Silero VAD model...
🔊  Silero VAD ready.

📂  Found 12 audio file(s) in: /music/arabic_tracks
════════════════════════════════════════════════════════════

[1/12] Fadhel_Shaker_Ya_Ghayeb.mp3
    🎵  Separating stems...
    🎤  Detecting vocal activity... 2 vocal window(s) found
    🥁  Detecting drum segments within vocal zones... 3 segment(s) found
    💾  Exported: Fadhel_Shaker_Ya_Ghayeb_drums_01.mp3
    💾  Exported: Fadhel_Shaker_Ya_Ghayeb_drums_02.mp3
    💾  Exported: Fadhel_Shaker_Ya_Ghayeb_drums_03.mp3

[2/12] Instrumental_Overture.mp3
    🎵  Separating stems...
    ⚠️   No vocals detected. Skipping extraction.

════════════════════════════════════════════════════════════
✅  Batch complete.
    Processed : 11
    Skipped   : 1  (no vocals detected or no drum segments found)
    Errors    : 0
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'demucs'` | Run `pip install -r requirements.txt` |
| `ffmpeg not found` | Install ffmpeg and ensure it is on your `PATH` |
| Vocal windows too short (Arabic ornaments cut off) | Increase `--vad-silence-duration` (try `8.0`) |
| Phantom vocal windows in instrumental sections | Increase `--vad-threshold` (try `0.50`) |
| True vocal windows being missed | Decrease `--vad-threshold` (try `0.40`) |
| Doumbek/riq tails clipped at segment end | Increase `--drum-tail-pad` (try `0.6`) |
| Too many tiny drum segments | Increase `--drum-min-segment` (try `2.0`) |
| Drum segments not splitting at breaks | Decrease `--drum-silence-duration` (try `1.0`) |
| CUDA out of memory | Use `--model htdemucs` (default) instead of larger variants |
| MPS errors on Apple Silicon | Export `PYTORCH_ENABLE_MPS_FALLBACK=1` before running |

---

## License

MIT — see `LICENSE` for details.
