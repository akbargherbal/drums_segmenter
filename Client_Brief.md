# Project Brief: Automated Drum Stem Extraction Based on Vocal Activity

**Revision 4 — Final**

---

## 1. Project Overview

**Objective:** Develop a Python script that automates extraction of specific drum segments from a directory of mixed audio tracks. The script isolates the drum stem and exports only the portions where drums are actively playing _during_ vocal performance — excluding any drums that occur during instrumental breaks.

**Background:** Currently done manually in Audacity using the OpenVINO plugin. The user visually identifies where the singer is active, uses that as a search zone on the drum track, then finds and exports each continuous drum burst within that zone. This project fully automates that batch process.

**Primary Runtime Environment:** Google Colab (NVIDIA T4 GPU via CUDA). Must also run on CPU as a fallback.

---

## 2. Core Workflow & Logic

### Step 1 — Stem Separation

- Separate each audio file into: **Vocals, Drums, Bass, Other**.
- Use **Demucs** (`htdemucs` model by default).
- If stems for a track are already cached on disk, skip re-separation and reuse them.

### Step 2 — Vocal Activity Detection (VAD)

- Analyze the **Vocal stem** to produce a list of vocal windows: `[(start_sec, end_sec), ...]`.
- Use **Silero VAD** (preferred for reliability over simple energy detection).
- **Definition of a vocal window:** A vocal window opens when the singer becomes active, and only _closes_ when the vocal track falls and stays below **-40 dB for more than 6 continuous seconds.** Any pause shorter than 6 seconds — breaths, brief rests, short gaps — must **not** split or close the current vocal window.
- The output of this step is a set of merged, broad vocal zones that represent complete sung sections of the track.

### Step 3 — Instrumental Check

- If VAD returns zero vocal windows, skip this file entirely.
- Log: `"[Filename]: No vocals detected. Skipping extraction."`

### Step 4 — Drum Segment Detection Within Vocal Zones

This step mirrors exactly what a human does in Audacity: use the vocal window as the time range to examine, then detect each continuous burst of drum activity within that range.

- For each vocal window `(start_sec, end_sec)`, slice the **Drum stem** at those exact timestamps. This is the **search zone**.
- Within the search zone, scan the drum audio for silence using the drum silence parameters (see §3).
- Any period where the drum track falls below **`DRUM_SILENCE_DB`** for at least **`DRUM_SILENCE_DURATION`** seconds is treated as a **cut point** — it splits the zone into separate segments.
- Each resulting continuous drum burst is one export segment.
- Discard any segment shorter than **`DRUM_MIN_SEGMENT_SEC`** (e.g. very brief drum fills or noise hits that are not meaningful segments).

> **Example:** A vocal window runs from 2:25 to 3:10. Inside that window, the drums play from 2:25–2:48, then fall silent for 2 seconds, then play again from 2:50–3:10. Given a `DRUM_SILENCE_DURATION` of 1.5 sec, this produces **two** drum segments: one from 2:25–2:48 and one from 2:50–3:10.

### Step 5 — Export

- Export each drum segment as a separate numbered `.mp3` file.
- **Naming convention:** For a source file `Fadhel_Shaker_Ya_Ghayeb.mp3` with 3 drum segments:
  ```
  Fadhel_Shaker_Ya_Ghayeb_drums_01.mp3
  Fadhel_Shaker_Ya_Ghayeb_drums_02.mp3
  Fadhel_Shaker_Ya_Ghayeb_drums_03.mp3
  ```
- Save all output to the **user-specified output directory**. Create it automatically if it does not exist.

---

## 3. Configuration Parameters

All parameters must be exposed as CLI arguments (with clear defaults), and optionally as labeled constants at the top of the script:

| Parameter                 | Default         | Description                                                                          |
| ------------------------- | --------------- | ------------------------------------------------------------------------------------ |
| `--input`                 | _(required)_    | Path to folder containing source audio files                                         |
| `--output`                | _(required)_    | Path for exported drum MP3s; created if missing                                      |
| `--stems-cache`           | `./stems_cache` | Where Demucs stores separated stems between runs                                     |
| `--vad-silence-db`        | `-40`           | dB level below which vocals are considered silent                                    |
| `--vad-silence-duration`  | `6.0`           | Seconds below threshold required to close a vocal window                             |
| `--drum-silence-db`       | `-40`           | dB level below which drums are considered silent (cut point threshold)               |
| `--drum-silence-duration` | `1.5`           | Seconds of drum silence required to create a cut between segments                    |
| `--drum-min-segment`      | `1.0`           | Minimum duration (sec) for a drum segment to be exported; shorter ones are discarded |
| `--model`                 | `htdemucs`      | Demucs model name                                                                    |

---

## 4. Hardware & Environment

Auto-detect compute device in this priority order:

1. **CUDA** — NVIDIA GPU (primary target: Google Colab T4)
2. **MPS** — Apple Silicon
3. **CPU** — Fallback; notify user that processing will be significantly slower

Log which device was selected before any processing begins.

---

## 5. Error Handling & Logging

- Do **not** crash on corrupted or unreadable files. Log the error and continue to the next file.
- Show clear per-file progress:
  ```
  [1/12] Fadhel_Shaker_Ya_Ghayeb.mp3
    🎵  Separating stems...
    🎤  Detecting vocal activity... 2 vocal window(s) found
    🥁  Detecting drum segments within vocal zones... 3 segment(s) found
    💾  Exported: Fadhel_Shaker_Ya_Ghayeb_drums_01.mp3
    💾  Exported: Fadhel_Shaker_Ya_Ghayeb_drums_02.mp3
    💾  Exported: Fadhel_Shaker_Ya_Ghayeb_drums_03.mp3
  ```
- Print a summary at end of batch: total files processed, skipped (no vocals), and errored.

---

## 6. Deliverables

1. **`main.py`** — The main script.
2. **`requirements.txt`** — All Python dependencies with pinned versions.
3. **`README.md`** — Setup instructions covering PyTorch/CUDA for Google Colab and local CPU, plus a CLI argument reference and usage examples.
