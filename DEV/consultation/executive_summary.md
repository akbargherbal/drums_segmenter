# Executive Summary: Expert Technical Review
## Automated Drum Stem Extractor — Arabic Music Pipeline

**Review Date:** May 2026
**Experts Consulted:** Three domain specialists covering VAD/model behavior, audio
segmentation, and neural source separation on Arabic music
**Codebase reviewed:** `main.py` and supporting files (Demucs + Silero VAD pipeline)

---

## Overview

Three independent domain experts were commissioned to audit the codebase against
nine specific questions spanning voice activity detection, RMS-based segmentation
logic, and the suitability of HTDemucs for Arabic music. Their findings reveal
**five confirmed bugs or design failures**, one behavioral non-issue, and three
domain-specific risks that are architectural in nature and cannot be fixed by
parameter tuning alone.

The most severe finding is a fundamental architectural mismatch: the pipeline's
entire drum-detection stage is built on the assumption that HTDemucs reliably
routes Arabic percussion to the `drums` stem. It does not. This is the root cause
of the pipeline's primary failure mode on its stated target genre.

---

## Finding 1 — `--vad-silence-db` Is Dead Code

**Severity: High | Field: A (VAD) | Status: Confirmed Bug**

The `--vad-silence-db` CLI parameter (defaulting to `-40 dBFS`) is accepted by
`detect_vocal_windows` as `silence_db` but is never referenced anywhere inside the
function body. Silero VAD does not operate on RMS energy levels; it outputs a
framewise speech-probability score from a neural network. The `-40 dB` intent from
the client brief is entirely unenforced during vocal detection, and the CLI exposes
a control that does nothing, creating a false impression of user configurability.

Note that the drum-detection stage correctly applies its own `drum_silence_db`
parameter via `_compute_rms_db`. The dead-code problem is isolated to the VAD path.

**Action required:** Either remove the parameter from the CLI and documentation, or
implement a meaningful post-processing gate (e.g., an RMS energy pre-filter that
masks frames below `-40 dBFS` before feeding audio to Silero).

---

## Finding 2 — Silero VAD Threshold of 0.30 Is Miscalibrated for Separated Stems

**Severity: High | Field: A (VAD) | Status: Confirmed Risk**

Silero VAD's default onset threshold is `0.50`. The script lowers this to `0.30`,
which was presumably intended to capture faint melismatic tail-ends of Maqam
vocals. However, this introduces a significant false-positive risk that the original
calibration was designed to guard against.

When fed a Demucs vocal stem rather than a raw mixed recording, the model's
statistical context shifts: the absence of a natural noise floor pushes
speech-probability scores higher during genuine speech (often near `1.0`), but
separator bleed artifacts — snare transients and synth hits that Demucs
mistakenly leaves in the vocal stem — also score higher than they would in a
natural acoustic environment. Silero was not trained to classify "Demucs bleed"
as non-speech. By lowering the onset threshold to `0.30`, the script becomes
*more* permissive precisely where it needs to be *more* selective.

**Action required:** The threshold must be empirically calibrated against the
specific artifact profile of `htdemucs` on the target repertoire. A value of
`0.30` is likely too low for reliable bleed rejection; `0.45`–`0.50` is a safer
starting point, with the melismatic tail problem addressed by adjusting
`min_silence_duration_ms` rather than lowering the onset threshold.

---

## Finding 3 — Drum Silence Threshold Fails on Unnormalized Stems

**Severity: High | Field: B (Segmentation) | Status: Confirmed Bug**

Demucs, by design, does not normalize its stem outputs. The separated drum stem's
absolute amplitude is strictly a function of the drum's original level in the
input mixture (the constraint being that all stems sum to the original). The
script applies a fixed `-40 dBFS` threshold to this unnormalized output.

On a quietly mastered recording (e.g., peaking at `-15 dBFS`) where the drum
sits `-10 dB` below the mix peak, the drum stem ceiling is approximately
`-25 dBFS`. Ghost notes, lighter percussion passes, or any hit in a dynamic
passage can easily sit at `-45 dBFS` or below — meaning the state machine
classifies them as silence while they are audibly present. This produces
systematically truncated and fragmented segments on quiet input material with no
warning to the user.

**Action required:** Replace the fixed threshold with a per-track dynamic
threshold computed relative to the stem's actual energy. The recommended
implementation is a peak-relative threshold bounded by an absolute floor:

```python
peak_db = 20.0 * np.log10(np.max(np.abs(audio)) + 1e-9)
dynamic_silence_db = max(peak_db - 35.0, -60.0)
```

A more robust alternative is an RMS-relative threshold (median active RMS minus
20 dB), which is more resistant to single anomalous transients skewing the peak
estimate, at the cost of a two-pass analysis or histogram computation.

---

## Finding 4 — Segment Closure Systematically Truncates Resonance Tails

**Severity: Medium | Field: B (Segmentation) | Status: Confirmed Bug**

When the state machine in `_split_on_silence` confirms a segment boundary, it
closes the segment at `silence_onset` — the timestamp of the *first* frame that
dipped below the threshold. The subsequent `silence_duration` seconds (1.5 s by
default) used to satisfy the silence confirmation window are discarded entirely.

For Arabic hand percussion with natural resonance tails (doumbek, riq, tabla
baladi), the audible decay of a hit extends well below `-40 dBFS`. Assuming a
hit that decays at 60 dB/second and peaks at `-10 dBFS`, the `-40 dBFS`
threshold crossing occurs at ~0.5 seconds, while the true noise floor is not
reached until ~1.0 second. Every exported segment is therefore clipped by
approximately 0.5 seconds of audible resonance at its tail.

**Action required:** Include a configurable hang-time pad in the segment closure
logic to preserve the natural decay:

```python
if t - silence_onset >= silence_duration:
    tail_pad = 0.5  # seconds; expose as CLI parameter
    seg_end = min(silence_onset + tail_pad, win_end)
    segments.append((seg_start, seg_end))
```

---

## Finding 5 — Arabic Percussion Does Not Route to the `drums` Stem (Architectural)

**Severity: Critical | Field: C (Source Separation) | Status: Confirmed — Architectural**

This is the most severe finding and explains what will appear, from the user's
perspective, as the pipeline simply producing empty or rhythmically incoherent
output on Arabic recordings without logging any error.

HTDemucs was trained exclusively on Western popular music. Its learned
definition of "drums" is the modern drum kit: kick drum (sub-bass + beater
click), snare (broadband noise burst), hi-hats (metallic short-decay highs).
When presented with Arabic percussion, its priors misfire systematically:

- **Doumbek _tek_ (600–1200 Hz rim stroke):** Interpreted as a melodic
  transient (staccato guitar, piano hammer, or woodblock). The majority of its
  energy routes to the `other` stem.
- **Doumbek _doum_ (bass tone, 100–200 Hz):** Fragmented — the attack may
  register faintly in `drums` while the resonant tail bleeds into `bass` or
  `other`.
- **Riq jingles (2–12 kHz sustained shimmer):** The model splits the signal.
  A phase-smeared ghost artifact remains in `drums` while the bulk of the
  sustained energy goes to `other`.

**Consequence:** The pipeline looks only at the `drums` stem for percussion
activity. It will silently produce sparse, empty, or rhythmically incoherent
output segments because the rhythmic information is distributed across `drums`
and `other`. No error is raised. The user receives output files with no
indication that the core rhythmic content is missing.

**Action required:** This cannot be fixed by parameter tuning or model switching.
The only viable remediation strategies are:
1. **Composite analysis:** Reconstruct the rhythmic picture by scanning both
   the `drums` and `other` stems simultaneously and merging their activity maps
   before applying the segmentation logic.
2. **Custom model:** Fine-tune an existing architecture (Open-Unmix, BandSplit
   RNN, or HTDemucs itself) on an annotated dataset of Arabic stems with doumbek
   and riq ground truth.

Switching to `htdemucs_ft` (noted in the README as a recommended alternative for
heavy bleed) does not address this problem. The `_ft` model's fine-tuning
improved audio fidelity and reduced phase-smearing artifacts but did not change
the model's fundamental taxonomy of what constitutes a "drum." The routing of
Arabic percussion to `other` is equally severe under both models.

---

## Finding 6 — Phantom Vocal Windows from Instrumental Bleed

**Severity: Medium | Field: C (Source Separation) + A (VAD) | Status: Confirmed Risk**

During purely instrumental sections (e.g., a *taqsim* or instrumental *lazma*
between sung verses), Silero VAD may detect phantom vocal windows due to
instrument bleed into the Demucs vocal stem.

The primary culprits are not percussion instruments (whose bleed tends to be
broadband noise, which Silero largely ignores) but rather **melodic ensemble
instruments**: the *nay* (oblique flute), Arabic-style violin with continuous
glissandi, and the mid-range of the *oud*. These instruments are highly harmonic
and carry spectral envelopes that closely mimic human vowel formants. Because
HTDemucs' vocal model was not trained on Maqam vocal techniques, its masking is
looser on these out-of-distribution signals, and reciprocal bleed from these
harmonic instruments into the vocal stem is a common failure mode.

**Action required:** Post-VAD validation using a secondary energy check on the
raw (unseparated) mix within detected vocal windows, or a confidence-gating step
that discards windows where the vocal stem's spectral centroid falls outside the
expected human voice range.

---

## Finding 7 — Two-Stage Merging Logic Is Correct (Non-Issue)

**Severity: None | Field: A (VAD) | Status: Verified — No Action Required**

A concern was raised about whether Silero's internal `min_silence_duration_ms=200`
could pre-discard short breath gaps characteristic of Maqam phrasing before the
script's own 6-second macro-merger processes them. This concern is unfounded.

The `min_silence_duration_ms` parameter controls the *minimum gap duration* required
to produce a split in the raw output. Gaps shorter than 200 ms are bridged
automatically — the segment is kept open, not closed. The 200 ms micro-merge is
then entirely subsumed by the script's 6-second macro-merge in the second pass.
The two stages do not conflict; they form a coherent two-tier hierarchy
(micro-bridge → macro-bridge) with no data loss between them.

---

## Finding 8 — Fast Transient / Micro-Gap Edge Case Is Safe (Non-Issue)

**Severity: None | Field: B (Segmentation) | Status: Verified — No Action Required**

A concern was raised about whether a doumbek hit lasting 20–30 ms, spanning two
consecutive 50 ms frames due to the 25 ms hop overlap, could be misread as a
segment boundary if the inter-hit gap is shorter than one frame. The state machine
is safe. A segment boundary is only emitted after `silence_duration` (1.5 s) of
continuous sub-threshold frames. A single or double-frame sub-threshold dip resets
`silence_onset` to `None` the moment the signal crosses back above the threshold,
and the 1.5-second counter never reaches completion. No false boundary is produced.

---

## Summary Table

| # | Finding | Severity | Status | Action Required |
|---|---------|----------|--------|-----------------|
| 1 | `--vad-silence-db` is dead code | High | Confirmed Bug | Remove or implement |
| 2 | VAD threshold 0.30 too permissive for stems | High | Confirmed Risk | Recalibrate to ~0.45–0.50 |
| 3 | Fixed -40 dBFS drum threshold fails on unnormalized stems | High | Confirmed Bug | Implement dynamic threshold |
| 4 | Segment closure truncates resonance tails | Medium | Confirmed Bug | Add configurable tail pad |
| 5 | Arabic percussion routes to `other`, not `drums` | **Critical** | Architectural | Composite stem analysis or custom model |
| 6 | Phantom vocal windows from harmonic instrument bleed | Medium | Confirmed Risk | Add post-VAD spectral validation |
| 7 | Two-stage merge logic conflict | None | Non-issue | No action required |
| 8 | Fast transient micro-gap misread | None | Non-issue | No action required |

---

## Prioritized Remediation Roadmap

**Immediate (pre-release blockers):**
- Fix Finding 3: dynamic drum silence threshold
- Fix Finding 1: remove or implement `--vad-silence-db`
- Acknowledge Finding 5 in the README; warn users that Arabic percussion results
  will be incomplete with the current architecture

**Short-term (next iteration):**
- Fix Finding 4: tail pad on segment closure
- Investigate Finding 2: empirically calibrate VAD threshold against htdemucs
  artifact profile on target repertoire
- Prototype Finding 5 mitigation: composite `drums + other` stem analysis

**Medium-term (architectural):**
- Address Finding 5 fully: evaluate BandSplit RNN or custom fine-tuning on
  annotated Arabic stems
- Address Finding 6: implement post-VAD spectral confidence gate

---

*This summary was compiled from expert reviews in Fields A (VAD & parameter calibration), B (RMS energy & segmentation logic), and C (neural source separation on Arabic music). Experts reviewed `main.py` and the client brief directly.*
