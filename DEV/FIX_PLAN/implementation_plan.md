# Arabic Drum Stem Extractor: Remediation Implementation Plan

## Executive Summary

- **Current State**: Working pipeline with 5 confirmed bugs/risks and 1 critical architectural flaw identified by expert review
- **Goal**: Remediate all findings from the expert audit, culminating in reliable Arabic percussion detection via composite stem analysis
- **Key Architectural Decision**: The pipeline's drum detection stage must be rebuilt around a composite `drums + other` activity map — parameter tuning alone cannot fix the root failure on Arabic music
- **Estimated Time**: 6–11 days across 4 phases (phases are independent and shippable)

---

## Phase 1: Pre-Release Bug Fixes (1–2 days)

### Goal
**Eliminate the two confirmed high-severity bugs that corrupt output silently and without warning.**

These are the only changes needed before the tool can be honestly described as working. They touch isolated functions and carry no architectural risk.

### Success Criteria
- ✅ `--vad-silence-db` either has a visible effect on output OR is removed from the CLI and README
- ✅ Dynamic drum threshold computes correctly on a track mastered at `-15 dBFS` (verify with a print/log statement showing computed threshold ≠ `-40`)
- ✅ All existing outputs for previously-passing Western tracks are bit-identical or acceptably equivalent after both changes
- ✅ `git diff` shows zero changes outside `detect_vocal_windows()`, `_compute_rms_db()`, `detect_drum_segments()`, and the CLI block

### Tasks

**1.1: Fix Finding 1 — Remove or implement `--vad-silence-db`** (2–4 hours)

The `silence_db` parameter is accepted by `detect_vocal_windows()` but never referenced in its body. Silero VAD does not operate on RMS energy — it emits a speech-probability score — so the parameter cannot be wired to Silero directly. Choose one path:

- **Path A (recommended — remove):** Delete `--vad-silence-db` from the `@click.command` decorator, from the `main()` signature, from the `process_file()` call site, and from the README CLI reference table. Add a one-line code comment in `detect_vocal_windows()` explaining why an energy gate was not applied.
- **Path B (implement):** Add an RMS pre-filter that zeros out audio frames below `silence_db` before the tensor is passed to Silero. Use the existing `_compute_rms_db()` helper.

```python
# Path B sketch — implement an RMS mask before Silero sees the audio
def _apply_energy_gate(audio: np.ndarray, sr: int, silence_db: float) -> np.ndarray:
    """Mute frames whose RMS falls below silence_db before VAD inference.
    This gives --vad-silence-db a real effect on Silero's input signal."""
    pass  # Implement during session: frame-level mask, apply to audio tensor
```

Path A is safer and faster. Path B only makes sense if the user actively relies on `--vad-silence-db` as a workflow parameter. Pick A unless there is a concrete reason not to.

**Key Decision:** Document the chosen path explicitly in a `# FINDING-1` comment at the call site in `detect_vocal_windows()` so the next developer understands the intent.

**1.2: Fix Finding 3 — Dynamic drum silence threshold** (2–4 hours)

`detect_drum_segments()` passes the user-supplied `silence_db` (default `-40 dBFS`) directly to `_split_on_silence()`. Because Demucs does not normalize its stem outputs, this fixed threshold silently truncates segments on quietly-mastered tracks.

Replace the fixed value with a per-stem dynamic computation immediately after loading the drum audio:

```python
def _compute_dynamic_silence_db(audio: np.ndarray, silence_db_override: float) -> float:
    """Return a stem-relative silence threshold.
    Falls back to a conservative absolute floor to avoid over-triggering on noise.
    The user-supplied silence_db is used as a floor guard, not the primary value."""
    pass  # Implement: peak_db = 20*log10(max(abs(audio)) + 1e-9)
          #            dynamic = max(peak_db - 35.0, -60.0)
          #            log computed value for transparency
```

Apply this in `detect_drum_segments()` before the per-window loop. Log the computed value at DEBUG level so the user can verify it when debugging quiet tracks.

**Important:** The user-supplied `--drum-silence-db` should be retained in the CLI and used as a manual override that replaces the dynamic value when explicitly set (i.e., when the user passes a value other than the default). Implement this as a sentinel: if `drum_silence_db == DEFAULT_DRUM_SILENCE_DB`, use the dynamic threshold; if the user passed a custom value, use theirs.

### Deliverables
- [ ] `detect_vocal_windows()`: `silence_db` parameter either removed or wired to a functional energy gate
- [ ] CLI, `process_file()`, and README updated to reflect Finding 1 resolution
- [ ] `_compute_dynamic_silence_db()` function implemented and called from `detect_drum_segments()`
- [ ] Dynamic threshold logged at DEBUG level for each processed file
- [ ] Manual test: run on one quiet track (or simulate by normalizing a test WAV to `-15 dBFS`) and confirm threshold ≠ `-40`
- [ ] Git commit at phase boundary: `fix: Finding 1 dead VAD param, Finding 3 dynamic drum threshold`

### Rollback Plan
**If** dynamic threshold produces drastically different segment counts on previously-working tracks: revert `_compute_dynamic_silence_db()` to `return silence_db` (a no-op) and re-examine the peak estimate — the `max(abs(audio))` approach is sensitive to single transient spikes; switch to median-based active RMS as an alternative.

---

## Phase 2: Calibration & Tail-Pad Fixes (1.5–3 days)

### Goal
**Correct two parameter-level errors that produce audibly wrong output: over-permissive VAD causing phantom detections, and segment closure that clips resonance tails.**

Both changes require empirical validation against real Arabic recordings. Time estimates assume access to a representative test set of 5–10 tracks.

### Success Criteria
- ✅ VAD threshold raised to `0.45` (or empirically determined value) — phantom windows during a purely instrumental taqsim section are eliminated on ≥3 test tracks
- ✅ Threshold change does not drop true-positive vocal window count on the same test tracks (same window count ± 1)
- ✅ Tail pad of `0.3–0.5 s` is audible in exported segments — doumbek decay is no longer clipped at the point of threshold crossing
- ✅ `--vad-threshold` and `--drum-tail-pad` appear in `--help` output and README

### Tasks

**2.1: Fix Finding 2 — Recalibrate Silero VAD threshold** (3–5 hours + validation time)

The hardcoded `threshold=0.30` in `get_speech_timestamps()` is too permissive for separated Demucs stems, where bleed artifacts score higher than they would in a natural acoustic mix.

Promote the threshold to a CLI parameter so it can be empirically calibrated without code changes:

```python
def detect_vocal_windows(
    vocal_path: Path,
    vad_model,
    vad_utils,
    silence_db: float,
    silence_duration: float,
    device: torch.device,
    vad_threshold: float = 0.45,  # New: was hardcoded 0.30
) -> list[tuple[float, float]]:
    """... vad_threshold: Silero onset probability threshold.
    Start at 0.45 for htdemucs stems; lower only if melismatic tails are missed."""
    pass  # Replace hardcoded 0.30 with vad_threshold
```

Wire `--vad-threshold` through the Click CLI → `main()` → `process_file()` → `detect_vocal_windows()`. Default to `0.45`.

**Validation protocol:** Run on ≥3 tracks that include both a vocal section and a purely instrumental section (taqsim or lazma). Confirm that the instrumental section produces zero or near-zero vocal windows. If melismatic tail ends are now being missed, adjust `min_silence_duration_ms` upward (try `300ms → 500ms`) before lowering the threshold.

**2.2: Fix Finding 4 — Tail pad on segment closure** (2–3 hours)

`_split_on_silence()` closes a segment at `silence_onset` — the first frame that dips below threshold — discarding the natural resonance decay of Arabic hand percussion. The fix is a configurable hang-time pad:

```python
def _split_on_silence(
    times: np.ndarray,
    rms_db: np.ndarray,
    silence_db: float,
    silence_duration: float,
    win_start: float,
    win_end: float,
    tail_pad: float = 0.4,   # New: seconds of decay to include after threshold crossing
) -> list[tuple[float, float]]:
    """... tail_pad: seconds to extend each segment past the silence_onset point.
    Preserves doumbek/riq resonance tails that decay below the threshold."""
    pass  # Change: seg_end = min(silence_onset + tail_pad, win_end)
          # instead of: segments.append((seg_start, silence_onset))
```

Wire `--drum-tail-pad` through the same chain. Default to `0.4` seconds. The existing `--drum-min-segment` filter still applies after the pad is added, so very short hits are not accidentally promoted.

**Note:** The final segment in each window (the one closed at `win_end`) already includes the full tail by definition. Apply the pad only to segments closed by a confirmed silence confirmation.

### Deliverables
- [ ] `--vad-threshold` CLI parameter wired end-to-end, default `0.45`
- [ ] `--drum-tail-pad` CLI parameter wired end-to-end, default `0.4`
- [ ] Validation log: 3+ test tracks showing phantom window elimination at `0.45`
- [ ] Audible verification: exported segment for doumbek hit includes perceptible decay tail
- [ ] README CLI reference table updated with both new parameters
- [ ] Git commit: `feat: Finding 2 VAD threshold param, Finding 4 resonance tail pad`

### Rollback Plan
**If** `vad_threshold=0.45` causes real vocal windows to be missed on a meaningful fraction of tracks (>20% of windows): try `0.40` before going lower. If melismatic tails are consistently missed below `0.45`, the correct fix is to raise `min_silence_duration_ms` to `400–600ms` rather than lowering the threshold.

**If** the tail pad causes two adjacent segments to overlap (pad extends into the next segment's start): clamp `seg_end = min(silence_onset + tail_pad, next_seg_start - 0.01)`. Implement overlap detection as a post-processing pass if needed.

---

## Phase 3: Composite Stem Analysis for Arabic Percussion (2.5–4 days)

### Goal
**Resolve the critical architectural finding: Arabic percussion (doumbek tek, doum, riq) routes primarily to the `other` stem, not `drums`, making the pipeline produce empty or incoherent output on Arabic recordings silently.**

This phase is the largest change. It restructures `detect_drum_segments()` to merge activity maps from both the `drums` and `other` stems before segmentation.

### Success Criteria
- ✅ A test track with doumbek-only percussion (no Western kit) produces at least one non-empty drum segment (currently produces zero)
- ✅ A Western-pop track produces the same segments as before (within ± 1 segment, same boundaries ± 0.5s) — the composite analysis must not degrade Western performance
- ✅ A `--percussion-mode` flag controls the behavior; `western` uses drums-only (legacy), `arabic` uses composite, `auto` is the new default
- ✅ No new dependency added — `other.wav` is already separated and cached by Demucs in Phase 1
- ✅ A clear warning is logged when `arabic` or `auto` mode routes to composite analysis, so the user understands what changed

### Tasks

**3.1: Expose the `other` stem path** (1–2 hours)

`separate_stems()` already returns paths for all four stems including `other`. However, `process_file()` currently only passes `stems["drums"]` to `detect_drum_segments()`. Pass `stems["other"]` as well:

```python
def detect_drum_segments(
    drum_path: Path,
    other_path: Path | None,         # New: None = legacy western-only mode
    vocal_windows: list[tuple[float, float]],
    silence_db: float,
    silence_duration: float,
    min_segment: float,
    tail_pad: float,
    percussion_mode: str = "auto",   # New: 'western' | 'arabic' | 'auto'
) -> list[tuple[float, float]]:
    """... other_path: when provided, merge drums+other RMS before segmentation.
    percussion_mode: 'western' ignores other_path; 'arabic' requires it;
    'auto' uses composite when other stem energy exceeds drums stem energy."""
    pass
```

**3.2: Implement RMS activity map merging** (4–6 hours)

The core of the composite approach: compute per-frame RMS in dB for both stems, then combine them into a single activity map before passing to `_split_on_silence()`.

```python
def _merge_percussion_activity(
    drums_audio: np.ndarray,
    other_audio: np.ndarray,
    sr: int,
    drums_weight: float = 1.0,
    other_weight: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine drums and other stems into a single RMS-dB activity trace.
    Returns (times, merged_rms_db) ready for _split_on_silence().
    Weights: drums gets full weight; other is down-weighted to reduce melodic
    instrument bleed (nay, violin, oud) from triggering false segments."""
    pass  # Compute RMS for each stem, convert to linear power, weighted sum,
          # convert back to dB. Do NOT average in dB domain — average in power domain.
```

**Key Decision — weighting:** The `other` stem contains both Arabic percussion (wanted) and melodic instruments (unwanted bleed). A weight of `0.7` for `other` is a starting heuristic. The correct value must be determined empirically on the test set. Expose `--other-stem-weight` as an advanced CLI parameter with default `0.7` and document the trade-off clearly.

**3.3: Implement `auto` mode heuristic** (2–3 hours)

For `percussion_mode='auto'`, decide at runtime whether to use composite analysis by comparing the total energy in `drums` vs `other` within the vocal windows:

```python
def _should_use_composite(
    drums_audio: np.ndarray,
    other_audio: np.ndarray,
    vocal_windows: list[tuple[float, float]],
    sr: int,
    ratio_threshold: float = 1.5,
) -> bool:
    """Return True if other stem has comparable energy to drums within vocal zones.
    When other_energy / drums_energy > ratio_threshold, Arabic routing is likely."""
    pass  # Compute RMS energy within each vocal window for each stem, compare ratio
```

Log the decision: `"🥁  Percussion mode: COMPOSITE (other/drums energy ratio: 2.3x)"` or `"🥁  Percussion mode: DRUMS-ONLY (other/drums energy ratio: 0.4x)"`.

**3.4: Update export to use composite activity, not composite audio** (1 hour)

Critically: the composite activity map is used only to **detect segment boundaries**. The actual exported audio must still come from `stems["drums"]` only — the `other` stem contains melodic instruments that must not appear in the drum export.

```python
# In process_file() Step 5 — no change needed:
export_segment(stems["drums"], seg_start, seg_end, out_path)
# stems["other"] is used for detection only, never for export
```

Add a comment to this effect at the export call site to prevent future confusion.

**3.5: Update README** (1–2 hours)

Add a dedicated "Arabic Percussion Note" section that replaces the current misleading guidance ("try `htdemucs_ft` for heavy bleed"). Document the composite mode behavior, the `--percussion-mode` flag, and the `--other-stem-weight` parameter. Explicitly state that `htdemucs_ft` does not fix the Arabic routing problem.

### Deliverables
- [ ] `detect_drum_segments()` signature updated with `other_path` and `percussion_mode` params
- [ ] `_merge_percussion_activity()` implemented (power-domain weighted sum)
- [ ] `_should_use_composite()` implemented and logging ratio at INFO level
- [ ] `--percussion-mode` and `--other-stem-weight` CLI parameters, wired end-to-end
- [ ] Test: doumbek-only track produces ≥1 non-empty segment
- [ ] Test: Western pop track segment output unchanged (same count and boundaries)
- [ ] Export call confirmed to use `stems["drums"]`, never `stems["other"]`
- [ ] README Arabic Percussion Note section updated
- [ ] Git commit: `feat: Finding 5 composite drums+other percussion analysis`

### Rollback Plan
**If** `auto` mode incorrectly switches to composite on a Western track, degrading output: lower `ratio_threshold` in `_should_use_composite()` (try `2.0`). If this is insufficient, add a genre hint flag (`--genre arabic|western`) as a manual override.

**If** melodic instrument bleed from `other` creates spurious segments even at `other_weight=0.7`: lower the weight further (try `0.5`) or implement a spectral centroid pre-filter that suppresses `other` frames whose centroid falls in the melodic instrument range (>3 kHz sustained) before merging. This is the lightweight precursor to Finding 6's full spectral gate.

---

## Phase 4: Post-VAD Spectral Confidence Gate (1.5–2.5 days, CONDITIONAL)

### Goal
**Eliminate phantom vocal windows caused by harmonic instrument bleed (nay, Arabic violin, oud) into the Demucs vocal stem during purely instrumental sections.**

This phase is conditional: execute it only if Phase 2's threshold recalibration (Finding 2) fails to sufficiently reduce phantom detections on the target repertoire. Do not implement this before validating Phase 2 on real tracks.

### Success Criteria
- ✅ Instrumental taqsim sections produce zero vocal windows on ≥4 of 5 test tracks after the gate
- ✅ Real vocal windows are not suppressed — true-positive count is unchanged
- ✅ Gate processing adds <5 seconds per track on CPU
- ✅ `--vad-spectral-gate` flag enables/disables the gate (default `False` — off until validated)

### Tasks

**4.1: Implement spectral centroid gate** (4–6 hours)

After Silero produces merged vocal windows, validate each window by checking whether the vocal stem's dominant spectral energy falls within the expected human voice range. The nay, violin, and oud produce high-centroid signals even when their energy mimics vocal formants:

```python
def _filter_phantom_windows(
    vocal_path: Path,
    windows: list[tuple[float, float]],
    sr: int,
    min_voice_centroid_hz: float = 150.0,
    max_voice_centroid_hz: float = 3500.0,
    min_voice_fraction: float = 0.6,
) -> list[tuple[float, float]]:
    """Discard windows where spectral centroid is outside human voice range.
    min_voice_fraction: fraction of frames that must pass to keep window.
    A window where >40% of frames have centroid >3500 Hz is likely instrument bleed."""
    pass  # librosa.feature.spectral_centroid() on vocal stem slice per window
          # Compute fraction of frames within [min, max] range
          # Keep window if fraction >= min_voice_fraction
```

**4.2: Wire as optional post-processing step** (1–2 hours)

```python
def detect_vocal_windows(..., spectral_gate: bool = False) -> list[tuple[float, float]]:
    """... spectral_gate: if True, run post-VAD spectral centroid filter."""
    merged = [...]  # existing merge logic
    if spectral_gate:
        merged = _filter_phantom_windows(vocal_path, merged, sr=16_000)
        log.info(f"    🎤  Spectral gate: {before} → {len(merged)} window(s)")
    return merged
```

Keep the gate off by default (`--vad-spectral-gate` flag, default `False`) until its behavior is validated on the full test set. This prevents regressions on Western material.

### Deliverables
- [ ] `_filter_phantom_windows()` implemented using librosa spectral centroid
- [ ] `--vad-spectral-gate` CLI flag, default `False`
- [ ] Validation: 5 tracks with instrumental sections — gate eliminates phantom windows without dropping real ones
- [ ] Gate is logged at INFO level (before/after window count)
- [ ] Git commit: `feat: Finding 6 post-VAD spectral confidence gate`

### Rollback Plan
**If** the gate suppresses real vocal windows (false negatives): widen the centroid bounds (`max_voice_centroid_hz` to `4500 Hz`) or lower `min_voice_fraction` (to `0.5`). If false negatives persist, disable the gate via the CLI flag and document the limitation.

---

## Decision Tree & Stop Conditions

```
START
  │
  ▼
PHASE 1: Dead-code fix + dynamic threshold (no validation needed — pure bugs)
  ├─ Both fixes verified → PHASE 2
  └─ Dynamic threshold destabilizes Western tracks → Fix threshold formula, then PHASE 2

PHASE 2: VAD threshold + tail pad (requires test tracks)
  ├─ Phantom windows eliminated AND tails preserved → PHASE 3
  ├─ Phantom windows persist → Adjust min_silence_duration_ms, re-test, then PHASE 3
  └─ Melismatic tails still clipped → Increase tail_pad, re-test, then PHASE 3

PHASE 3: Composite stem analysis (architectural — primary deliverable)
  ├─ Arabic percussion detected AND Western tracks unchanged → PHASE 4 (conditional)
  ├─ Melodic bleed creates false Arabic segments → Lower other_weight, re-test
  └─ auto-mode misclassifies genre → Add manual --genre flag

PHASE 4: Spectral gate (only if Phase 2 leaves residual phantom windows)
  ├─ Phantom windows eliminated AND no false negatives → DONE
  └─ False negatives (real windows suppressed) → Loosen centroid bounds or SKIP
```

### Explicit Stop Conditions
**STOP if:**
- Phase 3 composite analysis degrades Western track output by >20% after weight tuning — reassess whether `other_weight` gating is sufficient or if a spectral pre-filter on `other` is needed before merging
- Phase 3 composite produces more total segments than ground truth on Arabic tracks (over-detection) — the weight or the energy ratio threshold for `auto` mode needs recalibration
- Phase 4 introduces false negatives on >1 in 5 vocal tracks — disable gate by default permanently

---

## Risk Mitigation Summary

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dynamic threshold (Finding 3) over-triggers on quiet noise floor | Low | Medium | Add absolute floor at `-60 dBFS`; expose as `--drum-floor-db` if needed |
| Composite `other` weight causes melodic instrument bleed in segment detection | Medium | Medium | Start at `0.7`, tune empirically; add `--other-stem-weight` CLI param |
| VAD threshold raise (0.45) misses melismatic tail-ends | Medium | Low | Compensate via `min_silence_duration_ms` increase, not threshold reduction |
| `auto` percussion mode misclassifies Western tracks as Arabic | Low | Medium | Energy ratio threshold tunable; manual `--percussion-mode western` override available |
| Spectral gate (Phase 4) suppresses real vocals with high harmonic content | Medium | High | Gate off by default; validate on full test set before enabling |
| Tail pad causes adjacent segment overlap | Low | Low | Clamp `seg_end` to next segment start minus 10ms in post-processing pass |

---

## Success Metrics

### Minimum Viable Success (Phases 1–3)
- ✅ `--vad-silence-db` dead code resolved (remove or implement)
- ✅ Quiet-track drum detection no longer silently truncated (dynamic threshold)
- ✅ Doumbek/riq percussion detected in Arabic recordings (composite stems)
- ✅ Western pop tracks produce equivalent output to pre-remediation baseline

### Stretch Goals (Phase 4 + beyond)
- Phantom vocal window elimination via spectral gate
- Empirically tuned VAD threshold documented with specific track results
- Custom fine-tuning of HTDemucs on annotated Arabic stems (longer-term; not in this plan's scope)

---

## Scope Boundaries

### In Scope
- ✅ All six action-required findings from the expert report (Findings 1–4, 6 + architectural Finding 5)
- ✅ CLI parameter additions for newly-exposed controls
- ✅ README corrections (removing the misleading `htdemucs_ft` Arabic drum guidance)
- ✅ Preservation of all existing behavior on Western music

### Out of Scope
- ❌ Custom model fine-tuning on Arabic stems (requires annotated dataset and significant GPU time — separate project)
- ❌ BandSplit RNN or Open-Unmix evaluation (valid long-term path but not a code change; separate feasibility study)
- ❌ Changes to the Demucs separation stage, caching logic, or export format
- ❌ Findings 7 and 8 (verified non-issues; no action required)
- ❌ GUI, notebook, or API interface (out of original project scope)

---

## Next Steps

1. **Start Phase 1.1** — open `detect_vocal_windows()`, confirm `silence_db` is unreferenced, choose Path A or B, make the change, and verify via `--help` or a grep
2. **Start Phase 1.2** — implement `_compute_dynamic_silence_db()`, test on one track mastered quietly, log the computed value
3. **Before Phase 2** — prepare a test set of 5–10 Arabic tracks that include at least one purely instrumental section per track; this is required for empirical VAD threshold calibration
4. **Commit at each phase boundary** — each phase produces a working, shippable increment; do not bundle phases into a single PR
