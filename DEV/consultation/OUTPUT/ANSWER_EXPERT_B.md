### 1. Demucs Normalization and the Fixed -40 dBFS Threshold

**Demucs does not normalize its stem outputs.**
By design, source separation models like Demucs (including the `htdemucs` architecture) are trained to satisfy the condition $\sum \text{stems} \approx \text{mixture}$. The absolute amplitude of the separated drum stem is strictly a function of the drum's original level in the unnormalized mixture.

If an input track is mastered quietly (e.g., peaking at `-15 dBFS`), and the drums sit `-10 dB` below the mix peak, the maximum amplitude of the drum stem will be approximately `-25 dBFS`. If the track is highly dynamic, ghost notes or quieter percussion passes may easily sit at `-45 dBFS`. Applying a fixed `-40 dBFS` threshold to this unnormalized output will cause the state machine to classify active, audible drums as silence.

**A per-track dynamic threshold is strictly required.**
To ensure the state machine behaves consistently across varying input levels, the threshold must be calibrated relative to the stem's actual energy.

**Comparison of Threshold Strategies:**

| Strategy                       | Implementation                         | Failure Mode                                                                         |
| :----------------------------- | :------------------------------------- | :----------------------------------------------------------------------------------- |
| **Fixed dBFS** (Current)       | `threshold = -40.0`                    | Fails on quiet mixes; truncates dynamics on loud mixes.                              |
| **Peak-Relative**              | `threshold = peak_db - 30.0`           | Vulnerable to a single anomalous transient (e.g., a digital click) skewing the peak. |
| **RMS-Relative** (Recommended) | `threshold = median_active_rms - 20.0` | Requires a two-pass analysis or histogramming to find the median active level.       |

**Recommended Implementation (Peak-Relative Fallback):**

```python
# Compute peak dBFS of the drum stem
peak_db = 20.0 * np.log10(np.max(np.abs(audio)) + 1e-9)

# Set threshold relative to the peak, bounded by an absolute floor
dynamic_silence_db = max(peak_db - 35.0, -60.0)
```

---

### 2. State Machine Behavior on Fast Transients and Micro-Gaps

**No, a single fast transient separated by a brief sub-threshold gap will not be misread as a segment boundary.**

The state machine in `_split_on_silence` implements a hysteresis mechanism via the `silence_duration` parameter (defaulting to `1.5` seconds). A segment is only closed if the signal remains continuously below the threshold for this entire duration.

Here is the exact frame-by-frame execution trace demonstrating why a micro-gap (e.g., `50 ms`) between doumbek hits is safely absorbed:

1. **Frame $N$ ($t = 1.000$ s):** Transient peaks. `db = -15.0`.
   - Condition: `db >= silence_db`.
   - Action: `seg_start` is initialized to `1.000`. `silence_onset = None`.
2. **Frame $N+1$ ($t = 1.025$ s):** Signal drops between hits. `db = -45.0`.
   - Condition: `db < silence_db` and `silence_onset is None`.
   - Action: `silence_onset` is set to `1.025`.
3. **Frame $N+2$ ($t = 1.050$ s):** Signal remains low. `db = -42.0`.
   - Condition: `db < silence_db` and `silence_onset` is active.
   - Evaluation: `t - silence_onset` $\rightarrow 1.050 - 1.025 = 0.025$ seconds.
   - Action: $0.025 < 1.5$ (`silence_duration`), so the state machine does nothing. The segment remains open.
4. **Frame $N+3$ ($t = 1.075$ s):** Next transient hits. `db = -12.0`.
   - Condition: `db >= silence_db`.
   - Action: `silence_onset` is reset to `None`. The silence counter is cleared.

Because `silence_onset` is reset to `None` the moment the signal crosses back above the threshold, the `1.5` second requirement is never met. The micro-gap is correctly treated as continuous activity.

---

### 3. Systematic Truncation of Resonance Tails

The current logic systematically **truncates the entire audible decay tail that falls below the `-40 dBFS` threshold**, stripping the natural resonance of the instrument and the room.

Look at the segment closure logic:

```python
if t - silence_onset >= silence_duration:
    segments.append((seg_start, silence_onset))
```

When the state machine confirms that `1.5` seconds of silence have elapsed, it closes the segment at `silence_onset` — the exact timestamp of the _first_ frame that dipped below `-40 dBFS`. The subsequent `1.5` seconds of audio, which were used to satisfy the `silence_duration` condition, are completely discarded.

**Quantifying the Lost Decay:**
If an Arabic hand percussion instrument (like a riq or doumbek) decays exponentially, its dB envelope decreases linearly over time.
Assume a drum hit peaks at `-10 dBFS` and decays at a rate of $60 \text{ dB/second}$.

- The signal crosses the `-40 dBFS` threshold at $t = 0.5$ seconds after the peak.
- The true noise floor of the recording is `-70 dBFS`, which is reached at $t = 1.0$ seconds.
- Because the segment is cut exactly at `silence_onset` (the `-40 dBFS` crossing), **$0.5$ seconds of audible decay (from `-40 dBFS` down to `-70 dBFS`) is systematically deleted** from the end of the exported file.

**Correction:**
To preserve the natural decay, the state machine should include a portion of the confirmed silence window (a "hang time" or "release pad") in the exported segment, up to the `silence_duration` limit.

```python
# Corrected closure logic to preserve the decay tail
if t - silence_onset >= silence_duration:
    # Pad the end by 0.5 seconds (or the full silence_duration) to capture the tail
    tail_pad = 0.5
    seg_end = min(silence_onset + tail_pad, win_end)
    segments.append((seg_start, seg_end))
```
