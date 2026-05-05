This script assumes familiarity with the following fields:

════════════════════════════════════════
Field A: Voice Activity Detection — Model Behavior and Parameter Calibration
════════════════════════════════════════

Skills required:

1. Deep knowledge of Silero VAD's internal architecture: how it outputs
   speech-probability scores, what its default thresholds mean, and how
   those internals interact with the script's own post-processing logic
2. Understanding of how VAD models behave when fed source-separated audio
   (a clean isolated stem) rather than the real-world mixed audio they
   were trained and benchmarked on
3. Ability to read a function signature and trace which parameters actually
   reach the model versus which are accepted but silently ignored

Questions a domain expert must answer:

1. The parameter --vad-silence-db is accepted by detect_vocal_windows as
   silence_db but is never referenced anywhere inside the function body.
   Silero VAD's internal thresholding alone controls what counts as speech.
   Is the -40 dB intent from the client brief being enforced at all, or is
   that entire parameter dead code?
2. Silero VAD was trained and validated on realistic mixed recordings, not
   on stem-separated vocal tracks. When fed a Demucs vocal stem — which
   is clean but may carry its own bleed artifacts and a different dynamic
   profile than raw recordings — does the model's reliability hold, or
   does it require re-calibration of threshold (currently set to 0.30
   versus the library's typical default of 0.50)?
3. The script runs two stages of merging: Silero's internal
   min_silence_duration_ms=200 produces fine-grained raw timestamps, and
   then the script's own 6-second merger collapses those into vocal zones.
   Does the first stage's 200 ms minimum interfere with detection of the
   very short breath gaps that maqam phrasing produces, potentially
   pre-discarding events before the second stage ever sees them?

────────────────────────────────────────
Field B: RMS Energy Analysis and Audio Segmentation Logic
────────────────────────────────────────

Skills required:

1. Frame-level RMS energy computation — what the frame length and hop
   length actually control, and the relationship between those choices and
   the minimum transient duration that can be resolved
2. dBFS reference conventions and how they translate to real signal levels
   coming out of a neural source-separation model that may not normalize
   its output to full scale
3. Ability to trace state-machine logic for segment boundary detection and
   identify off-by-one or edge-case behaviors at window boundaries

Questions a domain expert must answer:

1. The drum silence threshold is a fixed -40 dBFS applied to every track.
   If Demucs does not normalize its drum stem output to a consistent peak
   level — and there is no evidence in the script that it does — then a
   quiet recording will have its entire drum stem sitting near or below
   -40 dBFS, causing every frame to register as silence. Does the expert
   know whether htdemucs normalizes stem output, and if not, is a
   per-track dynamic threshold needed?
2. The 50 ms frame with 25 ms hop means adjacent frames share half their
   samples. A doumbek hit lasting ~20–30 ms may appear in two consecutive
   frames. In \_split_on_silence, when the signal returns above silence_db
   after a streak, the silence_onset is reset to None but seg_start is
   unchanged. Is there any scenario where a single fast transient
   separated by a very brief sub-threshold gap (shorter than the frame)
   could be misread as a segment boundary rather than continuous activity?
3. At the end of \_split_on_silence, an open segment closes at silence_onset
   if silence has begun, otherwise at win_end. This means a segment whose
   drums fade out gradually — spending several frames near the threshold —
   will be clipped at the first frame that crossed below -40 dBFS rather
   than at the last audible drum hit. For Arabic hand percussion with
   natural resonance tails, how much of the audible decay is being
   systematically cut from every exported segment's tail?

────────────────────────────────────────
Field C: Neural Music Source Separation on Arabic Music
────────────────────────────────────────

Skills required:

1. Knowledge of htdemucs training data composition — specifically which
   genres and instrument types are represented, and which are absent or
   underrepresented
2. Familiarity with how Arabic percussion instruments (doumbek, riq, tabla
   baladi) differ spectrally and rhythmically from the Western drum kit
   patterns that dominate source-separation training sets
3. Understanding of stem bleed patterns in Demucs: which instrument types
   tend to leak into which stems, and whether that bleed is deterministic
   enough to be predicted in advance

Questions a domain expert must answer:

1. htdemucs was trained primarily on Western popular music. When presented
   with a track featuring doumbek or riq as the primary percussion, does
   the model reliably route that energy to the drums stem, or does a
   significant portion land in other — and if it lands in other, the
   script will silently produce empty or incomplete drum segments with
   no warning to the user?
2. Maqam vocal performances often include dense microtonal ornamentation
   that sits in frequency ranges also occupied by certain percussion
   instruments. Does Demucs' vocal stem for Arabic recordings stay clean
   enough that Silero VAD sees only voice activity, or does leaked
   percussion energy in the vocal stem risk producing phantom vocal
   windows during purely instrumental sections?
3. The README recommends htdemucs_ft as an alternative when drum bleed
   into other is heavy. Is this recommendation based on documented
   behavior with Arabic material specifically, or is it generic advice —
   and what is the actual performance delta between htdemucs and
   htdemucs_ft on non-Western percussion in terms of stem isolation
   quality?
