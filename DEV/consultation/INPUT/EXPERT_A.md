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
