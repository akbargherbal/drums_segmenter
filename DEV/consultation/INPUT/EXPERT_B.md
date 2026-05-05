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

