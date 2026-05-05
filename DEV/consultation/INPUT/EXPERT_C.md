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
