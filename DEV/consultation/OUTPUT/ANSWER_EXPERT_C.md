Here are the precise, domain-expert answers to your questions regarding the behavior of HTDemucs and downstream processing tools when applied to Arabic music.

### 1. Percussion Routing: Does doumbek/riq energy reliably route to the `drums` stem?

**No, it does not.** When a script relies on the HTDemucs `drums` stem to capture Arabic percussion, it will inevitably suffer from severe, silent data loss.

The fundamental issue is a taxonomic and spectral mismatch in the training data. HTDemucs was trained to isolate the modern Western drum kit. It expects a kick drum (heavy sub-bass below 80 Hz, sharp beater click), a snare (broadband noise burst, 200 Hz fundamental), and hi-hats/cymbals (metallic, often short-decay high frequencies).

When presented with a doumbek and riq, the model's learned priors misfire predictably:

- **The Doumbek _Tek_:** The high-pitched rim stroke (_tek_) sits in the 600–1200 Hz range with complex, ringing overtones and lacks the broadband noise burst of a snare drum. HTDemucs frequently interprets this as a melodic transient (similar to a staccato guitar chord, piano hammer, or woodblock) and routes a massive portion of this energy directly into the `other` stem.
- **The Doumbek _Doum_:** The open bass tone (_doum_) has a fundamental around 100–200 Hz but lacks the sub-bass extension of a Western kick drum. It is often fragmented—the attack might register faintly in `drums`, while the resonant tail bleeds into `bass` or `other`.
- **The Riq:** While the skin hits might partially route to `drums`, the continuous, dense 16th-note shimmer of the brass jingles (2–12 kHz) often confuses the model. Because it sustains much longer than a typical closed hi-hat, the model often splits it, leaving a ghostly, phase-smeared artifact in the `drums` stem while dumping the bulk of the sustained high-frequency energy into `other`.

**Consequence for automated scripts:** If your pipeline looks only at the `drums` stem to calculate percussion activity, it will miss the _teks_, _kas_, and riq subdivisions entirely. The script will silently produce empty, sparse, or rhythmically incoherent drum segments without throwing any errors, because it assumes the `drums` stem contains the totality of the rhythmic information.

### 2. Vocal Stem Bleed and Silero VAD: Do phantom vocal windows occur?

**Yes, phantom vocal windows are a high-probability risk**, but the primary culprit is usually not the percussion—it is the traditional Arabic melodic ensemble (_takht_) bleeding into the vocal stem, compounded by how Silero VAD interprets that bleed.

Here is the mechanical breakdown of what happens:

- **Vocal Model Mismatch:** HTDemucs' vocal model expects Western vocal formants, dynamics, and phrasing. Maqam-based vocals feature dense microtonal ornamentation, extended melismas, and glottal attacks that the model struggles to track perfectly.
- **Reciprocal Bleed:** Because the model is unsure of these out-of-distribution vocal techniques, it loosens its masking. Consequently, instruments that occupy the same frequency range and mimic vocal phrasing—specifically the _nay_ (oblique flute), the _violin_ (played in the Arabic style with continuous glissandi), and the mid-range of the _oud_—frequently leak into the vocal stem.
- **Silero VAD's Vulnerability:** Silero VAD is a neural network trained to detect speech by looking for harmonic stacks (formants) and specific temporal envelopes. While leaked percussion (like a smeared riq jingle) usually registers as broadband noise and is ignored by Silero, **leaked _nay_ or _violin_ is highly harmonic and mimics the spectral envelope of human vowels.**

Therefore, during purely instrumental sections (such as a _taqsim_ or an instrumental _lazma_ between vocal verses), the leaked harmonic energy of the strings or flute in the Demucs vocal stem will frequently trick Silero VAD into triggering false positives, resulting in phantom vocal windows.

### 3. `htdemucs_ft` vs. `htdemucs`: Is the recommendation generic or Arabic-specific?

The recommendation in the README to use `htdemucs_ft` (the fine-tuned model) for heavy drum bleed is **entirely generic advice**. It is not based on documented behavior with Arabic material, nor was `htdemucs_ft` fine-tuned on a corpus of MENA-region music.

The `_ft` model was fine-tuned on a larger, slightly more diverse internal dataset from Deezer to improve overall Signal-to-Distortion Ratio (SDR) and reduce phase artifacts.

**The actual performance delta on non-Western percussion:**

- **Audio Quality (Artifacts):** `htdemucs_ft` is generally better at preserving transients and reducing the "underwater" phase-smearing artifacts that occur when the base model struggles to separate overlapping frequencies. If a riq is playing over a dense string section, `htdemucs_ft` will yield a slightly crisper sound.
- **Taxonomic Routing (Stem Isolation):** The performance delta here is **negligible**. Fine-tuning did not change the model's fundamental definition of what constitutes a "drum." The model still believes "drums = drum kit." Therefore, `htdemucs_ft` will _still_ route the doumbek _tek_ and the riq shimmer into the `other` stem. It will just do so with slightly higher audio fidelity.

If your goal is to cleanly isolate Arabic percussion from a mix, switching to `htdemucs_ft` will not solve the problem. The only true solutions are either training a custom model (or fine-tuning an existing architecture like Open-Unmix or BandSplit RNN) on an annotated dataset of Arabic stems, or using post-processing heuristics that analyze both the `drums` and `other` stems simultaneously to reconstruct the full rhythmic composite.
