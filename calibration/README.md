# Calibration

The scripts that produced the two shipped constants. They are research scripts —
short, direct, and they print tables. They read two environment variables:

```bash
export KIKIRI_ROOT=/path/to/kikiri-tts       # needs dataset/ and StyleTTS2/
export ALBA_WORK=/path/to/working/dir        # holds config.json, alba_stock.pth, packs
```

`KIKIRI_ROOT` is needed for the real recordings (`dataset/audio/`), their
transcripts (`dataset/metadata.csv`), the validation split, and StyleTTS2's JDCNet
pitch extractor (`StyleTTS2/Utils/JDC/bst.t7`) — the same extractor training used,
which is what makes the comparison like-for-like.

## The method

Every number here comes from the same shape of experiment: **synthesize the exact
sentences the speaker actually recorded, then compare the two through one
extractor.** Not "does the synthesis sound good" but "how does it differ from her,
in Hz, on the same words".

| script | question |
|---|---|
| `synth_pairs.py` | build the REAL/SYNTH pairs — everything else reads these |
| `pitch_cmp.py` | is the synth pitched higher than she is, and by how much |
| `scale_sweep.py` | render the voicepack at several prosodic-magnitude scales |
| `sweep_measure.py` | which scale's median pitch matches hers |
| `qc.py` | did the scaling damage anything (duration, zcr, rms, peak) |
| `speedtest.py` | render a speed sweep |
| `declination.py`, `decl2.py` | does the pitch trail down at phrase end the way hers does |
| `f0_probe.py` | the speaker's own pitch distribution — what an F0 loss number means |
| `make_clips.py` | render a fixed set of sentences for A/B listening |

## Things these measurements taught us

**The prosodic half of the voicepack is the pitch knob.** A pack is `[510, 1, 256]`:
dims 0–127 timbre, 128–255 prosody. Scaling the prosodic half moves pitch level
almost linearly in semitones. `tools/apply_pitch_scale.py` does the scaling.

**Calibrate against the speaker, not against stock.** Stock Kokoro's prosodic norm
is 0.367. Matching it here (scale 0.2) *overshoots downward* by 0.64 semitones. The
speaker's measured median is the only target that means anything.

**Scaling costs contour rate.** Going 1.0 → 0.35 moved declination slope from −43.8
to −25.7 Hz/s. The fixed pack drops the right *total* (−63.9 Hz against her −65.3)
but at roughly half the *rate* — it falls lazily, spread thin. That is a real cost
of the calibration, paid knowingly.

**The speed sweep is non-monotonic.** 1.16 was worse on declination than both 1.00
and 1.25. Compressing time does not simply steepen the fall. Eight clips may be too
few to say more; do not build a theory on it.

**F0 loss is not comparable across speakers.** `train_second.py` computes
`F.l1_loss(F0_real, F0_fake) / 10` on the extractor's raw Hz output — no log, no
normalization, no per-speaker scaling. So the loss is mean absolute error in Hz ÷
10, and a higher-pitched or more expressive speaker mechanically scores worse at
equal quality. `f0_probe.py` measures the speaker's distribution so you can read
the loss in her terms: 2.712 here is 27.1 Hz, which is 20% of her mean voiced pitch.

**A pitch offset and a pitch *error* look the same in the loss.** The +35 Hz global
shift found here accounted for essentially the whole 27 Hz F0 error. That is why
weighting the F0 loss harder did nothing — weighting a bias term harder does not
remove the bias.

## And then listen

These scripts generate candidates. They do not pick winners. Every shipped decision
was made by ear on A/B pairs.
