# kokoro-alba

A Scottish-voiced fine-tune of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
trained on the [Alba speech corpus](https://doi.org/10.7488/ds/2506) from CSTR at
the University of Edinburgh, packaged so it runs on **stock `pip install kokoro`** —
no fork, no vendored package, no `sys.path` insert.

It is 24 kHz mono and runs on CPU at RTF 0.54 on two threads, in about 1.3 GB of
RAM, with no network access and no base model download. See [SERVING.md](SERVING.md)
before deploying it — the 5.7-second cold start is the thing that decides how you
should run it.

```
pip install kokoro==0.9.4 soundfile
python -m spacy download en_core_web_sm
python alba/alba_say.py "Aye, it's a bonnie morning for a walk by the water." out.wav
```

Three renders with the shipped defaults are in [`samples/`](samples/).

## Credit

The voice belongs to a real person who was recorded by researchers who released
the corpus properly. Read [NOTICE](NOTICE) — it is short and it matters.

> Valentini-Botinhao, Cassia; Yamagishi, Junichi. (2019). *Alba speech corpus*
> [dataset]. University of Edinburgh. https://doi.org/10.7488/ds/2506

The corpus is CC BY 4.0, which permits redistribution, derivatives and commercial
use with attribution. **It also carries a term plain CC BY does not:** it preserves
the voice talent's moral rights, including the right of integrity, and forbids uses
derogatory to the voice talent. These weights inherit that.

The training recipe is [kikiri-tts](https://github.com/semidark/kikiri-tts) by
semidark — a *German* Kokoro fine-tuning recipe. All the structural work is theirs.
This project adapted it to English and to a Scottish speaker.

The corpus audio is **not** redistributed here. Fetch it from the DOI.

## What's in here

| path | what |
|---|---|
| `alba/` | the runtime. Drop `alba_stock.pth` in beside these and it works. |
| `tools/` | checkpoint conversion, voicepack pitch scaling, load verification |
| `calibration/` | the measurement scripts that produced the two shipped constants |
| `training/` | diffs and configs against kikiri-tts, the pre-flight guard, and the generated train/val lists |
| `dataset/` | the phoneme and transcript CSVs this fine-tune was built from |
| `samples/` | three renders, so you can hear it before installing anything |

The 312 MB model file is **not in git** — it is attached to the
[latest release](https://github.com/a-mcf/kokoro-alba/releases/latest) as `alba_stock.pth`. Download it into
`alba/`.

```
curl -L -o alba/alba_stock.pth \
  https://github.com/a-mcf/kokoro-alba/releases/latest/download/alba_stock.pth
```

## Using it

Full deployment guide, with measured costs and the two ways this fails silently,
is in [SERVING.md](SERVING.md). The short version — two entry points, same engine:

```bash
# interactive
python alba/alba_say.py "text to speak" out.wav

# command-provider interface, for wiring into an agent / gateway
python alba/alba_tts.py --text-file in.txt --out out.wav
```

Both print a duration and a zero-crossing rate. **Check the zcr.** Speech is
around 0.13; 0.33 or higher means you are listening to static (see below).

Runtime is the whole dependency list:

```bash
python3 -m venv venv
venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
venv/bin/pip install kokoro==0.9.4 soundfile
venv/bin/python -m spacy download en_core_web_sm   # misaki needs it
```

**You do not need the base Kokoro-82M weights**, despite `repo_id=` appearing in
the scripts — verified by running with an empty HF cache and downloading nothing.
espeak-ng ships inside the venv via `espeakng-loader`; no system package.

The spaCy model is not optional — misaki does grapheme-to-phoneme conversion and
requires it. A venv built without pip cannot self-download it and fails with
`No package installer found`, which reads like a packaging bug rather than a TTS one.

## What we did

Three things beyond running the recipe.

### 1. Made the checkpoint run on stock `kokoro`

Training emits state-dict keys in PyTorch's newer weight-norm form
(`parametrizations.weight.original0` / `original1`), because the StyleTTS2 fork the
recipe uses migrated to `torch.nn.utils.parametrizations.weight_norm`. PyPI
`kokoro` 0.9.4 still uses the older `torch.nn.utils.weight_norm`, whose keys are
`weight_g` / `weight_v`.

The failure mode is the interesting part. `KModel.__init__` wraps the load in a
bare `except` that strips seven characters off every key and retries with
`strict=False`. **A name mismatch therefore raises nothing.** 178 keys go
unmatched, 318 of 688 parameters stay randomly initialised, and you get static
out of a program that reported no error.

`tools/convert_to_stock.py` renames the keys 1:1. No tensor values change. Measured
on the same phonemes:

| path | zcr | result |
|---|---|---|
| stock pip 0.9.4 + converted checkpoint | 0.1340 | speech |
| upstream `kokoro` main + original checkpoint | 0.1340 | speech |
| stock pip 0.9.4 + original checkpoint | 0.4971 | **static** |

Rows 1 and 2 are bit-identical (`max|diff| = 0.0`). Row 3 correlates 0.0018 with
row 2 — noise.

`tools/verify_load.py` checks this properly, tensor by tensor, instead of trusting
the forgiving loader: converted gives 548/548 landed, 0 mismatched, 0 orphaned.

### 2. Fixed the pitch by calibrating the voicepack

The raw extracted voicepack renders **+3.09 semitones sharp** — confirmed on 14 of
14 clips, synthesizing the same sentences the speaker actually recorded and running
both through the same JDCNet pitch extractor:

| | median | mean | p10 | p90 |
|---|---|---|---|---|
| real recordings | 197.3 Hz | 197.9 | 95.2 | 283.3 |
| synthesized, raw pack | 235.8 Hz | 232.1 | 103.2 | 327.6 |

A Kokoro voicepack is `[510, 1, 256]`: the first 128 dims are timbre, the last 128
are prosody. The **magnitude of the prosodic half is the pitch knob**. Sweeping it:

| scale | prosodic norm | synth median | vs real | semitones |
|---|---|---|---|---|
| 1.0 (raw) | 1.869 | 235.8 Hz | +38.9 | +3.12 |
| 0.75 | 1.402 | 221.7 | +24.9 | +2.06 |
| 0.5 | 0.934 | 205.9 | +9.0 | +0.78 |
| **0.35** | **0.654** | **197.5** | **+0.7** | **+0.06** |
| 0.2 | 0.374 | 189.7 | −7.1 | −0.64 |

`alba_voicepack_pitchfix.pt` is the raw pack at 0.35, produced by
`tools/apply_pitch_scale.py`. No audio quality cost detected (zcr 0.1455 → 0.1487
across the whole sweep; rms and peak flat).

Two things worth carrying forward:

- **Calibrate against the speaker's measured pitch, not against stock Kokoro's
  norm.** Stock's prosodic norm is 0.367, which corresponds to scale 0.2 here —
  and that overshoots downward by 0.64 semitones. The stock number is a red herring.
- **The scaling flattens the contour**, and that is a real cost, not a hypothetical
  one: declination slope went from −43.8 to −25.7 Hz/s. It buys the right pitch
  level and the right *total* fall (−63.9 Hz against her −65.3) at the price of the
  *rate* of fall.

### 3. Fixed the recipe's German assumptions

kikiri-tts is a German recipe. Adapting it to English needed more than a config
change — see [`training/README.md`](training/README.md). The one worth naming here:
`StyleTTS2/kokoro_tb_utils.py` held German test sentences, phonemized with
espeak-de, and that file drives the per-epoch TensorBoard audio previews for both
training stages. Every "listen as it trains" sample was grading an English model on
German through the wrong phoneme alphabet.

`training/` also carries `verify_alignment.py`, a pre-flight guard that refuses to
certify a dataset whose text and audio are not actually paired. It exists because a
13-hour run was lost to exactly that.

## Shipped constants — do not change casually

```
--speed 1.25                     matches her utterance length within 3%
alba_voicepack_pitchfix.pt       the raw pack is +3.1 semitones sharp
```

Both were **measured against the real speaker's recordings**, not chosen by taste,
and both are **specific to this checkpoint**. If you retrain, re-derive them with
the scripts in `calibration/`. Do not carry the numbers across.

## Known limits

**It produces a plausible reading, not *her* reading of a given sentence.** The
model sees text plus one global style vector; it never sees which word she leaned
on or where she broke the phrase. That information is not in the words. Sometimes
the two coincide and it is uncanny; sometimes it is the same voice making a
different, equally valid choice. This is the ceiling of text-conditioned TTS, not a
defect in this model.

**Coda L is pronounced, and she vocalizes it.** She says *"concede a ho"*; the
model says *"concede a hole"*, because the RP phoneme labels contain an explicit
`l` her audio does not. Rewriting the phonemes at synthesis time was tried and
rejected — her articulation is a softened, breathy L, and the phoneme inventory
offers only a full `l`, a full vowel, or nothing. Capturing it needs a label symbol
that does not exist in this alphabet. Note the trade: the RP labels also act as an
intelligibility filter, and the result is a Scottish-*flavoured* voice that is
easier to parse than the real speaker on some clips.

## Reproducing it

See [REPRODUCE.md](REPRODUCE.md). The generated artifacts are committed, not just
described — `dataset/phonemes.csv`, `dataset/metadata.csv` and the repaired
`training/{train,val}_list.txt` are all here, so reproducing this needs only the
corpus audio from the DOI.

## License

Apache 2.0 for the code and the weights. See [LICENSE](LICENSE) and, more
importantly, [NOTICE](NOTICE).
