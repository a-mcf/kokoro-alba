# Reproducing kokoro-alba

End to end: corpus → two-stage fine-tune → portable artifact → calibration.
Roughly 13 hours of training on a single RTX 3090, plus an hour of fiddling.

This is a guide to *this* fine-tune. The recipe underneath it is
[kikiri-tts](https://github.com/semidark/kikiri-tts), and its
`docs/TRAINING_GUIDE.md` remains the authority on the training machinery.

## 0. What you need

- A CUDA GPU with **24 GB**. Batch size 3 peaks around 20 GB and climbs during a
  run — an early 13.6 GB reading is not the peak. Batch 4 OOMs. The recipe's
  "batch_size=4 works on 12 GB" does not survive this corpus's clip lengths.
- The **Alba speech corpus**: https://doi.org/10.7488/ds/2506 (CC BY 4.0).
  The ~4-hour "plain" read set is the 4,613 clips used here.
- A checkout of kikiri-tts with its submodules.

```bash
git clone --recurse-submodules https://github.com/semidark/kikiri-tts
export KIKIRI_ROOT=$PWD/kikiri-tts
```

## 1. Apply this project's changes to the recipe

kikiri-tts is a German recipe. Four changes make it English and Scottish; details
and rationale in [`training/README.md`](training/README.md).

```bash
cp training/config_alba_ft.yml  "$KIKIRI_ROOT/configs/"
cp training/OOD_texts.txt       "$KIKIRI_ROOT/training/"
cp training/verify_alignment.py "$KIKIRI_ROOT/checks/"
cp training/test_alba.py        "$KIKIRI_ROOT/scripts/"
git -C "$KIKIRI_ROOT/StyleTTS2" apply "$OLDPWD/training/kokoro_tb_utils.english.patch"
```

## 2. Bring in the dataset

**The generated text side is already in this repo** — you do not need to
regenerate it. Copy it across and add the audio:

```bash
cp dataset/metadata.csv dataset/phonemes.csv "$KIKIRI_ROOT/dataset/"
cp training/train_list.txt training/val_list.txt "$KIKIRI_ROOT/training/"
# then unpack the corpus audio from the DOI into:
#   $KIKIRI_ROOT/dataset/audio/alba/
```

`train_list.txt` is the **repaired** list — see `training/README.md` for why that
word is doing work.

If you would rather rebuild it, use the recipe's `scripts/prepare_dataset.py` and
`scripts/prepare_training.py`, and phonemize with **misaki `en.G2P(british=True)`**
— that is what the shipped voicepack and both runtime scripts assume
(`lang_code="b"`). Two of the recipe's prep scripts hardcode
`EspeakG2P(language="de")`; they are inert on the English path but they are live
traps if you edit around them.

misaki's English phoneme strings contain literal capitals: `W A I Q` are
/aʊ/ /eɪ/ /aɪ/ /əʊ/. That is not corruption. Do not "fix" it.

## 3. Verify the pairing before you spend 13 hours

```bash
cd "$KIKIRI_ROOT" && ./.venv/bin/python checks/verify_alignment.py
```

Non-negotiable. This exists because a full run was lost to a `train_list.txt`
where **4,613 of 4,613 entries carried a different clip's phonemes**. The model
trained to convergence, validation loss looked survivable, and it produced crisp
audio in the correct voice that was not English.

The guard runs two independent checks: an exact re-pair against `phonemes.csv`,
and a correlation between phoneme-string length and audio duration. A correct
corpus sits near **+0.95**; random pairing sits near **0**. It exits non-zero
below 0.90.

That correlation generalises to any paired-data pipeline, and applied stage by
stage it isolates the broken step exactly.

## 4. Train

```bash
KIKIRI_ROOT=$KIKIRI_ROOT ./training/run_stage1.sh   # ~3h,  10 epochs
KIKIRI_ROOT=$KIKIRI_ROOT ./training/run_stage2.sh   # ~10h, 10 epochs
```

Stage 2 seeds from stage 1's `first_stage.pth`. Keep that file — it is three hours
to recreate.

What to watch, in `logs/kokoro-alba/train.log`:

- **`Dur` and `F0` losses**, not validation loss. They govern intelligibility, and
  they are what stayed high (2.238 / 4.064) during the corrupted run while val loss
  looked fine. A healthy stage 2 lands `Dur` under 0.5.
- Read the **validation** F0, not the per-step training F0. The latter swings
  1.7–2.5 within a single epoch and will flatter you.

Two calibration notes on the numbers themselves:

- `loss_F0 = F.l1_loss(F0_real, F0_fake) / 10` — plain L1 on the pitch extractor's
  **raw Hz output**, un-normalized. So the loss is literally mean absolute error in
  Hz ÷ 10, and a higher-pitched or more expressive speaker mechanically scores
  worse at equal quality. The guide's 1.8 target came from a German corpus and a
  different speaker; it is not a like-for-like goal.
- Judge stage 1 by S2S loss, not by ear. Its TensorBoard samples are near-static
  because the prosody predictor is a stage 2 component.

**`save_freq` gotcha:** an even `epochs_2nd` with `save_freq: 2` silently discards
the final epoch — only even indices are written, so index 11 of 12 never lands.
The config here uses `epochs_2nd: 10`, `save_freq: 1`.

## 5. Build the portable artifact

One script, no GPU needed:

```bash
python tools/build_artifact.py \
    --kikiri-root "$KIKIRI_ROOT" \
    --stage2 "$KIKIRI_ROOT/StyleTTS2/logs/kokoro-alba/epoch_2nd_00009.pth" \
    --stage1 "$KIKIRI_ROOT/StyleTTS2/logs/kokoro-alba/first_stage.pth" \
    --out-dir alba/
```

It runs the recipe's `convert_checkpoint`, then `tools/convert_to_stock.py`, then
the recipe's `extract_voicepack.py` at `--num-samples 200`, and copies
`config.json` in alongside. Output: `alba_stock.pth`, `alba_voicepack.pt`,
`config.json`.

`--stage1` is passed through as `--style-encoder-model`: stage 2 training can
degrade the style_encoder through spectral_norm buffer drift, so the extractor
takes the timbre half from stage 1 and the prosody half from stage 2.

✅ **The model half of this is exactly reproducible, and that was checked.**
Re-running this script against `epoch_2nd_00009.pth` produces a checkpoint whose
tensors are **identical to the shipped `alba_stock.pth`, 548 of 548**, and which
verifies at 548 landed / 0 mismatched / 0 orphaned. The file's md5 differs, but
only because `torch.save` serialization metadata is not stable — the weights are
the same.

Step 2 is the one that fails silently if skipped. See the README section on it.

Confirm the tensors actually landed, rather than trusting the loader:

```bash
python tools/verify_load.py alba/config.json alba/alba_stock.pth converted
```

Expect **548 landed, 0 mismatched, 0 orphaned**. It will also report ~140 model
params the checkpoint did not cover — those are AdaLayerNorm `.norm` affine
weights, default-initialised on every path including the known-good one. Normal,
not a defect.

⚠️ **The shipped voicepack is not bit-reproducible, and that is a property of the
tooling, not a mistake here.** `alba/alba_voicepack.pt` was extracted through the
audition path (`StyleTTS2/kokoro_tb_utils.py::extract_voicepack`), which calls
`random.shuffle` with **no seed** — so which 200 clips it averaged is not
recoverable. `scripts/extract_voicepack.py` seeds with `random.Random(42)` and *is*
deterministic, but lands about 1% away: re-running it against the same checkpoint
gives norms **1.3513 / 1.8507** against the shipped pack's **1.3550 / 1.8694**.

The practical consequence: **the 0.35 pitch scale belongs to the shipped pack.** If
you extract your own, re-derive the scale in step 6 rather than reusing the number.
The shipped raw pack is committed at `alba/alba_voicepack.pt` if you would rather
skip extraction entirely.

Traps in this step, each of which fails unhelpfully:

- Pass `config.json`, **not** the training `.yml`. `KModel` calls `json.load()`,
  and the YAML has no `vocab` key.
- The voicepack must be `[510, 1, 256]`, not a flat `(256,)`. Kokoro does
  `pack[len(ps)-1]` then `ref_s[:, 128:]`; a flat vector dies with
  `IndexError: too many indices for tensor of dimension 0` — an error that never
  mentions shape. It is one global style vector repeated 510 times.
- Extract with **`n_samples=200`**. 40 gives a prosodic norm around 2.0 and
  exaggerates the pitch offset you are about to spend an hour correcting.

## 6. Calibrate

Two constants, both derived by measurement against the speaker's own recordings.
**Re-derive them for every checkpoint** — they do not transfer.

Set the paths the scripts read:

```bash
export KIKIRI_ROOT=/path/to/kikiri-tts
export ALBA_WORK=/path/to/your/working/dir     # holds config.json, alba_stock.pth, packs
```

### Pitch

```bash
python calibration/synth_pairs.py     # synthesize the same sentences she recorded
python calibration/pitch_cmp.py       # how far off is the pitch, and in which direction
python calibration/scale_sweep.py     # render the sweep at 1.0 / 0.75 / 0.5 / 0.35 / 0.2
python calibration/sweep_measure.py   # pick the scale whose median matches hers
python calibration/qc.py              # confirm no quality damage across the sweep
python tools/apply_pitch_scale.py raw_pack.pt alba/alba_voicepack_pitchfix.pt 0.35
```

Calibrate against **her measured median**, not against stock Kokoro's prosodic
norm — matching stock's 0.367 overshoots downward here.

### Speed

```bash
python calibration/speedtest.py
python calibration/declination.py     # does the pitch trail down the way she does
python calibration/decl2.py
```

Then pick by ear. 1.25 won here on both duration (within 3% of her) and
declination slope (−41.6 against her −51), but the speed sweep is **non-monotonic**
— 1.16 was worse than both 1.00 and 1.25 — so do not assume the relationship is
smooth and do not extrapolate. Eight clips may be too few to say more than "this
one sounds right".

`calibration/f0_probe.py` measures the speaker's own pitch distribution through the
same extractor, which is what tells you whether an F0 loss number means anything.

## 7. Listen

The measurements narrow the search. They do not close it. Every decision that
shipped here — the 0.35 scale, the 1.25 speed, keeping run 1 over run 2, rejecting
the coda-L rewrite — was settled by a human listening to A/B pairs, with the
numbers only used to generate the candidates worth listening to.
