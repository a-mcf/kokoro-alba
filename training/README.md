# Changes to the kikiri-tts recipe

[kikiri-tts](https://github.com/semidark/kikiri-tts) is a **German** Kokoro
fine-tuning recipe. These are the files that make it English and Scottish. They
are diffs and drop-ins against that repo, not a standalone training system —
clone it, then apply these.

| file | goes to | what |
|---|---|---|
| `config_alba_ft.yml` | `configs/` | training config for this corpus and a 24 GB card |
| `OOD_texts.txt` | `training/` | English out-of-distribution sentences |
| `kokoro_tb_utils.english.patch` | `StyleTTS2/` | English TensorBoard preview sentences |
| `verify_alignment.py` | `checks/` | pre-flight guard on text↔audio pairing |
| `test_alba.py` | `scripts/` | convert-and-synthesize a checkpoint |
| `run_stage1.sh`, `run_stage2.sh` | anywhere | launchers; set `KIKIRI_ROOT` |

## The German recipe leaks in more places than you would expect

Assume more than are listed here.

**`OOD_texts.txt` was 20 German sentences.** It feeds stage 2's SLM adversarial
loss, so a German out-of-distribution set was steering the discriminator on an
English model. Replaced with English.

**`StyleTTS2/kokoro_tb_utils.py` held German `TEST_SENTENCES`, phonemized with
espeak-de.** That file drives the per-epoch TensorBoard audio previews for *both*
stages. Every "listen as it trains" sample in every run was grading an English
model on German text through the wrong phoneme alphabet — which is why the built-in
previews could never have caught a corpus problem. The patch swaps in English
sentences matched to the corpus domain and phonemizes them with
`misaki en.G2P(british=True)`, the same G2P as the training labels. Verified 7/7
sentences with zero out-of-vocabulary characters.

**`scripts/prepare_dataset.py` and `scripts/prepare_training.py` hardcode
`EspeakG2P(language="de")`.** Inert on the English path as used here, but live
traps if you edit around them.

## `verify_alignment.py`

Run it from the recipe root before every training launch:

```bash
./.venv/bin/python checks/verify_alignment.py
```

It refuses to certify a dataset whose entries do not carry their own phonemes, and
exits non-zero so it can gate a launcher.

It exists because a full 10-epoch stage 2 run — 13 hours — was wasted on a
`train_list.txt` in which **4,613 of 4,613 entries carried a different clip's
phonemes**. Training converged. Validation loss looked survivable. The output was
crisp audio in the correct voice, and it was not English. The only signal in the
logs was `Dur` and `F0` losses stalling high.

Two independent checks, because either alone can be fooled:

1. **Exact re-pair** against `dataset/phonemes.csv` — catches any mismatch, but is
   blind to a `phonemes.csv` that is itself wrong.
2. **Correlation of phoneme-string length against audio duration** — a correct TTS
   corpus sits near **+0.95**, random pairing near **0**. Catches a bad
   `phonemes.csv`. Fails below 0.90.

Applied stage by stage, check 2 isolates the broken step exactly. On the corrupted
run: metadata↔audio +0.970, phonemes↔audio +0.972, phonemes↔metadata +0.994,
train_list↔source **−0.061**. Every stage clean except the last one. The technique
is not specific to TTS — it works on any paired-data pipeline.

## `config_alba_ft.yml` notes

- `batch_size: 3` is the **ceiling** on a 24 GB card for this corpus. It peaks
  around 20 GB and climbs during the run; batch 4 OOMs.
- `save_freq: 1` with `epochs_2nd: 10`. An even epoch count with `save_freq: 2`
  silently discards the final epoch — only even indices are written.
- `lambda_F0: 1.0`. Raising it does not buy pitch accuracy; a 2.5× increase moved
  validation F0 by ~0.015, inside the run's own epoch-to-epoch jitter. Pitch level
  is a voicepack calibration problem, not a loss-weighting one.
