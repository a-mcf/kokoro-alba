# Training side

The recipe with our changes already applied lives in its own fork:

**https://github.com/a-mcf/kikiri-tts** — it is `main`, nothing to check out

```bash
git clone --recurse-submodules https://github.com/a-mcf/kikiri-tts
```

That is the whole setup. Nothing to copy in, nothing to patch. It pulls
`a-mcf/StyleTTS2 @ main` and `semidark/kokoro @ b96fef9` as submodules,
both pinned.

## What this directory holds

The fork carries the *code* changes. These are the things that are ours and are
not part of a recipe:

| file | what |
|---|---|
| `train_list.txt`, `val_list.txt` | the **repaired** splits, 4,383 + 230 (the recipe gitignores these) |
| `prepare_corpus.py` | corpus download → 24 kHz training audio |
| `check_corpus.py` | confirms the lists resolve against that audio before you train |
| `run_stage1.sh`, `run_stage2.sh` | launchers; set `KIKIRI_ROOT` |

The two list files are the ones this fine-tune actually trained on. Their
phoneme source is [`../dataset/phonemes.csv`](../dataset/); format is
`path|phonemes|speaker_id`, paths relative to `dataset/audio/`.

## What changed in the recipe, and why

Upstream is [`semidark/kikiri-tts`](https://github.com/semidark/kikiri-tts), a
**German** Kokoro fine-tuning recipe. Everything structural is theirs. The full
diff is one link each:

- recipe: [`semidark/kikiri-tts@a12d041 … a-mcf:main`](https://github.com/semidark/kikiri-tts/compare/a12d041...a-mcf:kikiri-tts:main)
- training code: [`semidark/StyleTTS2@b1956da … a-mcf:main`](https://github.com/semidark/StyleTTS2/compare/b1956da...a-mcf:StyleTTS2:main)

**`configs/config_alba_ft.yml`** — English config for a 24 GB card.
`batch_size: 3`, not 4. Batch 4 completes stage 1 and then dies partway through
stage 2, which is the memory-hungrier stage; 3 peaks around 20 GB of 24 and
climbs during the run, so an early reading is not the peak. `save_freq: 1`,
because an even `epochs_2nd` with `save_freq: 2` writes only even indices and
silently discards the final epoch.

**`training/OOD_texts.txt`** — was 20 German sentences. It feeds stage 2's SLM
adversarial loss, so a German out-of-distribution set was steering the
discriminator on an English model.

**`StyleTTS2/kokoro_tb_utils.py`** — the per-epoch TensorBoard audio previews,
for both stages, were German rendered through espeak-de. Every "listen as it
trains" sample in every run was grading an English model on German through the
wrong phoneme alphabet.

**`checks/verify_alignment.py`** — a pre-flight guard, described below.

Two traps remain in the recipe and are not ours to fix.
`scripts/prepare_dataset.py` and `scripts/prepare_training.py` hardcode
`EspeakG2P(language="de")` — inert on the path used here, but live traps if you
edit around them. And `prepare_dataset.py` as a whole is a Polly-MP3-plus-Whisper
pipeline filtering on `TARGET_LANGUAGE = "de"`; it does not apply to a corpus
that ships ground-truth transcripts.

## Run the guard before you train

```bash
cd <kikiri-tts> && ./.venv/bin/python checks/verify_alignment.py
```

It refuses to certify a dataset whose entries do not carry their own phonemes,
and exits non-zero so it can gate a launcher. `check_corpus.py` here does the
same job against this repo's committed lists.

It exists because a full 10-epoch stage 2 run — 13 hours — was lost to a
`train_list.txt` in which **4,613 of 4,613 entries carried a different clip's
phonemes**. Training converged. Validation loss looked survivable. The output
was crisp audio in the correct voice, and it was not English. The only signal in
the logs was `Dur` and `F0` losses stalling high.

Two independent checks, because either alone can be fooled:

1. **Exact re-pair** against `dataset/phonemes.csv` — catches any mismatch, but
   is blind to a `phonemes.csv` that is itself wrong.
2. **Correlation of phoneme-string length against audio duration** — a correct
   TTS corpus sits near **+0.95**, random pairing near **0**. Fails below 0.90.

Applied stage by stage, check 2 isolates the broken step exactly. On the
corrupted run: metadata↔audio +0.970, phonemes↔audio +0.972, phonemes↔metadata
+0.994, train_list↔source **−0.061**. Every stage clean except the last. The
technique is not specific to TTS — it works on any paired-data pipeline.

### The bug, so nobody writes it again

The list builder shuffled two parallel lists to make the train/val split:

```python
rng = random.Random(42)
rng.shuffle(meta_rows)   # permutation A
rng.shuffle(pho_rows)    # permutation B -- the generator has advanced!
```

One seeded generator, two `shuffle` calls, **two different permutations**.
Seeding makes a run repeatable; it does not make consecutive draws identical.
After those two lines `meta_rows[i]` and `pho_rows[i]` describe different clips,
and zipping them pairs one utterance's filename with another's phonemes — for
every row.

The correct pattern — shuffle an index, apply it to both — was already written
directly below, but it ran *after* the damage, so it permuted the mismatch
instead of repairing it.

`metadata.csv` and `phonemes.csv` were both fine throughout, because each row
kept its own filename with its own content. Only the zip of the two was wrong.
That is why every check on the inputs passed.

**The lesson the guard encodes: a paired-data pipeline needs a check that the
pairing survived, and it has to run on the file training actually reads.**
