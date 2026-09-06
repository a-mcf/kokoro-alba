# Dataset

The generated text side of the corpus, committed so that reproducing this
fine-tune needs only the audio.

| file | rows | what |
|---|---|---|
| `metadata.csv` | 4,613 | `filename\|text\|speaker` — the corpus transcripts |
| `phonemes.csv` | 4,613 | `filename\|ipa` — misaki `en.G2P(british=True)` output |

Both are derived from the **Alba speech corpus** (CSTR, University of Edinburgh,
CC BY 4.0) and are redistributed here under that licence with the attribution in
[NOTICE](../NOTICE). The audio itself is **not** included.

## Getting the audio, and exactly where it goes

Download **`plain.zip`** (1019 MB, the ~4-hour read set) from
https://doi.org/10.7488/ds/2506 — that is the style this fine-tune used. The other
three archives (`fast`, `clear_c`, `clear_h`) are not used.

It unpacks a deep AFS path. That is normal, and you do not need to flatten it:

```
afs/inf.ed.ac.uk/group/cstr/projects/scar/SCRIPT/Release_Alba/plain/
    wav/<stem>.wav      4613 files, 48 kHz
    txt/<stem>.txt      4613 files, one sentence each
```

Training wants **24 kHz mono 16-bit**, at `<kikiri-tts>/dataset/audio/alba/`:

```
<kikiri-tts>/dataset/audio/alba/<stem>.wav      4613 files, 24 kHz mono PCM_16
```

**The stems do not change.** `wav/3585.wav` becomes `alba/3585.wav`. Verified: the
sorted stem list of the corpus and of the prepared dataset are identical, so the
committed CSVs and lists key directly onto the corpus's own names. Note the stems
are not a tidy 1..4613 sequence — there are names like `1_368` — so do not
renumber them.

```bash
python training/prepare_corpus.py \
    --corpus-root /path/to/unzipped/plain \
    --out /path/to/kikiri-tts/dataset/audio/alba

python training/check_corpus.py \
    --audio-root /path/to/kikiri-tts/dataset/audio
```

`prepare_corpus.py` finds the `wav/` directory wherever it sits under
`--corpus-root`, resamples, and keeps the filenames. `check_corpus.py` then
confirms every list row resolves to a real 24 kHz mono file whose phonemes match
`phonemes.csv`.

The `txt/` side of the corpus is not needed — those transcripts are already here
as `metadata.csv`.

The train and validation splits built from these live in
[`../training/`](../training/): `train_list.txt` (4,383) and `val_list.txt` (230).

## Reading `phonemes.csv`

misaki's English output contains literal capital letters. They are phonemes, not
corruption, and there is nothing to fix:

| symbol | sound |
|---|---|
| `W` | /aʊ/ |
| `A` | /eɪ/ |
| `I` | /aɪ/ |
| `Q` | /əʊ/ |

The labels are non-rhotic RP. That is a deliberate mismatch with a Scottish
speaker and it has consequences — see the "Known limits" section of the top-level
README, which covers coda-L in particular.

## If you regenerate these

Run `checks/verify_alignment.py` before training. The correlation between phoneme
string length and audio duration should sit near **+0.95**; anything under 0.90
means the rows are not paired with their own audio, and the guard exits non-zero.
A full 13-hour run was lost to exactly that failure, which is invisible in the
training logs.
