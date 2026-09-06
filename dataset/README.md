# Dataset

The generated text side of the corpus, committed so that reproducing this
fine-tune needs only the audio.

| file | rows | what |
|---|---|---|
| `metadata.csv` | 4,613 | `filename\|text\|speaker` — the corpus transcripts |
| `phonemes.csv` | 4,613 | `filename\|ipa` — misaki `en.G2P(british=True)` output |

Both are derived from the **Alba speech corpus** (CSTR, University of Edinburgh,
CC BY 4.0) and are redistributed here under that licence with the attribution in
[NOTICE](../NOTICE). The audio itself is **not** included — get it from
https://doi.org/10.7488/ds/2506 and unpack it to `dataset/audio/alba/`, matching the
paths in the CSVs.

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
