#!/usr/bin/env python3
"""Guard: does every training entry carry ITS OWN phonemes?

Run from the repo root BEFORE launching any training run:

    ./.venv/bin/python checks/verify_alignment.py

Exits non-zero if the text<->audio pairing is broken. On 2026-08-30 a full
10-epoch Stage 2 run was wasted because train_list.txt had been rebuilt by an
ad-hoc step that zipped filenames against an independently-ordered phoneme
list: 4613 of 4613 entries carried some OTHER clip's phonemes. Training
converged fine and produced clean audio in the correct voice that was not
English. Nothing in the logs indicated a problem -- only the duration and F0
losses stalling high (2.238 / 4.064) hinted at it.

Two independent checks, because either alone can be fooled:
  1. exact re-pair against dataset/phonemes.csv  (catches any mismatch)
  2. phoneme-count vs audio-duration correlation (catches a bad phonemes.csv)
"""
import os
import statistics as st
import sys

import soundfile as sf

AUDIO = "dataset/audio"
LISTS = ("training/train_list.txt", "training/val_list.txt")
MIN_R = 0.90


def load_csv(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("|")
        if len(parts) >= 2:
            out[os.path.basename(parts[0])] = parts[1]
    return out


def corr(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs) ** 0.5
    vy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def main():
    phon = load_csv("dataset/phonemes.csv")
    failed = False

    for lst in LISTS:
        if not os.path.exists(lst):
            print("MISSING %s" % lst)
            failed = True
            continue

        rows = []
        for line in open(lst, encoding="utf-8"):
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))

        mismatched = [f for f, ipa in rows if phon.get(os.path.basename(f)) != ipa]

        lens, durs = [], []
        for path, ipa in rows[:400]:
            wav = os.path.join(AUDIO, path)
            if os.path.exists(wav):
                info = sf.info(wav)
                lens.append(len(ipa))
                durs.append(info.frames / info.samplerate)
        r = corr(lens, durs) if len(lens) > 10 else 0.0

        bad = bool(mismatched) or r < MIN_R
        failed = failed or bad
        print("%-26s  entries=%-5d  mispaired=%-5d  r=%+.3f  %s"
              % (lst, len(rows), len(mismatched), r, "FAIL" if bad else "ok"))
        for f in mismatched[:3]:
            print("      e.g. %s does not match dataset/phonemes.csv" % f)

    if failed:
        print("\nREFUSING to certify this dataset. Do not start training.")
        return 1
    print("\nAlignment OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
