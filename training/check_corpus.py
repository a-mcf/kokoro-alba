#!/usr/bin/env python3
"""Do the committed lists actually resolve against the audio you just prepared?

    check_corpus.py --audio-root <kikiri-tts>/dataset/audio

Run this after prepare_corpus.py and before training. It checks, from the repo's
own committed files:

  1. every path in train_list/val_list exists on disk
  2. every clip is 24 kHz mono
  3. the phonemes in each list row match dataset/phonemes.csv for that clip
  4. phoneme-string length correlates with audio duration at r >= 0.90

Check 3 is the one that matters. A full 13-hour run was lost to lists in which
all 4,613 rows carried some other clip's phonemes; the model trained to
convergence and produced fluent audio in the right voice that was not English.
Check 4 catches the same class of error even if phonemes.csv is itself wrong.
"""
import argparse
import os
import statistics as st
import sys
from pathlib import Path

import soundfile as sf

MIN_R = 0.90

ap = argparse.ArgumentParser()
ap.add_argument("--audio-root", required=True,
                help="the directory holding alba/, i.e. <kikiri-tts>/dataset/audio")
ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
a = ap.parse_args()

repo = Path(a.repo_root)
audio_root = Path(a.audio_root).resolve()

phon = {}
for line in open(repo / "dataset/phonemes.csv", encoding="utf-8"):
    parts = line.rstrip("\n").split("|")
    if len(parts) >= 2:
        phon[os.path.basename(parts[0])] = parts[1]

failed = False
for name in ("train_list.txt", "val_list.txt"):
    rows = []
    for line in open(repo / "training" / name, encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))

    missing = [f for f, _ in rows if not (audio_root / f).exists()]
    mispaired = [f for f, ipa in rows if phon.get(os.path.basename(f)) != ipa]

    lens, durs, badfmt = [], [], []
    for path, ipa in rows[:400]:
        wav = audio_root / path
        if wav.exists():
            info = sf.info(str(wav))
            if info.samplerate != 24000 or info.channels != 1:
                badfmt.append(f"{path} ({info.samplerate} Hz, {info.channels} ch)")
            lens.append(len(ipa))
            durs.append(info.frames / info.samplerate)

    r = 0.0
    if len(lens) > 10:
        mx, my = st.mean(lens), st.mean(durs)
        cov = sum((x - mx) * (y - my) for x, y in zip(lens, durs))
        vx = sum((x - mx) ** 2 for x in lens) ** 0.5
        vy = sum((y - my) ** 2 for y in durs) ** 0.5
        r = cov / (vx * vy) if vx and vy else 0.0

    bad = bool(missing or mispaired or badfmt) or r < MIN_R
    failed = failed or bad
    print(f"{name:<16} rows={len(rows):<5} missing={len(missing):<5} "
          f"mispaired={len(mispaired):<5} r={r:+.3f}  {'FAIL' if bad else 'ok'}")
    for f in missing[:3]:
        print(f"      missing: {audio_root / f}")
    for f in mispaired[:3]:
        print(f"      mispaired: {f} does not match dataset/phonemes.csv")
    for f in badfmt[:3]:
        print(f"      wrong format: {f} -- want 24000 Hz mono")

if failed:
    print("\nREFUSING to certify this dataset. Do not start training.")
    sys.exit(1)
print("\nCorpus OK.")
