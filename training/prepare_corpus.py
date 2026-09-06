#!/usr/bin/env python3
"""Turn the Alba corpus download into the training audio layout.

    prepare_corpus.py --corpus-root UNZIPPED_PLAIN --out KIKIRI/dataset/audio/alba

The corpus ships 48 kHz WAVs; training wants 24 kHz mono 16-bit. **Filenames are
not changed** — the committed metadata.csv, phonemes.csv and train/val lists all
key on the corpus's own stems, so a straight resample is the whole job.

Where to get it:
    https://doi.org/10.7488/ds/2506  ->  plain.zip  (1019 MB, the ~4h read set)

What is inside plain.zip (it unpacks a deep AFS path, which is fine):
    afs/inf.ed.ac.uk/group/cstr/projects/scar/SCRIPT/Release_Alba/plain/
        wav/<stem>.wav      4613 files, 48 kHz
        txt/<stem>.txt      4613 files, one sentence each

Point --corpus-root at anything above that `plain/wav` directory; this script
finds it. The `txt/` side is not needed — the transcripts are already committed
as dataset/metadata.csv.

Needs torch, torchaudio and soundfile: run it in the training venv, not the
lean runtime venv from SERVING.md.
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

TARGET_SR = 24000
EXPECTED = 4613

ap = argparse.ArgumentParser()
ap.add_argument("--corpus-root", required=True,
                help="anywhere above the corpus's plain/wav directory")
ap.add_argument("--out", required=True,
                help="destination, normally <kikiri-tts>/dataset/audio/alba")
a = ap.parse_args()

root = Path(a.corpus_root).resolve()
matches = [p for p in root.rglob("wav") if p.is_dir() and any(p.glob("*.wav"))]
if not matches:
    sys.exit(f"no directory containing .wav files found under {root}")
if len(matches) > 1:
    # prefer the 'plain' style if several styles were unpacked side by side
    plain = [p for p in matches if p.parent.name == "plain"]
    if len(plain) != 1:
        sys.exit("ambiguous corpus root; point --corpus-root at the 'plain' style:\n  "
                 + "\n  ".join(str(p) for p in matches))
    matches = plain
src = matches[0]

out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)

wavs = sorted(src.glob("*.wav"))
print(f"source: {src}\n  {len(wavs)} wav files -> {out}")

for i, w in enumerate(wavs, 1):
    # soundfile rather than torchaudio.load: torchaudio 2.14 routes load()
    # through TorchCodec, which is not in the recipe's dependency list.
    data, sr = sf.read(str(w), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SR)
    sf.write(str(out / w.name), audio.squeeze(0).numpy(), TARGET_SR, subtype="PCM_16")
    if i % 500 == 0:
        print(f"  {i}/{len(wavs)}")

written = sorted(out.glob("*.wav"))
print(f"\nwrote {len(written)} files at {TARGET_SR} Hz mono PCM_16")
if len(written) != EXPECTED:
    print(f"WARNING: expected {EXPECTED} clips from the 'plain' set, got {len(written)}")
print("\nNext: check the lists resolve against it —")
print("  python training/check_corpus.py --audio-root <kikiri-tts>/dataset/audio")
