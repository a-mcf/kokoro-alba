#!/usr/bin/env python3
"""What scale is StyleTTS2's F0 loss actually on?

loss_F0 = l1_loss(F0_real, F0_fake)/10, so val F0 loss 2.712 == mean abs error
of 27.1 in whatever units the pitch extractor emits. Measure those units.
"""
import sys, os, torch, torchaudio, numpy as np, warnings, random
warnings.filterwarnings("ignore")
KIKIRI = os.environ.get("KIKIRI_ROOT", os.path.expanduser("~/kokoro_alba/kikiri-tts"))
WORK   = os.environ.get("ALBA_WORK",   os.path.expanduser("~/kokoro_alba/export_proof"))
ROOT = KIKIRI; EP = WORK
torch.set_num_threads(2)
sys.path.insert(0, KIKIRI + "/StyleTTS2")
from models import load_F0_models

ROOT = KIKIRI
F0 = load_F0_models(ROOT + "/StyleTTS2/Utils/JDC/bst.t7").eval()

to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
MEAN, STD = -4, 4

lines = [l.strip() for l in open(ROOT + "/training/val_list.txt", encoding="utf-8") if l.strip()]
random.seed(0); random.shuffle(lines)
vals = []
used = 0
for ln in lines[:40]:
    wav_rel = ln.split("|")[0]
    p = f"{ROOT}/dataset/audio/{wav_rel}"
    try:
        w, sr = torchaudio.load(p)
    except Exception:
        continue
    if sr != 24000:
        w = torchaudio.functional.resample(w, sr, 24000)
    m = to_mel(w.mean(0))
    m = (torch.log(1e-5 + m.unsqueeze(0)) - MEAN) / STD
    with torch.no_grad():
        f0_real, _, _ = F0(m.unsqueeze(1))
    a = f0_real.squeeze().numpy()
    vals.append(a); used += 1

if not used:
    print("no wavs found - check dataset path"); sys.exit(1)
allv = np.concatenate([v.ravel() for v in vals])
voiced = allv[allv > 1e-3]
print(f"clips used: {used}")
print(f"F0 extractor output — min {allv.min():.2f}  max {allv.max():.2f}  mean {allv.mean():.2f}")
print(f"voiced frames only  — mean {voiced.mean():.2f}  median {np.median(voiced):.2f}  p10 {np.percentile(voiced,10):.2f}  p90 {np.percentile(voiced,90):.2f}  std {voiced.std():.2f}")
print()
print(f"val F0 loss 2.712  => mean abs error {2.712*10:.1f} in these units")
print(f"recipe target 1.8  => mean abs error {1.8*10:.1f} in these units")
print(f"our MAE as % of voiced mean : {2.712*10/voiced.mean()*100:.1f}%")
print(f"our MAE as % of voiced std  : {2.712*10/voiced.std()*100:.1f}%")
