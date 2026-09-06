#!/usr/bin/env python3
"""Is the synthesized voice pitched HIGHER than the real recordings?
Same sentences, same pitch extractor, paired comparison."""
import sys, os, glob, torch, torchaudio, numpy as np, warnings
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
KIKIRI = os.environ.get("KIKIRI_ROOT", os.path.expanduser("~/kokoro_alba/kikiri-tts"))
WORK   = os.environ.get("ALBA_WORK",   os.path.expanduser("~/kokoro_alba/export_proof"))
sys.path.insert(0,KIKIRI + "/StyleTTS2")
from models import load_F0_models
F0=load_F0_models(KIKIRI + "/StyleTTS2/Utils/JDC/bst.t7").eval()
to_mel=torchaudio.transforms.MelSpectrogram(n_mels=80,n_fft=2048,win_length=1200,hop_length=300)

def pitch(p):
    w,sr=torchaudio.load(p)
    if sr!=24000: w=torchaudio.functional.resample(w,sr,24000)
    m=(torch.log(1e-5+to_mel(w.mean(0)).unsqueeze(0))-(-4))/4
    with torch.no_grad(): f,_,_=F0(m.unsqueeze(1))
    a=f.squeeze().numpy().ravel()
    v=a[a>50]                      # real voiced frames only; drops silence
    return v

D=WORK + "/pitchcmp"
rows=[]
for real in sorted(glob.glob(D+"/*_REAL.wav")):
    syn=real.replace("_REAL.wav","_SYNTH.wav")
    if not os.path.exists(syn): continue
    r,s=pitch(real),pitch(syn)
    if len(r)<10 or len(s)<10: continue
    rows.append((os.path.basename(real)[:-9], np.median(r), np.median(s), r, s))

print(f"{'clip':<10}{'REAL med':>10}{'SYNTH med':>11}{'diff':>9}")
for n,rm,sm,_,_ in rows: print(f"{n:<10}{rm:>10.1f}{sm:>11.1f}{sm-rm:>+9.1f}")
R=np.concatenate([x[3] for x in rows]); S=np.concatenate([x[4] for x in rows])
dm=np.array([x[2]-x[1] for x in rows])
print(f"\npooled voiced frames: real n={len(R)} synth n={len(S)}")
print(f"REAL : median {np.median(R):.1f} Hz  mean {R.mean():.1f}  p10 {np.percentile(R,10):.1f}  p90 {np.percentile(R,90):.1f}  std {R.std():.1f}")
print(f"SYNTH: median {np.median(S):.1f} Hz  mean {S.mean():.1f}  p10 {np.percentile(S,10):.1f}  p90 {np.percentile(S,90):.1f}  std {S.std():.1f}")
print(f"\nper-clip median shift: mean {dm.mean():+.1f} Hz   median {np.median(dm):+.1f} Hz")
print(f"clips where synth is HIGHER: {int((dm>0).sum())} of {len(dm)}")
print(f"semitones (pooled medians): {12*np.log2(np.median(S)/np.median(R)):+.2f}")
