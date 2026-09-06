#!/usr/bin/env python3
"""Does the synth trail its pitch down at the end the way she does?"""
import sys, os, glob, torch, torchaudio, numpy as np, warnings
import soundfile as sf
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
KIKIRI = os.environ.get("KIKIRI_ROOT", os.path.expanduser("~/kokoro_alba/kikiri-tts"))
WORK   = os.environ.get("ALBA_WORK",   os.path.expanduser("~/kokoro_alba/export_proof"))
ROOT = KIKIRI; EP = WORK
sys.path.insert(0,KIKIRI + "/StyleTTS2")
from models import load_F0_models
F0=load_F0_models(KIKIRI + "/StyleTTS2/Utils/JDC/bst.t7").eval()
to_mel=torchaudio.transforms.MelSpectrogram(n_mels=80,n_fft=2048,win_length=1200,hop_length=300)

def contour(p):
    d,sr=sf.read(p, dtype="float32", always_2d=True)
    w=torch.from_numpy(d.T)
    if sr!=24000: w=torchaudio.functional.resample(w,sr,24000)
    m=(torch.log(1e-5+to_mel(w.mean(0)).unsqueeze(0))-(-4))/4
    with torch.no_grad(): f,_,_=F0(m.unsqueeze(1))
    a=f.squeeze().numpy().ravel()
    idx=np.where(a>50)[0]
    if len(idx)<20: return None
    return idx/ (24000/300.0), a[idx]        # seconds, Hz (voiced only)

def stats(p):
    c=contour(p)
    if c is None: return None
    t,hz=c
    tn=(t-t.min())/max(t.max()-t.min(),1e-6)          # normalized 0..1
    slope=np.polyfit(t,hz,1)[0]                        # Hz per second
    head=hz[tn<=0.25].mean(); tail=hz[tn>=0.75].mean()
    return slope, tail-head, head, tail

clips=[os.path.basename(x)[:-4] for x in sorted(glob.glob(EP+"/sweep/s1.0/*.wav"))]
sets={"REAL":lambda c:f"{ROOT}/dataset/audio/alba/{c}.wav",
      "SYNTH s1.0 (sharp)":lambda c:f"{EP}/sweep/s1.0/{c}.wav",
      "SYNTH s0.35 (fixed)":lambda c:f"{EP}/sweep/s0.35/{c}.wav"}
print(f"{'':<22}{'slope Hz/s':>12}{'tail-head Hz':>14}{'head':>8}{'tail':>8}")
for name,fn in sets.items():
    rows=[stats(fn(c)) for c in clips if os.path.exists(fn(c))]
    rows=[r for r in rows if r]
    sl=np.array([r[0] for r in rows]); dl=np.array([r[1] for r in rows])
    hd=np.array([r[2] for r in rows]); tl=np.array([r[3] for r in rows])
    print(f"{name:<22}{sl.mean():>12.1f}{dl.mean():>+14.1f}{hd.mean():>8.1f}{tl.mean():>8.1f}")
print(f"\nclips: {len(clips)}   (negative slope / negative tail-head = pitch trails DOWN)")
