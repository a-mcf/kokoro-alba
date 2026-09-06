#!/usr/bin/env python3
import sys, os, glob, torch, torchaudio, numpy as np, soundfile as sf, warnings
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
KIKIRI = os.environ.get("KIKIRI_ROOT", os.path.expanduser("~/kokoro_alba/kikiri-tts"))
WORK   = os.environ.get("ALBA_WORK",   os.path.expanduser("~/kokoro_alba/export_proof"))
ROOT = KIKIRI; EP = WORK
sys.path.insert(0,KIKIRI + "/StyleTTS2")
from models import load_F0_models
F0=load_F0_models(KIKIRI + "/StyleTTS2/Utils/JDC/bst.t7").eval()
to_mel=torchaudio.transforms.MelSpectrogram(n_mels=80,n_fft=2048,win_length=1200,hop_length=300)
def stats(p):
    d,sr=sf.read(p, dtype="float32", always_2d=True)
    w=torch.from_numpy(d.T)
    if sr!=24000: w=torchaudio.functional.resample(w,sr,24000)
    dur=w.shape[-1]/24000
    m=(torch.log(1e-5+to_mel(w.mean(0)).unsqueeze(0))-(-4))/4
    with torch.no_grad(): f,_,_=F0(m.unsqueeze(1))
    a=f.squeeze().numpy().ravel(); idx=np.where(a>50)[0]
    if len(idx)<20: return None
    t=idx/(24000/300.0); hz=a[idx]; tn=(t-t.min())/max(t.max()-t.min(),1e-6)
    return np.polyfit(t,hz,1)[0], hz[tn>=.75].mean()-hz[tn<=.25].mean(), np.median(hz), dur
clips=[os.path.basename(x)[:-4] for x in sorted(glob.glob(EP+"/sweep/s1.0/*.wav"))]
sets=[("REAL",lambda c:f"{ROOT}/dataset/audio/alba/{c}.wav"),
      ("fixed pack, speed 1.00",lambda c:f"{EP}/sweep/s0.35/{c}.wav"),
      ("fixed pack, speed 1.16",lambda c:f"{EP}/speed/sp1.16/{c}.wav"),
      ("fixed pack, speed 1.25",lambda c:f"{EP}/speed/sp1.25/{c}.wav")]
print(f"{'':<24}{'slope Hz/s':>12}{'fall Hz':>10}{'median Hz':>11}{'dur s':>8}{'vs real':>9}")
rd=np.mean([stats(f"{ROOT}/dataset/audio/alba/{c}.wav")[3] for c in clips])
for name,fn in sets:
    r=[stats(fn(c)) for c in clips if os.path.exists(fn(c))]; r=[x for x in r if x]
    sl,fa,md,du=(np.mean([x[i] for x in r]) for i in range(4))
    print(f"{name:<24}{sl:>12.1f}{fa:>+10.1f}{md:>11.1f}{du:>8.2f}{100*(du/rd-1):>+8.1f}%")
