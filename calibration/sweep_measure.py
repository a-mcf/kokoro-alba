#!/usr/bin/env python3
import sys, os, glob, torch, torchaudio, numpy as np, warnings
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
KIKIRI = os.environ.get("KIKIRI_ROOT", os.path.expanduser("~/kokoro_alba/kikiri-tts"))
WORK   = os.environ.get("ALBA_WORK",   os.path.expanduser("~/kokoro_alba/export_proof"))
ROOT = KIKIRI; EP = WORK
sys.path.insert(0,KIKIRI + "/StyleTTS2")
from models import load_F0_models
F0=load_F0_models(KIKIRI + "/StyleTTS2/Utils/JDC/bst.t7").eval()
to_mel=torchaudio.transforms.MelSpectrogram(n_mels=80,n_fft=2048,win_length=1200,hop_length=300)

def pitch(p):
    d,sr=sf.read(p, dtype="float32", always_2d=True)
    w=torch.from_numpy(d.T)
    if sr!=24000: w=torchaudio.functional.resample(w,sr,24000)
    m=(torch.log(1e-5+to_mel(w.mean(0)).unsqueeze(0))-(-4))/4
    with torch.no_grad(): f,_,_=F0(m.unsqueeze(1))
    a=f.squeeze().numpy().ravel(); return a[a>50]

clips=[os.path.basename(x)[:-4] for x in sorted(glob.glob(EP+"/sweep/s1.0/*.wav"))]
real={}
for c in clips:
    src=f"{ROOT}/dataset/audio/alba/{c}.wav"
    if os.path.exists(src): real[c]=pitch(src)
R=np.concatenate(list(real.values()))
print(f"REAL   median {np.median(R):7.1f} Hz   (n={len(R)})\n")
print(f"{'scale':>7}{'p.norm':>9}{'median':>9}{'vs real':>9}{'semitones':>11}{'dur':>8}")
for d in sorted(glob.glob(EP+"/sweep/s*"), key=lambda x: -float(os.path.basename(x)[1:])):
    sc=float(os.path.basename(d)[1:])
    vs=[]; dur=0
    for c in clips:
        p=f"{d}/{c}.wav"
        if os.path.exists(p):
            vs.append(pitch(p))
            import soundfile as sf; dur+=len(sf.read(p)[0])/24000
    S=np.concatenate(vs)
    st=12*np.log2(np.median(S)/np.median(R))
    print(f"{sc:>7}{1.869*sc:>9.3f}{np.median(S):>9.1f}{np.median(S)-np.median(R):>+9.1f}{st:>+11.2f}{dur/len(clips):>8.2f}")
