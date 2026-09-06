#!/usr/bin/env python3
import sys, os, glob, numpy as np, soundfile as sf
KIKIRI = os.environ.get("KIKIRI_ROOT", os.path.expanduser("~/kokoro_alba/kikiri-tts"))
WORK   = os.environ.get("ALBA_WORK",   os.path.expanduser("~/kokoro_alba/export_proof"))
ROOT = KIKIRI; EP = WORK
def zcr(x): return float(np.mean(np.abs(np.diff(np.sign(x)))>0))
clips=[os.path.basename(x)[:-4] for x in sorted(glob.glob(EP+"/sweep/s1.0/*.wav"))]
rd=[]; rz=[]
for c in clips:
    p=f"{ROOT}/dataset/audio/alba/{c}.wav"
    if os.path.exists(p):
        x,sr=sf.read(p); x=x if x.ndim==1 else x.mean(1)
        rd.append(len(x)/sr); rz.append(zcr(x))
print(f"REAL   mean dur {np.mean(rd):.2f}s   zcr {np.mean(rz):.4f}")
print(f"\n{'scale':>7}{'dur':>8}{'vs real':>9}{'zcr':>9}{'rms':>9}{'peak':>8}")
for d in sorted(glob.glob(EP+"/sweep/s*"), key=lambda x:-float(os.path.basename(x)[1:])):
    sc=float(os.path.basename(d)[1:]); ds=[];zs=[];rs=[];ps=[]
    for c in clips:
        p=f"{d}/{c}.wav"
        if os.path.exists(p):
            x,sr=sf.read(p); x=x if x.ndim==1 else x.mean(1)
            ds.append(len(x)/sr); zs.append(zcr(x)); rs.append(np.sqrt((x**2).mean())); ps.append(np.abs(x).max())
    print(f"{sc:>7}{np.mean(ds):>8.2f}{100*(np.mean(ds)/np.mean(rd)-1):>+8.1f}%{np.mean(zs):>9.4f}{np.mean(rs):>9.4f}{np.mean(ps):>8.3f}")
