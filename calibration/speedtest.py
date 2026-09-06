#!/usr/bin/env python3
import os, torch, soundfile as sf, warnings, random
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
KIKIRI = os.environ.get("KIKIRI_ROOT", os.path.expanduser("~/kokoro_alba/kikiri-tts"))
WORK   = os.environ.get("ALBA_WORK",   os.path.expanduser("~/kokoro_alba/export_proof"))
ROOT = KIKIRI; EP = WORK
text={}
for line in open(ROOT+"/dataset/metadata.csv",encoding="utf-8"):
    p=line.rstrip("\n").split("|")
    if len(p)>=2: text[p[0].strip()]=p[1].strip()
import glob
clips=[os.path.basename(x)[:-4] for x in sorted(glob.glob(EP+"/sweep/s1.0/*.wav"))]
from kokoro import KModel, KPipeline
km=KModel(repo_id="hexgrad/Kokoro-82M", config=EP+"/config.json", model=EP+"/alba_stock.pth").eval()
pipe=KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=km)
pack=torch.load(EP+"/alba_voicepack_pitchfix.pt", map_location="cpu", weights_only=True)
for spd in [1.16, 1.25]:
    d=f"{EP}/speed/sp{spd}"; os.makedirs(d, exist_ok=True)
    for c in clips:
        t=text.get(f"alba/{c}.wav")
        if not t: continue
        for _,_ps,a in pipe(t, voice=pack, speed=spd):
            sf.write(f"{d}/{c}.wav", a.detach().cpu().numpy().astype("float32"), 24000); break
    print("done speed", spd, "->", d)
