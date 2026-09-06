#!/usr/bin/env python3
"""Does the prosodic half's MAGNITUDE control pitch? Sweep it and synthesize."""
import os, torch, numpy as np, soundfile as sf, warnings, random
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
KIKIRI = os.environ.get("KIKIRI_ROOT", os.path.expanduser("~/kokoro_alba/kikiri-tts"))
WORK   = os.environ.get("ALBA_WORK",   os.path.expanduser("~/kokoro_alba/export_proof"))
ROOT = KIKIRI; EP = WORK
OUT=EP+"/sweep"; os.makedirs(OUT, exist_ok=True)

text={}
for line in open(ROOT+"/dataset/metadata.csv", encoding="utf-8"):
    p=line.rstrip("\n").split("|")
    if len(p)>=2: text[p[0].strip()]=p[1].strip()
val=[l.strip().split("|")[0] for l in open(ROOT+"/training/val_list.txt",encoding="utf-8") if l.strip()]
random.seed(0); random.shuffle(val)

from kokoro import KModel, KPipeline
km=KModel(repo_id="hexgrad/Kokoro-82M", config=EP+"/config.json", model=EP+"/alba_stock.pth").eval()
pipe=KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=km)
base=torch.load(EP+"/alba_voicepack.pt", map_location="cpu", weights_only=True)

# pick the same 8 clips we already have REAL audio for
pairs=[]
for rel in val:
    stem=rel.split("/")[-1].replace(".wav","")
    t=text.get(rel) or text.get(stem) or text.get(stem+".wav")
    if t and os.path.exists(f"{ROOT}/dataset/audio/{rel}"): pairs.append((stem,rel,t))
    if len(pairs)>=8: break

SCALES=[1.0, 0.75, 0.5, 0.35, 0.2]
v=base[0,0].clone()
print("orig prosodic norm %.3f"%v[128:].norm())
for sc in SCALES:
    pk=base.clone()
    pk[:,:,128:]=base[:,:,128:]*sc
    d=f"{OUT}/s{sc}"; os.makedirs(d, exist_ok=True)
    for stem,rel,t in pairs:
        for _,_ps,audio in pipe(t, voice=pk, speed=1):
            sf.write(f"{d}/{stem}.wav", audio.detach().cpu().numpy().astype("float32"), 24000); break
    print(f"scale {sc}: prosodic norm {(v[128:]*sc).norm():.3f}  -> {d}")
    if sc==1.0: torch.save(pk, EP+"/pack_s1.0.pt")
print("clips:", [p[0] for p in pairs])
