#!/usr/bin/env python3
"""Synthesize the SAME sentences as real val clips, for a like-for-like pitch compare."""
import os, sys, csv, torch, numpy as np, soundfile as sf, warnings
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
KIKIRI = os.environ.get("KIKIRI_ROOT", os.path.expanduser("~/kokoro_alba/kikiri-tts"))
WORK   = os.environ.get("ALBA_WORK",   os.path.expanduser("~/kokoro_alba/export_proof"))
ROOT = KIKIRI; EP = WORK
OUT=os.path.join(EP,"pitchcmp"); os.makedirs(OUT, exist_ok=True)

text={}
for line in open(ROOT+"/dataset/metadata.csv", encoding="utf-8"):
    p=line.rstrip("\n").split("|")
    if len(p)>=2: text[p[0].strip()]=p[1].strip()

val=[l.strip().split("|")[0] for l in open(ROOT+"/training/val_list.txt",encoding="utf-8") if l.strip()]
import random; random.seed(0); random.shuffle(val)

from kokoro import KModel, KPipeline
km=KModel(repo_id="hexgrad/Kokoro-82M", config=EP+"/config.json", model=EP+"/alba_stock.pth").eval()
pipe=KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=km)
voice=torch.load(EP+"/alba_voicepack.pt", map_location="cpu", weights_only=True)

n=0
for rel in val:
    stem=rel.split("/")[-1].replace(".wav","")
    t=None
    for k in (rel, stem, stem+".wav"):
        if k in text: t=text[k]; break
    if not t: continue
    src=f"{ROOT}/dataset/audio/{rel}"
    if not os.path.exists(src): continue
    try:
        for _,_ps,audio in pipe(t, voice=voice, speed=1):
            sf.write(f"{OUT}/{stem}_SYNTH.wav", audio.detach().cpu().numpy().astype("float32"), 24000)
            break
    except Exception as e:
        print("skip", stem, e); continue
    os.system(f'cp "{src}" "{OUT}/{stem}_REAL.wav"')
    n+=1
    if n>=14: break
print("pairs:", n, "->", OUT)
