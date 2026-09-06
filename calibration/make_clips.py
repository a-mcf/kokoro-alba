#!/usr/bin/env python3
import os, torch, soundfile as sf, warnings
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
KIKIRI = os.environ.get("KIKIRI_ROOT", os.path.expanduser("~/kokoro_alba/kikiri-tts"))
WORK   = os.environ.get("ALBA_WORK",   os.path.expanduser("~/kokoro_alba/export_proof"))
ROOT = KIKIRI; EP = WORK
OUT=EP+"/clips"; os.makedirs(OUT, exist_ok=True)
PICKS=[("alba/3557.wav","We were going to have all the family together.","allfamily"),
       ("alba/1852.wav","He will hopefully lead us to great things.","willlead"),
       ("alba/2198.wav","It makes a huge difference to me and my family.","difference"),
       ("alba/2911.wav","The mood in the factory is great.","factory")]
from kokoro import KModel, KPipeline
km=KModel(repo_id="hexgrad/Kokoro-82M", config=EP+"/config.json", model=EP+"/alba_stock.pth").eval()
pipe=KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=km)
pack=torch.load(EP+"/alba_voicepack_pitchfix.pt", map_location="cpu", weights_only=True)
for rel,t,name in PICKS:
    os.system(f'cp "{ROOT}/dataset/audio/{rel}" "{OUT}/{name}_1REAL.wav"')
    for _,_ps,a in pipe(t, voice=pack, speed=1):
        sf.write(f"{OUT}/{name}_2SYNTH.wav", a.detach().cpu().numpy().astype("float32"), 24000); break
    print("ok", name, "|", t)
