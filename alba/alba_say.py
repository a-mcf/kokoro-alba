#!/usr/bin/env python
"""Alba TTS on STOCK kokoro. No fork, no vendored package, no sys.path games.

    alba_say.py "text" out.wav [--speed S] [--model M] [--voice V] [--config C]

Calibrated defaults (measured against the real speaker, 2026-08-31):
  --speed 1.25   matches her utterance length within 3%
  --voice        pitch-corrected pack; the raw pack is +3.1 semitones sharp

NOTE: no phoneme rewriting. Dark-L vocalization was tested and REJECTED
(2026-08-31) — dropping the L ruins "all" -> "aw", and her real articulation is
a softened L, which discrete phoneme substitution cannot express.

Requires a checkpoint converted by convert_to_stock.py: stock kokoro 0.9.4 uses
torch.nn.utils.weight_norm (weight_g/weight_v); training emits the newer
parametrizations form. Unconverted weights load silently-partial -> static.
"""
import argparse, os, sys, torch, numpy as np, soundfile as sf, warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("ALBA_THREADS", "2")))

D = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("text"); p.add_argument("out", nargs="?", default="/tmp/alba.wav")
p.add_argument("--model",  default=os.path.join(D, "alba_stock.pth"))
p.add_argument("--voice",  default=os.path.join(D, "alba_voicepack_pitchfix.pt"))
p.add_argument("--config", default=os.path.join(D, "config.json"))
p.add_argument("--speed", type=float, default=1.25)
a = p.parse_args()

from kokoro import KModel, KPipeline
km = KModel(repo_id="hexgrad/Kokoro-82M", config=a.config, model=a.model).eval()
# lang_code "b" -> misaki en.G2P(british=True): the G2P the corpus was phonemized with.
pipe = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=km)
voice = torch.load(a.voice, map_location="cpu", weights_only=True)   # [510,1,256]

for _, ps, audio in pipe(a.text, voice=voice, speed=a.speed):
    au = audio.detach().cpu().numpy().astype("float32")
    sf.write(a.out, au, 24000)
    zcr = float(np.mean(np.abs(np.diff(np.sign(au))) > 0))
    verdict = "speech" if zcr < 0.25 else "STATIC - checkpoint not converted?"
    print("%s  %.2fs  zcr=%.4f  (%s)" % (a.out, len(au) / 24000, zcr, verdict))
    break
