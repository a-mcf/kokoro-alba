#!/usr/bin/env python3
"""Alba TTS — command-provider wrapper.

    alba_tts.py --text-file IN --out OUT [--voice V.pt] [--model M.pth]
                [--config C.json] [--speed S]

Matches the common `type: command` TTS provider interface ({input_path} /
{output_path}). Outputs 24 kHz mono WAV.

Runs on STOCK kokoro (pip install kokoro==0.9.4). No fork, no vendored package.

Calibrated defaults, measured against the real speaker 2026-08-31:
  speed 1.25            matches her utterance length within 3%
  *_pitchfix voicepack  the raw pack renders +3.1 semitones sharp

⚠️ The model file MUST be the stock-converted one (convert_to_stock.py).
Training emits torch.nn.utils.parametrizations.weight_norm keys; stock kokoro
0.9.4 expects the older weight_g/weight_v names. KModel swallows the mismatch
in a bare except + strict=False, so unconverted weights raise NO error and
produce static. The zcr line below is the guard: speech ~0.13, static ~0.33+.
"""
import warnings
warnings.filterwarnings("ignore")          # stock kokoro 0.9.4 emits deprecation
                                           # noise; keep the provider's stderr clean
import argparse, os, sys, numpy as np

DEF = os.environ.get("ALBA_DIR", os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--text-file", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model",  default=os.path.join(DEF, "alba_stock.pth"))
ap.add_argument("--voice",  default=os.path.join(DEF, "alba_voicepack_pitchfix.pt"))
ap.add_argument("--config", default=os.path.join(DEF, "config.json"))
ap.add_argument("--speed", type=float, default=1.25)
ap.add_argument("--threads", type=int, default=int(os.environ.get("ALBA_THREADS", "2")))
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

import torch
torch.set_num_threads(a.threads)          # shared box: do not raise casually
import soundfile as sf
from kokoro import KModel, KPipeline

for pth in (a.model, a.voice, a.config):
    if not os.path.exists(pth):
        sys.exit(f"alba_tts: missing required file: {pth}")

text = open(a.text_file, encoding="utf-8").read().strip()
if not text:
    sys.exit("alba_tts: --text-file is empty")

km = KModel(repo_id="hexgrad/Kokoro-82M", config=a.config, model=a.model).eval()
# lang_code "b" -> misaki en.G2P(british=True): the G2P the corpus was phonemized with.
pipe = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=km)
voice = torch.load(a.voice, map_location="cpu", weights_only=True)   # [510,1,256]

chunks = [audio.detach().cpu().numpy().astype("float32")
          for _, _ps, audio in pipe(text, voice=voice, speed=a.speed)]
if not chunks:
    sys.exit("alba_tts: pipeline produced no audio")
au = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
sf.write(a.out, au, 24000)

zcr = float(np.mean(np.abs(np.diff(np.sign(au))) > 0))
if zcr >= 0.25:
    print(f"alba_tts: WARNING zcr={zcr:.3f} looks like STATIC — is --model the "
          f"stock-converted checkpoint?", file=sys.stderr)
if not a.quiet:
    print(f"{a.out} {len(au)/24000:.2f}s zcr={zcr:.4f}")
