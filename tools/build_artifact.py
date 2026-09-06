#!/usr/bin/env python3
"""Build the shippable artifact from a finished training run.

    build_artifact.py --kikiri-root PATH --stage2 EPOCH.pth --stage1 EPOCH.pth \
                      --out-dir alba/

Three steps, none of which need a GPU:

  1. the recipe's convert_checkpoint: training checkpoint -> Kokoro format
     (pulls out bert, bert_encoder, predictor, text_encoder, decoder)
  2. convert_to_stock: rename weight-norm keys so stock PyPI kokoro can load it
  3. the recipe's extract_voicepack.py: average style vectors over 200 clips

Produces `alba_stock.pth` and `alba_voicepack.pt` in --out-dir. The voicepack is
the RAW one — it will be sharp. Measure the offset and scale it with
apply_pitch_scale.py; see calibration/README.md.

--stage1 is passed to the extractor as --style-encoder-model, because stage 2
training can degrade the style_encoder (spectral_norm buffer drift). Omit it only
if you know you want stage 2's own style_encoder.
"""
import argparse
import functools
import shutil
import subprocess
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # keep our output interleaved with subprocesses'

ap = argparse.ArgumentParser()
ap.add_argument("--kikiri-root", required=True, help="checkout of semidark/kikiri-tts")
ap.add_argument("--stage2", required=True, help="final stage 2 checkpoint (epoch_2nd_*.pth)")
ap.add_argument("--stage1", default=None, help="stage 1 checkpoint for the style encoder")
ap.add_argument("--audio-dir", default=None, help="default: <kikiri-root>/dataset/audio/alba")
ap.add_argument("--out-dir", default="alba")
ap.add_argument("--num-samples", type=int, default=200)
ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
a = ap.parse_args()

root = Path(a.kikiri_root).resolve()
out = Path(a.out_dir).resolve()
out.mkdir(parents=True, exist_ok=True)
audio = Path(a.audio_dir).resolve() if a.audio_dir else root / "dataset/audio/alba"

for p in (root / "scripts/test_inference.py", root / "scripts/extract_voicepack.py"):
    if not p.exists():
        sys.exit(f"not a kikiri-tts checkout: missing {p}")

# 1. training checkpoint -> Kokoro format
sys.stdout.reconfigure(line_buffering=True)
sys.path[:0] = [str(root / "scripts"), str(root)]
from test_inference import convert_checkpoint

kokoro_fmt = out / "_kokoro_format.pth"
convert_checkpoint(a.stage2, str(kokoro_fmt))

# 2. -> stock kokoro key convention
here = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(here / "convert_to_stock.py"),
                str(kokoro_fmt), str(out / "alba_stock.pth")], check=True)
kokoro_fmt.unlink()

# 3. voicepack
cmd = [sys.executable, str(root / "scripts/extract_voicepack.py"),
       "--model", a.stage2,
       "--audio-dir", str(audio),
       "--output", str(out / "alba_voicepack.pt"),
       "--num-samples", str(a.num_samples),
       "--device", a.device]
if a.stage1:
    cmd += ["--style-encoder-model", a.stage1]
subprocess.run(cmd, check=True)

# 4. the Kokoro model config the runtime needs alongside the weights
cfg = root / "training/config.json"
if cfg.exists():
    shutil.copy2(cfg, out / "config.json")
else:
    print(f"WARNING: {cfg} not found -- copy config.json into {out} yourself")

print(f"\n-> {out}/alba_stock.pth")
print(f"-> {out}/alba_voicepack.pt  (RAW -- calibrate before shipping)")
print(f"-> {out}/config.json")
print("\nNext: verify the weights actually landed, then calibrate pitch.")
print(f"  python {here / 'verify_load.py'} {out}/config.json {out}/alba_stock.pth built")
