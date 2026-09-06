#!/usr/bin/env python3
"""Test the trained Scottish (Alba) voicepack in Kokoro format."""
import argparse, os, sys, torch
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_kokoro_submodule = _repo_root / "kokoro"
if _kokoro_submodule.exists() and str(_kokoro_submodule) not in sys.path:
    sys.path.insert(0, str(_kokoro_submodule))

from kokoro import KModel, KPipeline

TEST_SENTENCES = [
    "The Edinburgh fog curled round the old castle gates at dawn.",
    "She walked along the loreshore, her scarf streaming in the salt-wind breeze.",
    "Aye, it's a bonnie morning for a walk by the water, is it not?",
    "The hills stood quiet beneath a sky the colour of polished steel.",
    "I'll tell you a story, lad, of the old days when the city slept.",
    "Between the fields and the river, the summer light went golden and slow.",
]

def convert(checkpoint_path, output_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]
    def ensure_prefix(sd):
        return {("module." + k if not k.startswith("module.") else k): v for k, v in sd.items()}
    kok = {}
    for key in ["bert","bert_encoder","predictor","text_encoder","decoder"]:
        if key in net:
            kok[key] = ensure_prefix(net[key])
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kok, str(out))
    return str(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--voicepack", required=True)
    p.add_argument("--config", default=str(_repo_root/"configs/config_alba_ft.yml"))
    p.add_argument("--out", default=str(_repo_root/"test_output/alba"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    model_path = convert(args.checkpoint, os.path.join(args.out, "kokoro-alba_model.pth"))
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", model=model_path)
    kmodel = kmodel.to(args.device).eval()

    pipeline = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=kmodel)  # b = en_GB

    voice = torch.load(args.voicepack, map_location="cpu", weights_only=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import numpy as np, soundfile as sf
    for i, text in enumerate(TEST_SENTENCES):
        for _, ps, audio in pipeline(text, voice=voice, speed=1):
            wav = out / f"alba_{i+1:02d}.wav"
            sf.write(str(wav), audio.detach().cpu().numpy().astype("float32"), 24000)
            print(f"saved {wav} ({len(audio)/24000:.1f}s)")

if __name__ == "__main__":
    main()
