#!/usr/bin/env python
"""Did the checkpoint tensors ACTUALLY land in the model, or is it random weights?

Compares every parameter of the built model against the checkpoint tensor-by-tensor.
Reports params the checkpoint did not cover -> those are randomly initialised.
"""
import sys, torch, warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

CFG, CKPT, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
if len(sys.argv) > 4: sys.path.insert(0, sys.argv[4])
import kokoro
from kokoro import KModel
print(f"[{TAG}] kokoro from: {kokoro.__file__}")

km = KModel(repo_id="hexgrad/Kokoro-82M", config=CFG, model=CKPT).eval()

raw = torch.load(CKPT, map_location="cpu", weights_only=True)
covered, mismatched, missing_in_model = 0, [], []
for top, sub in raw.items():
    msd = getattr(km, top).state_dict()
    for k, v in sub.items():
        kk = k[7:] if k.startswith("module.") else k
        if kk not in msd:
            missing_in_model.append(f"{top}.{kk}"); continue
        if msd[kk].shape != v.shape or not torch.equal(msd[kk], v):
            mismatched.append(f"{top}.{kk}")
        else:
            covered += 1

model_params, uncovered = 0, []
for top in raw.keys():
    ckpt_keys = {(k[7:] if k.startswith("module.") else k) for k in raw[top]}
    for k in getattr(km, top).state_dict():
        model_params += 1
        if k not in ckpt_keys: uncovered.append(f"{top}.{k}")

print(f"[{TAG}] ckpt tensors landed EXACTLY : {covered}")
print(f"[{TAG}] ckpt tensors MISMATCHED     : {len(mismatched)}  {mismatched[:4]}")
print(f"[{TAG}] ckpt keys not in model      : {len(missing_in_model)}  {missing_in_model[:4]}")
print(f"[{TAG}] model params RANDOM (uncovered): {len(uncovered)} of {model_params}  {uncovered[:6]}")
