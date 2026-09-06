#!/usr/bin/env python
"""Convert a Kokoro checkpoint to the weight_norm key convention stock PyPI
kokoro expects. Nothing is degraded: the tensors are untouched, only renamed.

Training (and upstream hexgrad/kokoro main) emit the newer
torch.nn.utils.parametrizations.weight_norm keys; stock kokoro 0.9.4 uses the
older torch.nn.utils.weight_norm keys.

new: <m>.parametrizations.weight.original0  (magnitude g)
     <m>.parametrizations.weight.original1  (direction v)
old: <m>.weight_g / <m>.weight_v
Exact 1:1 tensor correspondence; no values change.
"""
import sys, torch, collections

SRC = sys.argv[1]; DST = sys.argv[2]

def conv(k):
    if k.endswith(".parametrizations.weight.original0"):
        return k[: -len(".parametrizations.weight.original0")] + ".weight_g"
    if k.endswith(".parametrizations.weight.original1"):
        return k[: -len(".parametrizations.weight.original1")] + ".weight_v"
    return k

sd = torch.load(SRC, map_location="cpu", weights_only=True)
out, n = {}, 0
for top, sub in sd.items():
    new = collections.OrderedDict()
    for k, v in sub.items():
        nk = conv(k)
        if nk != k: n += 1
        assert nk not in new, f"collision {nk}"
        new[nk] = v
    out[top] = new
    assert len(new) == len(sub), top

torch.save(out, DST)
print(f"renamed {n} keys across {len(out)} submodules -> {DST}")
