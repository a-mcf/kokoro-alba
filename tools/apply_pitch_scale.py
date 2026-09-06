#!/usr/bin/env python3
"""Scale the prosodic half of a Kokoro voicepack to correct a global pitch offset.

    apply_pitch_scale.py IN.pt OUT.pt 0.35

A Kokoro voicepack is [510, 1, 256]. The first 128 dims are timbre; the last 128
are prosody, and their MAGNITUDE sets the pitch level. Extracting a pack from a
fine-tune can leave it systematically sharp — this one came out +3.1 semitones
above the speaker's own recordings. Scaling the prosodic half down corrects it.

The scale factor is specific to a checkpoint and a speaker. Re-derive it by
measurement (calibration/scale_sweep.py then calibration/sweep_measure.py);
never carry the number across a retrain.

Trade-off, measured: scaling also flattens the pitch contour. Going from 1.0 to
0.35 moved declination slope from -43.8 to -25.7 Hz/s. It buys a correct pitch
level and a correct total fall at the cost of contour rate.
"""
import sys

import torch

if len(sys.argv) != 4:
    sys.exit(__doc__)

src, dst, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])

pack = torch.load(src, map_location="cpu", weights_only=True)
if pack.ndim != 3 or pack.shape[-1] != 256:
    sys.exit(f"expected a [N, 1, 256] voicepack, got {tuple(pack.shape)}")

out = pack.clone()
out[:, :, 128:] = pack[:, :, 128:] * scale

torch.save(out, dst)
print(f"prosodic norm {pack[0, 0, 128:].norm():.4f} -> {out[0, 0, 128:].norm():.4f} "
      f"(x{scale})  timbre untouched  -> {dst}")
