# Serving

Everything here was measured on 4 cores of a Xeon-class VM, CPU only, no GPU,
2026-09-06.

## You do not need the base model

The scripts pass `repo_id="hexgrad/Kokoro-82M"` to `KModel` and `KPipeline`, which
looks like it wants the base weights. It does not. `repo_id` is only consulted
when `config=` and `model=` are absent, and both are always supplied here.

Verified: running `alba_tts.py` with `HF_HOME` pointed at an empty directory
produced correct audio and **downloaded nothing** — the cache was still empty
afterwards. There is no network call at inference time, so `HF_HUB_OFFLINE=1` is
safe and an air-gapped box is fine.

What you need is the four files in `alba/`: `alba_stock.pth`,
`alba_voicepack_pitchfix.pt`, `config.json`, and one of the two entry scripts.

## Dependencies

```bash
python3 -m venv venv
venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
venv/bin/pip install kokoro==0.9.4 soundfile
venv/bin/python -m spacy download en_core_web_sm
```

That is the whole list. Two things worth knowing about it:

**spaCy `en_core_web_sm` is not optional.** misaki does grapheme-to-phoneme
conversion and requires it. A venv built without pip cannot self-download the
model and fails with `No package installer found`, which reads like a packaging
problem rather than a TTS one. If you built the venv with `uv`, also
`uv pip install pip` into it.

**espeak-ng ships inside the venv.** `kokoro` pulls `misaki`, which pulls
`espeakng-loader`, which bundles `libespeak-ng.so` and its data directory. You do
**not** need the system `espeak-ng` package. It is used as a fallback for words
outside misaki's dictionary — most sentences never touch it, but unusual words do,
so do not strip it out.

## Cost

The headline number depends on when you measure, so here is the whole curve —
`VmRSS` from `/proc/self/status`, CPU-only, 2 threads:

| stage | RSS | peak so far |
|---|---|---|
| bare interpreter | 8 MB | 8 MB |
| `import torch` | 212 MB | 212 MB |
| `import kokoro` | 375 MB | 375 MB |
| **after loading the model** | **1010 MB** | **1231 MB** |
| after `gc.collect()` + `malloc_trim` | 918 MB | 1231 MB |
| pipeline + voicepack ready | 1031 MB | 1231 MB |
| after 5 syntheses | 1223 MB | 1359 MB |
| after 5, then gc + trim | 1073 MB | 1359 MB |

**Working set is roughly 0.9–1.1 GB; transient peaks reach 1.25–1.35 GB.**

The 1231 MB spike at load is exactly what it looks like: `torch.load` materialises
the 312 MB checkpoint while `KModel` is simultaneously holding its own 312 MB of
parameters. It is transient — a `gc.collect()` plus `malloc_trim(0)` after startup
gives back about 90 MB, and another 150 MB after a few syntheses, so a
long-running server settles near 1 GB rather than growing without bound.

For scale, the model itself is **81.8M parameters = 312 MB in fp32**. Everything
above that is torch's own footprint (212 MB before you load anything), spaCy and
misaki, and allocator slack.

**Size a box for 1.5 GB per worker.** The earlier claim that this needs ~312 MB
resident was wrong — that was the file size mislabelled as memory.

Cold start is **5.7s** — 3.9s importing torch and kokoro, 1.0s loading the model,
0.7s building the pipeline. Thread scaling is close to linear up to 4:

| threads | wall for a 3.92s utterance | RTF |
|---|---|---|
| 1 | 4.04s | 1.03 |
| 2 | 2.10s | 0.54 |
| 4 | 1.31s | 0.33 |

Set it with `--threads` or `ALBA_THREADS`. The default is 2, which is a shared-box
compromise, not a recommendation — raise it if you have the cores to spare.

## Pick a shape: per-call or persistent

**The 5.7s cold start dominates everything.** A 3.9-second utterance takes 2.1s of
actual synthesis and 5.7s of getting ready. That single fact decides how you should
run this.

### Per-call (`alba_tts.py`)

Simple, stateless, and what the command-provider interface below expects. Every
invocation pays the full 5.7s. Fine for batch rendering, notifications, cron, or
anything where a ~8-second turnaround is invisible.

```bash
venv/bin/python alba/alba_tts.py --text-file in.txt --out out.wav
```

### Persistent

For anything interactive — an agent that talks back, a doorbell, a voice assistant
— load once and keep the process alive. The import and model load happen at
startup, and each utterance then costs only its RTF.

The pieces are three lines; wrap them in whatever server you already run:

```python
import torch, soundfile as sf
from kokoro import KModel, KPipeline

torch.set_num_threads(4)
km = KModel(repo_id="hexgrad/Kokoro-82M",
            config="alba/config.json", model="alba/alba_stock.pth").eval()
pipe = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=km)
voice = torch.load("alba/alba_voicepack_pitchfix.pt",
                   map_location="cpu", weights_only=True)

def say(text, out, speed=1.25):
    for _, _ps, audio in pipe(text, voice=voice, speed=speed):
        sf.write(out, audio.detach().cpu().numpy().astype("float32"), 24000)
        return

say("Aye, it is a bonnie morning.", "out.wav")
```

`lang_code="b"` is load-bearing — it selects misaki `en.G2P(british=True)`, the G2P
the training labels were built with. A different `lang_code` feeds the model
phonemes it never saw.

`KPipeline` is not thread-safe in any way this project has verified. Serialize
calls, or run one process per worker and pay ~1 GB each.

## Wiring into an agent gateway

Anything that supports a shell command provider:

```yaml
tts:
  provider: alba
  providers:
    alba:
      type: command
      command: /path/to/venv/bin/python /path/to/alba/alba_tts.py --text-file {input_path} --out {output_path}
      output_format: wav
      voice_compatible: true
```

`alba_tts.py` suppresses kokoro 0.9.4's deprecation warnings so the provider's
stderr stays clean, writes 24 kHz mono WAV, and prints one line of
`path duration zcr` to stdout.

If the gateway runs as a systemd **user** service, note that a session running
inside the gateway cannot restart it — you need an outside shell.

## Verifying a deployment

Do not verify by absence of errors. This project's recurring failure mode is
confident, plausible, wrong audio.

```bash
printf 'It makes a huge difference to me and my family.' > /tmp/t.txt
venv/bin/python alba/alba_tts.py --text-file /tmp/t.txt --out /tmp/alba.wav
```

Expect roughly `3.40s zcr=0.1406`. **Check the zcr**: speech lands near 0.13,
static at 0.33 and above. `alba_tts.py` prints a warning above 0.25, but read the
number yourself.

Two failure modes that produce audio and no error:

**zcr 0.33+, static.** `alba_stock.pth` is not the key-converted checkpoint.
`KModel.__init__` swallows a key mismatch in a bare `except`, retries with
`strict=False`, and leaves 318 of 688 parameters randomly initialised. Run the
checkpoint through `tools/convert_to_stock.py`. Do not try to fix this by
installing a different `kokoro` version.

**Speech that is almost but not quite her.** You are running the *stock* Kokoro
model with Alba's voicepack. A generic Kokoro wrapper loads
`KModel(repo_id="hexgrad/Kokoro-82M")` with no `model=` and swaps only the style
vector via `--voice`. This voice is a fine-tuned **model plus** its voicepack —
the pack alone gets you stock speech with a hint of her, and nothing raises.

To be certain, check tensor by tensor rather than trusting the loader:

```bash
venv/bin/python tools/verify_load.py alba/config.json alba/alba_stock.pth deployed
```

Expect **548 landed, 0 mismatched, 0 orphaned**. It also reports ~140 model params
the checkpoint did not cover — AdaLayerNorm `.norm` affine weights, default-init on
every path including the known-good one. Normal.

## Calibration

`--speed 1.25` and `alba_voicepack_pitchfix.pt` were measured against the speaker's
own recordings and are specific to this checkpoint. Changing them is a quality
decision, not a config tweak. If you retrain, re-derive both — see
`calibration/README.md`.
