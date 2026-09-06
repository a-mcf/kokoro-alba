# Alba — a Scottish voice for Kokoro-82M

Self-contained. Runs on stock `pip install kokoro` — no fork, no vendored
package. 24 kHz mono, CPU only, faster than real time on two threads.

## Quick start

```bash
./install.sh
venv/bin/python alba_say.py "Aye, it's a bonnie morning for a walk by the water." out.wav
```

`install.sh` builds a venv beside this file, installs the four dependencies, and
runs a smoke test. Expect roughly `3.40s zcr=0.14`. **`zcr` at 0.25 or above
means static rather than speech** — see Troubleshooting.

If you are reading this inside a git checkout rather than the release bundle,
`alba_stock.pth` is not in git. Fetch it first:

```bash
curl -L -o alba_stock.pth \
  https://github.com/a-mcf/kokoro-alba/releases/latest/download/alba_stock.pth
```

## What is here

| file | size | what |
|---|---|---|
| `alba_stock.pth` | 312 MB | the fine-tuned model, key-converted for stock kokoro |
| `alba_voicepack_pitchfix.pt` | 512 KB | the voicepack — pitch-calibrated, and the default |
| `config.json` | 2 KB | Kokoro model config |
| `alba_say.py` | | CLI: `alba_say.py "text" out.wav` |
| `alba_tts.py` | | `--text-file` / `--out`, for wiring into an agent |
| `install.sh` | | builds the venv and smoke-tests it |

The git repo also carries `alba_voicepack.pt`, the *uncalibrated* extraction that
renders +3.1 semitones sharp. It is deliberately **not** in the release bundle —
it is a research artifact for the calibration write-up, and using it by mistake
produces a confidently wrong voice with no error.

## Two entry points

```bash
# interactive
venv/bin/python alba_say.py "text to speak" out.wav

# command-provider interface, for an agent or gateway
venv/bin/python alba_tts.py --text-file in.txt --out out.wav
```

Both print duration and a zero-crossing rate. Speech sits near 0.13.

## Calibrated defaults — do not change casually

`--speed 1.25` and `alba_voicepack_pitchfix.pt` were measured against the
speaker's own recordings, not chosen by ear alone. The raw pack renders **+3.1
semitones sharp**; speed 1.0 runs **16% longer** than she does. Both numbers are
specific to this checkpoint.

## You do not need the base Kokoro-82M weights

`repo_id=` appears in both scripts but is only consulted when `config=` and
`model=` are absent, and they never are. Verified by running with an empty HF
cache: correct audio, nothing downloaded. Inference is fully offline.

## Troubleshooting

**`zcr` 0.33+, output is static.** The model file is not the key-converted one.
`KModel.__init__` swallows a state-dict key mismatch in a bare `except`, retries
with `strict=False`, and leaves 318 of 688 parameters randomly initialised — so
nothing raises. Use the `alba_stock.pth` shipped here. Do not try to fix it by
installing a different `kokoro` version.

**Speech that is almost but not quite her.** You are running the *stock* Kokoro
model with Alba's voicepack. A generic Kokoro wrapper loads
`KModel(repo_id="hexgrad/Kokoro-82M")` with no `model=` and swaps only the style
vector. This voice is a fine-tuned **model plus** its voicepack; the pack alone
gets you stock speech with a hint of her, and no error.

**`No package installer found`.** spaCy could not download `en_core_web_sm`. If
the venv was built with `uv`, also `uv pip install pip` into it.

**`python3 -m venv` fails.** Missing `ensurepip`, and you may not have sudo.
Use `uv venv venv --python 3.11`, then `uv pip install --python venv/bin/python pip`.

## Credit

The voice is a real person's, recorded by researchers who released the corpus
properly. See `NOTICE` — it is short and it matters. The corpus is CC BY 4.0 and
adds a term plain CC BY does not: it preserves the voice talent's moral rights,
including the right of integrity, and forbids uses derogatory to the voice
talent. These weights inherit that.

> Valentini-Botinhao, Cassia; Yamagishi, Junichi. (2019). *Alba speech corpus*
> [dataset]. University of Edinburgh. https://doi.org/10.7488/ds/2506

Apache 2.0. Full project, training recipe and reproduction guide:
https://github.com/a-mcf/kokoro-alba
