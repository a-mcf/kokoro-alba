# The runtime

Self-contained. Both scripts default to finding their model, voicepack and config
next to themselves, so this directory can be copied anywhere as a unit.

## Get the model

The 312 MB checkpoint is not in git. It is attached to the
[latest release](https://github.com/a-mcf/kokoro-alba/releases/latest):

```bash
curl -L -o alba_stock.pth \
  https://github.com/a-mcf/kokoro-alba/releases/latest/download/alba_stock.pth
```

Files, once complete:

| file | size | what |
|---|---|---|
| `alba_stock.pth` | 312 MB | the fine-tuned model, key-converted for stock kokoro |
| `alba_voicepack_pitchfix.pt` | 512 KB | **the shipped voicepack** — pitch-calibrated |
| `alba_voicepack.pt` | 512 KB | the raw extracted pack, +3.1 semitones sharp |
| `config.json` | 2 KB | Kokoro model config |
| `alba_say.py` | | CLI |
| `alba_tts.py` | | `--text-file` / `--out`, for command-provider wiring |

## Environment

```bash
python3 -m venv venv
venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
venv/bin/pip install kokoro==0.9.4 soundfile
venv/bin/python -m spacy download en_core_web_sm
```

On a box where `python3 -m venv` fails for want of `ensurepip` and you have no
sudo, `uv` works — but `uv pip install pip` into the venv as well, or spaCy cannot
self-download the model and reports `No package installer found`.

## Smoke test

```bash
printf 'It makes a huge difference to me and my family.' > /tmp/t.txt
venv/bin/python alba_tts.py --text-file /tmp/t.txt --out /tmp/alba.wav
```

Expect a duration and `zcr=0.13`-ish. **`zcr` at 0.25 or above means static**, and
almost always means `alba_stock.pth` is not the key-converted file — run it through
`tools/convert_to_stock.py`. Do not try to fix static by installing a different
`kokoro`.

## Wiring it into an agent gateway

Anything that supports a shell command provider:

```yaml
tts:
  provider: alba
  providers:
    alba:
      type: command
      command: /path/to/alba/venv/bin/python /path/to/alba/alba_tts.py --text-file {input_path} --out {output_path}
      output_format: wav
      voice_compatible: true
```

`alba_tts.py` suppresses kokoro 0.9.4's deprecation noise so the provider's stderr
stays clean, and defaults to 2 torch threads (`--threads` or `ALBA_THREADS`).

**A generic Kokoro wrapper will not work.** Those load
`KModel(repo_id="hexgrad/Kokoro-82M")` — the *stock* base weights — and swap only
the style vector via `--voice`. This voice is a fine-tuned **model plus** its
voicepack. Pointing a stock-model wrapper at `alba_voicepack_pitchfix.pt` produces
confident, plausible, wrong audio: stock speech with a hint of her, and no error.

## Calibrated defaults

`--speed 1.25` and the `_pitchfix` pack were measured, not chosen. They are
specific to this checkpoint. See the repo README.
