# The runtime

Self-contained. Both scripts default to finding their model, voicepack and config
next to themselves, so this directory can be copied anywhere as a unit.

**Deployment guide, with measured costs and the failure modes, is in
[SERVING.md](../SERVING.md).** This page is just the contents.

## Get the model

The 312 MB checkpoint is not in git. It is attached to the
[latest release](https://github.com/a-mcf/kokoro-alba/releases/latest):

```bash
curl -L -o alba_stock.pth \
  https://github.com/a-mcf/kokoro-alba/releases/latest/download/alba_stock.pth
```

`md5 f1b6fcbc0f432cbfc852bd6f2b744d9e`

| file | size | what |
|---|---|---|
| `alba_stock.pth` | 312 MB | the fine-tuned model, key-converted for stock kokoro |
| `alba_voicepack_pitchfix.pt` | 512 KB | **the shipped voicepack** — pitch-calibrated |
| `alba_voicepack.pt` | 512 KB | the raw extracted pack, +3.1 semitones sharp |
| `config.json` | 2 KB | Kokoro model config |
| `alba_say.py` | | CLI: `alba_say.py "text" out.wav` |
| `alba_tts.py` | | `--text-file` / `--out`, for command-provider wiring |

You do **not** need the base Kokoro-82M weights. `repo_id=` appears in both scripts
but is only consulted when `config=` and `model=` are absent, and they never are.

## Smoke test

```bash
printf 'It makes a huge difference to me and my family.' > /tmp/t.txt
venv/bin/python alba_tts.py --text-file /tmp/t.txt --out /tmp/alba.wav
```

Expect roughly `3.40s zcr=0.1406`. **`zcr` at 0.25 or above means static** — see
SERVING.md, which covers that and the subtler failure where you get stock Kokoro
speech wearing a hint of her voice.

## Calibrated defaults

`--speed 1.25` and the `_pitchfix` pack were measured against the speaker's own
recordings, not chosen. They are specific to this checkpoint.
