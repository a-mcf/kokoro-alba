# Samples

Rendered 2026-09-06 with the shipped defaults — `alba/alba_tts.py`, no arguments
beyond `--text-file` and `--out`, so `--speed 1.25` and
`alba_voicepack_pitchfix.pt`. 24 kHz mono, CPU, 2 threads.

| file | text | duration | zcr |
|---|---|---|---|
| `bonnie_morning.wav` | "Aye, it is a bonnie morning for a walk by the water, is it not?" | 3.95s | 0.1199 |
| `edinburgh_fog.wav` | "The Edinburgh fog curled round the old castle gates at dawn." | 3.92s | 0.1286 |
| `family.wav` | "It makes a huge difference to me and my family." | 3.40s | 0.1404 |

The zcr column is the health check, not a quality metric: speech lands near 0.13,
static at 0.33 and above.
