#!/usr/bin/env bash
# Build the runtime venv next to this script and prove it works.
#
#   ./install.sh [venv-dir]
#
# CPU only. Needs ~1.5 GB RAM at run time and about 2 GB of disk for the venv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-$HERE/venv}"

for f in alba_stock.pth alba_voicepack_pitchfix.pt config.json; do
    [ -f "$HERE/$f" ] || { echo "missing $HERE/$f"; exit 1; }
done

PY="${PYTHON:-python3}"
"$PY" -m venv "$VENV" 2>/dev/null || {
    echo "python3 -m venv failed. If ensurepip is missing and you have no sudo, use uv:"
    echo "  uv venv \"$VENV\" --python 3.11 && uv pip install --python \"$VENV/bin/python\" pip"
    exit 1
}

"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet --index-url https://download.pytorch.org/whl/cpu torch
"$VENV/bin/pip" install --quiet kokoro==0.9.4 soundfile
# misaki does grapheme-to-phoneme and needs this model; it cannot self-download
# from a venv without pip, where it fails with "No package installer found".
"$VENV/bin/python" -m spacy download en_core_web_sm --quiet

echo
echo "Smoke test:"
printf 'It makes a huge difference to me and my family.' > "$HERE/.smoke.txt"
"$VENV/bin/python" "$HERE/alba_tts.py" --text-file "$HERE/.smoke.txt" --out "$HERE/.smoke.wav"
rm -f "$HERE/.smoke.txt"

echo
echo "Expect roughly '3.40s zcr=0.14'. zcr at 0.25 or above means static, not speech."
echo "Wrote $HERE/.smoke.wav — listen to it."
echo
echo "Use it:  $VENV/bin/python $HERE/alba_say.py \"your text here\" out.wav"
