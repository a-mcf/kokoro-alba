#!/usr/bin/env bash
set -e
# KIKIRI_ROOT = your checkout of github.com/semidark/kikiri-tts
KIKIRI_ROOT="${KIKIRI_ROOT:?set KIKIRI_ROOT to the kikiri-tts checkout}"
cd "$KIKIRI_ROOT/StyleTTS2"
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$KIKIRI_ROOT/../hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
accelerate launch --num_processes 1 train_first.py --config_path ../configs/config_alba_ft.yml
