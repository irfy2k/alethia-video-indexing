#!/usr/bin/env bash
# Dev helper: download the first 50 FineVideo parquet files to D: (WSL /mnt/d).
set -e
source "$HOME/alethia-venv/bin/activate"
export HF_HOME=/mnt/d/hf_cache
# Token may have been saved under the default HF home if the shell predates HF_HOME.
[ -f "$HF_HOME/token" ] || { [ -f "$HOME/.cache/huggingface/token" ] && export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"; }
hf auth whoami >/dev/null
hf download HuggingFaceFV/finevideo --repo-type dataset --include "data/train-000[0-4]*" --local-dir /mnt/d/finevideo
ls /mnt/d/finevideo/data | wc -l
