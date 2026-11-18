#!/usr/bin/env bash
# One-time dev setup inside WSL Ubuntu 22.04 (not used by grader).
set -e
python3 -m pip install --user -q uv
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 "$HOME/alethia-venv" -q --allow-existing
source "$HOME/alethia-venv/bin/activate"
uv pip install -q -U huggingface_hub
mkdir -p /mnt/d/finevideo /mnt/d/hf_cache
grep -q HF_HOME ~/.bashrc || cat >> ~/.bashrc <<'RC'
export HF_HOME=/mnt/d/hf_cache
export PATH="$HOME/.local/bin:$PATH"
RC
python --version; hf version 2>/dev/null || hf --help | head -1
