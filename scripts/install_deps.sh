#!/usr/bin/env bash
# Dev helper: install runtime deps into the WSL venv. Torch pinned to a CUDA 12 build
# because the grader's driver only guarantees CUDA 12.
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/alethia-venv/bin/activate"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install ultralytics av pyarrow numpy tensorrt-cu12 onnx onnxslim
python - <<'PY'
import torch, ultralytics, av, pyarrow, tensorrt
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("ultralytics", ultralytics.__version__, "av", av.__version__, "pyarrow", pyarrow.__version__, "trt", tensorrt.__version__)
PY
