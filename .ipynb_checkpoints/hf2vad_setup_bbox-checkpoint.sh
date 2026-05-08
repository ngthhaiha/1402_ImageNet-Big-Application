#!/usr/bin/env bash
set -euo pipefail

# This script ONLY creates a conda environment and installs packages.
# It does NOT modify your repo files.

ENV_NAME="${ENV_NAME:-hf2vad_setup_bbox}"
PY_VER="${PY_VER:-3.10}"

echo "[1/6] Checking conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH."
  exit 1
fi

# Make conda activate available in non-interactive shell
eval "$(conda shell.bash hook)"

echo "[2/6] Creating environment: ${ENV_NAME} (python=${PY_VER})"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} already exists. Reusing it."
else
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

echo "[3/6] Activating environment"
conda activate "${ENV_NAME}"

echo "[4/6] Upgrading packaging tools"
python -m pip install -U "pip<25.2" wheel
python -m pip install -U "setuptools<82"

echo "[5/6] Cleaning conflicting OpenMMLab packages"
python -m pip uninstall -y torch torchvision torchaudio mmcv mmcv-lite mmengine mmdet || true

echo "[6/6] Installing PyTorch + OpenMMLab packages"
# RTX 5060 Ti / Blackwell-oriented stack
python -m pip install \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  opencv-python-headless numpy pillow tqdm pyyaml matplotlib "setuptools<82"

# Build mmcv from source against the installed torch/cu128 stack
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

python -m pip install "mmengine==0.10.5"
python -m pip install --no-build-isolation "mmcv==2.1.0"
python -m pip install --no-deps "mmdet==3.3.0"

echo
echo "Installed packages:"
python - <<'PY'
import sys
print("python =", sys.version.split()[0])

try:
    import torch
    print("torch =", torch.__version__)
    print("cuda =", torch.version.cuda)
    print("cuda_available =", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("gpu =", torch.cuda.get_device_name(0))
        except Exception as e:
            print("gpu = <failed to query>", e)
except Exception as e:
    print("torch import failed:", e)

for name in ["mmengine", "mmcv", "mmdet"]:
    try:
        mod = __import__(name)
        print(f"{name} =", mod.__version__)
    except Exception as e:
        print(f"{name} import failed:", e)
PY

echo
echo "Done."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "This script only set up the environment. It did NOT modify your repo."
