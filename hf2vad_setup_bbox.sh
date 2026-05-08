#!/usr/bin/env bash
set -euo pipefail

# This script ONLY creates a conda environment and installs packages.
# It does NOT modify your repo files.

ENV_NAME="${ENV_NAME:-hf2vad_setup_bbox}"
PY_VER="${PY_VER:-3.10}"

echo "[1/7] Checking conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH."
  exit 1
fi

# Make conda activate available in non-interactive shell
set +u
eval "$(conda shell.bash hook)"
set -u

echo "[2/7] Creating environment: ${ENV_NAME} (python=${PY_VER})"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} already exists. Reusing it."
else
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

echo "[3/7] Activating environment"
set +u
conda activate "${ENV_NAME}"
set -u

echo "[4/7] Configuring CUDA/build tools"
# mmcv builds CUDA/C++ extensions. Prefer the existing CUDA 12.8 toolkit
# instead of installing large CUDA packages into conda on every setup run.
CUDA_CANDIDATES=(
  "${CUDA_HOME:-}"
  "${CUDA_PATH:-}"
  "/home/grouphahieu/cuda128"
  "/usr/local/cuda-12.8"
  "/usr/local/cuda"
)

CUDA_ROOT=""
for candidate in "${CUDA_CANDIDATES[@]}"; do
  if [[ -n "${candidate}" && -x "${candidate}/bin/nvcc" ]]; then
    CUDA_ROOT="${candidate}"
    break
  fi
done

if [[ -z "${CUDA_ROOT}" ]]; then
  NVCC_BIN="$(command -v nvcc || true)"
  if [[ -n "${NVCC_BIN}" ]]; then
    CUDA_ROOT="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
  fi
fi

if [[ -z "${CUDA_ROOT}" || ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
  echo "Error: CUDA nvcc not found. Set CUDA_HOME to your CUDA 12.8 toolkit path."
  exit 1
fi

export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export PATH="${CUDA_HOME}/bin:${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

if [[ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" && -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++" ]]; then
  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
fi

echo "Using nvcc: $(command -v nvcc)"
nvcc --version
if ! nvcc --version | grep -q "release 12.8"; then
  echo "Error: CUDA 12.8 nvcc is required for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}."
  echo "       Found a different nvcc. Set CUDA_HOME=/home/grouphahieu/cuda128 or your CUDA 12.8 path."
  exit 1
fi

echo "[5/7] Upgrading packaging tools"
python -m pip install -U "pip<25.2" wheel
python -m pip install -U "setuptools<81" ninja

echo "[6/7] Cleaning conflicting OpenMMLab packages"
python -m pip uninstall -y torch torchvision torchaudio mmcv mmcv-lite mmengine mmdet opencv-python opencv-python-headless || true

echo "[7/7] Installing PyTorch + OpenMMLab packages"
# RTX 5060 Ti / Blackwell-oriented stack
python -m pip install \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  "numpy<2" "opencv-python<4.12" "opencv-python-headless<4.12" \
  pillow tqdm pyyaml matplotlib addict rich termcolor yapf \
  pycocotools scipy shapely terminaltables joblib scikit-learn "setuptools<81"

# Build mmcv from source against the installed torch/cu128 stack
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-1}"
echo "Using MAX_JOBS=${MAX_JOBS} for mmcv build"

python -m pip install --no-deps "mmengine==0.10.5"
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
