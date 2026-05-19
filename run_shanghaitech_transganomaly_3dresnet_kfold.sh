#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "$ROOT"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage: ./run_shanghaitech_transganomaly_3dresnet_kfold.sh [--check]

Train ShanghaiTech with TransGANomaly-3DResNet.

Defaults:
  KFOLD=5
  EPOCHS=50
  FOLD=all
  EARLY_STOP_PATIENCE=0

Common overrides:
  FOLD=0 ./run_shanghaitech_transganomaly_3dresnet_kfold.sh
  RESUME=1 ./run_shanghaitech_transganomaly_3dresnet_kfold.sh
  RUN_TEST=1 ./run_shanghaitech_transganomaly_3dresnet_kfold.sh
  FOLD=0 TEST_ONLY=1 ./run_shanghaitech_transganomaly_3dresnet_kfold.sh
  PY=/path/to/python ./run_shanghaitech_transganomaly_3dresnet_kfold.sh

Checks only, does not train:
  ./run_shanghaitech_transganomaly_3dresnet_kfold.sh --check
EOF
    exit 0
    ;;
  --check)
    CHECK_ONLY=1
    shift
    ;;
  "")
    CHECK_ONLY=${CHECK_ONLY:-0}
    ;;
  *)
    echo "Unknown argument: $1" >&2
    echo "Use --help for usage." >&2
    exit 1
    ;;
esac

export CUDA_HOME=${CUDA_HOME:-/home/grouphahieu/cuda128}
HF2VAD_SITE=${HF2VAD_SITE:-$ROOT/hf2vad_bbox/lib/python3.10/site-packages}
DEFAULT_PY=$ROOT/hf2vad_bbox/bin/python
CONDA_LIBEXPAT=${CONDA_LIBEXPAT:-}
if [[ -z "${PY:-}" && -x /opt/miniforge3/pkgs/python-3.10.20-h3c07f61_0_cpython/bin/python3.10 ]]; then
  DEFAULT_PY=/opt/miniforge3/pkgs/python-3.10.20-h3c07f61_0_cpython/bin/python3.10
  if [[ -d /opt/miniforge3/pkgs/libexpat-2.7.4-hecca717_0/lib ]]; then
    CONDA_LIBEXPAT=/opt/miniforge3/pkgs/libexpat-2.7.4-hecca717_0/lib
  fi
fi

export TORCH_LIB=${TORCH_LIB:-$HF2VAD_SITE/torch/lib}
export LD_LIBRARY_PATH=${CONDA_LIBEXPAT:+$CONDA_LIBEXPAT:}$TORCH_LIB:/home/grouphahieu/cuda128/lib:/home/grouphahieu/cuda128/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$HF2VAD_SITE:${PYTHONPATH:-}

PY=${PY:-$DEFAULT_PY}

DATASET=${DATASET:-shanghaitech}
EPOCHS=${EPOCHS:-50}
KFOLD=${KFOLD:-5}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-0}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
MAX_CACHE_CHUNKS=${MAX_CACHE_CHUNKS:-2}
SPLIT_UNIT=${SPLIT_UNIT:-frame}
EVAL_EVERY=${EVAL_EVERY:-1}
STATS_EVERY=${STATS_EVERY:-1}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-0}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0}
FOLD=${FOLD:-all}          # all, or one 0-based fold index
RESUME=${RESUME:-0}        # set RESUME=1 to continue from latest model.pth-N per fold
RUN_TEST=${RUN_TEST:-0}    # set RUN_TEST=1 to evaluate folds after training
TEST_ONLY=${TEST_ONLY:-0}  # set TEST_ONLY=1 to evaluate existing checkpoints without training
EVAL_TEST_DURING_TRAIN=${EVAL_TEST_DURING_TRAIN:-0}
DETACH_RECON_MOTION=${DETACH_RECON_MOTION:-0}

RUN_NAME=${RUN_NAME:-TransGANomaly-3DResNet_${DATASET}_kfold${KFOLD}_e${EPOCHS}_seed${SEED}}
SAVE_DIR=${SAVE_DIR:-outputs/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-logs}

TRAIN_DIR="data/${DATASET}/training/chunked_samples"
TEST_DIR="data/${DATASET}/testing/chunked_samples"
GT_DIR="data/${DATASET}/ground_truth_demo"

if [[ ! -x "$PY" ]]; then
  echo "Missing Python executable: $PY" >&2
  echo "Set PY=/path/to/python if you want to use another environment." >&2
  exit 1
fi

"$PY" - <<'PY'
import sys

try:
    import torch
    import joblib
    import sklearn
    import scipy
    import matplotlib.pyplot
except Exception as exc:
    raise SystemExit(
        "Python environment is not ready. "
        f"executable={sys.executable!r}, error={exc!r}. "
        "Set PY=/path/to/python or HF2VAD_SITE=/path/to/site-packages."
    )

print(f"python={sys.executable}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "Missing training chunk dir: $TRAIN_DIR" >&2
  exit 1
fi

if [[ "$EVAL_TEST_DURING_TRAIN" == "1" || "$RUN_TEST" == "1" || "$TEST_ONLY" == "1" ]]; then
  if [[ ! -d "$TEST_DIR" ]]; then
    echo "Missing testing chunk dir: $TEST_DIR" >&2
    exit 1
  fi

  if [[ ! -f "$GT_DIR/gt_label_12fps.json" && ! -f "$GT_DIR/gt_label.json" ]]; then
    echo "Missing ShanghaiTech ground-truth file in: $GT_DIR" >&2
    exit 1
  fi
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "CHECK_ONLY=1, environment and dataset checks passed. Not starting training."
  exit 0
fi

mkdir -p "$LOG_DIR" "$SAVE_DIR"

COMMON_ARGS=(
  --dataset_name "$DATASET"
  --dataset_base_dir ./data
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --max_cache_chunks "$MAX_CACHE_CHUNKS"
  --seed "$SEED"
  --eval_every "$EVAL_EVERY"
  --stats_every "$STATS_EVERY"
  --save_dir "$SAVE_DIR"
  --split_unit "$SPLIT_UNIT"
  --kfold "$KFOLD"
  --early_stop_patience "$EARLY_STOP_PATIENCE"
  --early_stop_min_delta "$EARLY_STOP_MIN_DELTA"
)

if [[ "$PERSISTENT_WORKERS" == "1" ]]; then
  COMMON_ARGS+=(--persistent_workers)
fi

if [[ "$RESUME" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

if [[ "$EVAL_TEST_DURING_TRAIN" == "1" ]]; then
  COMMON_ARGS+=(--eval_test_during_train)
fi

if [[ "$DETACH_RECON_MOTION" == "1" ]]; then
  COMMON_ARGS+=(--detach_recon_motion)
fi

if [[ "$FOLD" != "all" ]]; then
  COMMON_ARGS+=(--fold "$FOLD")
fi

echo "==== Train TransGANomaly-3DResNet ShanghaiTech kfold=$KFOLD epochs=$EPOCHS fold=$FOLD ===="
echo "save_dir=$SAVE_DIR"

if [[ "$TEST_ONLY" == "1" ]]; then
  echo "==== Test only TransGANomaly-3DResNet ShanghaiTech kfold=$KFOLD fold=$FOLD ===="
  "$PY" TransGANomaly-3DResNet.py --mode test "${COMMON_ARGS[@]}" 2>&1 | tee "$LOG_DIR/${RUN_NAME}_test.log"
  exit 0
fi

"$PY" TransGANomaly-3DResNet.py --mode train "${COMMON_ARGS[@]}" 2>&1 | tee "$LOG_DIR/${RUN_NAME}_train.log"

if [[ "$RUN_TEST" == "1" ]]; then
  echo "==== Test TransGANomaly-3DResNet ShanghaiTech kfold=$KFOLD fold=$FOLD ===="
  "$PY" TransGANomaly-3DResNet.py --mode test "${COMMON_ARGS[@]}" 2>&1 | tee "$LOG_DIR/${RUN_NAME}_test.log"
fi
