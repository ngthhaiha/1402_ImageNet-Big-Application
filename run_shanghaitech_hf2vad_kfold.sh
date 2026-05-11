#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/grouphahieu/imagenet/hf2vad-master}
cd "$ROOT"

export CUDA_HOME=${CUDA_HOME:-/home/grouphahieu/cuda128}
export TORCH_LIB=${TORCH_LIB:-$ROOT/hf2vad_bbox/lib/python3.10/site-packages/torch/lib}
export LD_LIBRARY_PATH=$TORCH_LIB:/home/grouphahieu/cuda128/lib:/home/grouphahieu/cuda128/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:${PYTHONPATH:-}

PY=${PY:-$ROOT/hf2vad_bbox/bin/python}

BASE_CFG=${BASE_CFG:-cfgs/cfg_shanghaitech.yaml}
STAGE1_BASE_CFG=${STAGE1_BASE_CFG:-cfgs/ml_memAE_sc_cfg_shanghaitech.yaml}
EPOCHS=${EPOCHS:-50}
STAGE1_EPOCHS=${STAGE1_EPOCHS:-$EPOCHS}
KFOLD=${KFOLD:-5}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}
BATCHSIZE=${BATCHSIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-8}
STAGES=${STAGES:-all}      # all, stage1, or stage2
FOLD=${FOLD:-all}          # all, or one 0-based fold index
FORCE=${FORCE:-0}          # set FORCE=1 to reuse an existing experiment dir
RESUME=${RESUME:-0}        # set RESUME=1 to continue from latest model.pth-N per fold
SPLIT_ONLY=${SPLIT_ONLY:-0}

DATASET_LOGIC=${DATASET_LOGIC:-shanghaitech}
DATASET_DIR=${DATASET_DIR:-shanghaitech}
DATASET_TAG=${DATASET_TAG:-$DATASET_DIR}
SPLIT_ROOT=${SPLIT_ROOT:-data/${DATASET_DIR}/kfold${KFOLD}_seed${SEED}}
STAGE1_RUN_NAME=${STAGE1_RUN_NAME:-${DATASET_TAG}_ML_MemAE_SC_kfold${KFOLD}_e${STAGE1_EPOCHS}_seed${SEED}}
RUN_NAME=${RUN_NAME:-${DATASET_TAG}_ML_MemAE_SC_CVAE_kfold${KFOLD}_e${EPOCHS}_seed${SEED}}
STAGE1_CFG_ROOT=${STAGE1_CFG_ROOT:-cfgs/generated/${STAGE1_RUN_NAME}}
CFG_ROOT=${CFG_ROOT:-cfgs/generated/${RUN_NAME}}
TRAIN_DIR=${TRAIN_DIR:-data/${DATASET_DIR}/training/chunked_samples}
TEST_CHUNK=${TEST_CHUNK:-data/${DATASET_DIR}/testing/chunked_samples}
ML_MEMAE_PRETRAINED=${ML_MEMAE_PRETRAINED:-./ckpt/shanghaitech_ML_MemAE_SC/best.pth}

if [[ ! -f "$BASE_CFG" ]]; then
  echo "Missing BASE_CFG: $BASE_CFG" >&2
  exit 1
fi

if [[ ! -f "$STAGE1_BASE_CFG" ]]; then
  echo "Missing STAGE1_BASE_CFG: $STAGE1_BASE_CFG" >&2
  exit 1
fi

if [[ "$STAGES" != "all" && "$STAGES" != "stage1" && "$STAGES" != "stage2" ]]; then
  echo "Invalid STAGES=$STAGES. Expected all, stage1, or stage2." >&2
  exit 1
fi

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "Missing training chunk dir: $TRAIN_DIR" >&2
  exit 1
fi

if [[ ! -d "$TEST_CHUNK" ]]; then
  echo "Missing testing chunk dir: $TEST_CHUNK" >&2
  exit 1
fi

if [[ "$STAGES" == "stage2" && ! -f "$ML_MEMAE_PRETRAINED" ]]; then
  echo "Missing ML-MemAE-SC pretrained checkpoint: $ML_MEMAE_PRETRAINED" >&2
  exit 1
fi

mkdir -p logs "$STAGE1_CFG_ROOT" "$CFG_ROOT"

"$PY" - <<PY
from pathlib import Path
import json
import numpy as np
import joblib
import yaml
from sklearn.model_selection import KFold

base_cfg_path = Path("$BASE_CFG")
stage1_base_cfg_path = Path("$STAGE1_BASE_CFG")
train_dir = Path("$TRAIN_DIR")
split_root = Path("$SPLIT_ROOT")
stage1_cfg_root = Path("$STAGE1_CFG_ROOT")
cfg_root = Path("$CFG_ROOT")
stage1_run_name = "$STAGE1_RUN_NAME"
run_name = "$RUN_NAME"
dataset_dir = "$DATASET_DIR"
dataset_logic = "$DATASET_LOGIC"
kfold = int("$KFOLD")
seed = int("$SEED")
epochs = int("$EPOCHS")
stage1_epochs = int("$STAGE1_EPOCHS")
device = "$DEVICE"
batchsize = int("$BATCHSIZE")
num_workers = int("$NUM_WORKERS")
pretrained = "$ML_MEMAE_PRETRAINED"

chunk_files = sorted(train_dir.glob("chunked_samples_*.pkl"))
if not chunk_files:
    raise FileNotFoundError(f"No chunked_samples_*.pkl found in {train_dir}")

chunk_meta = []
total_samples = 0
for path in chunk_files:
    data = joblib.load(path, mmap_mode="r")
    n = len(data["sample_id"])
    chunk_meta.append({
        "path": str(path),
        "samples": int(n),
        "global_start": int(total_samples),
        "global_end": int(total_samples + n),
    })
    total_samples += n

splitter = KFold(n_splits=kfold, shuffle=True, random_state=seed)
fold_meta = []

split_root.mkdir(parents=True, exist_ok=True)
stage1_cfg_root.mkdir(parents=True, exist_ok=True)
cfg_root.mkdir(parents=True, exist_ok=True)

stage1_base_cfg = yaml.safe_load(stage1_base_cfg_path.read_text())
base_cfg = yaml.safe_load(base_cfg_path.read_text())

for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.arange(total_samples))):
    fold_dir = split_root / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_index_path = fold_dir / "train_indices.npy"
    val_index_path = fold_dir / "val_indices.npy"
    np.save(train_index_path, train_idx.astype(np.int64))
    np.save(val_index_path, val_idx.astype(np.int64))

    stage1_cfg = dict(stage1_base_cfg)
    stage1_cfg["dataset_name"] = dataset_logic
    stage1_cfg["dataset_dir_name"] = dataset_dir
    stage1_cfg["dataset_base_dir"] = "./data"
    stage1_cfg["exp_name"] = f"{stage1_run_name}/fold_{fold_idx}"
    stage1_cfg["num_epochs"] = stage1_epochs
    stage1_cfg["device"] = device
    stage1_cfg["batchsize"] = batchsize
    stage1_cfg["num_workers"] = num_workers
    stage1_cfg["pretrained"] = False

    stage1_cfg_path = stage1_cfg_root / f"fold_{fold_idx}.yaml"
    stage1_cfg_path.write_text(yaml.safe_dump(stage1_cfg, sort_keys=False))

    cfg = dict(base_cfg)
    cfg["dataset_name"] = dataset_logic
    cfg["dataset_dir_name"] = dataset_dir
    cfg["dataset_base_dir"] = "./data"
    cfg["exp_name"] = f"{run_name}/fold_{fold_idx}"
    cfg["num_epochs"] = epochs
    cfg["device"] = device
    cfg["batchsize"] = batchsize
    cfg["num_workers"] = num_workers
    cfg["pretrained"] = False
    cfg["ML_MemAE_SC_pretrained"] = f"./ckpt/{stage1_run_name}/fold_{fold_idx}/best.pth"

    cfg_path = cfg_root / f"fold_{fold_idx}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    fold_meta.append({
        "fold": fold_idx,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_indices": str(train_index_path),
        "val_indices": str(val_index_path),
        "stage1_cfg": str(stage1_cfg_path),
        "stage1_checkpoint": f"ckpt/{stage1_run_name}/fold_{fold_idx}/best.pth",
        "cfg": str(cfg_path),
    })

meta = {
    "dataset": dataset_dir,
    "dataset_logic": dataset_logic,
    "training_chunks": chunk_meta,
    "total_samples": int(total_samples),
    "kfold": kfold,
    "seed": seed,
    "stage1_epochs": stage1_epochs,
    "epochs": epochs,
    "stage1_run_name": stage1_run_name,
    "run_name": run_name,
    "stage2_fallback_pretrained": pretrained,
    "folds": fold_meta,
}
(split_root / "kfold_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
PY

if [[ "$SPLIT_ONLY" == "1" ]]; then
  echo "SPLIT_ONLY=1, not starting training."
  exit 0
fi

if [[ "$FOLD" == "all" ]]; then
  FOLDS=$(seq 0 $((KFOLD - 1)))
else
  FOLDS="$FOLD"
fi

latest_checkpoint_info() {
  local exp_dir="$1"
  "$PY" - "$exp_dir" <<'PY'
from pathlib import Path
import re
import sys

exp_dir = Path(sys.argv[1])
best = None
for path in exp_dir.glob("model.pth-*"):
    match = re.search(r"-(\d+)$", path.name)
    if match is None:
        continue
    epoch = int(match.group(1))
    if best is None or epoch > best[0]:
        best = (epoch, path)

if best is not None:
    print(f"{best[1]} {best[0]}")
PY
}

for fold in $FOLDS; do
  if [[ "$fold" -lt 0 || "$fold" -ge "$KFOLD" ]]; then
    echo "Invalid FOLD=$fold. Expected 0..$((KFOLD - 1)) or all." >&2
    exit 1
  fi

  STAGE1_CFG_FILE="$STAGE1_CFG_ROOT/fold_${fold}.yaml"
  CFG_FILE="$CFG_ROOT/fold_${fold}.yaml"
  TRAIN_INDEX="$SPLIT_ROOT/fold_${fold}/train_indices.npy"
  STAGE1_EXP_DIR="ckpt/${STAGE1_RUN_NAME}/fold_${fold}"
  EXP_DIR="ckpt/${RUN_NAME}/fold_${fold}"

  if [[ "$STAGES" == "all" || "$STAGES" == "stage1" ]]; then
    STAGE1_RESUME_CKPT=""
    STAGE1_RESUME_EPOCH=0
    RUN_STAGE1_EPOCHS="$STAGE1_EPOCHS"

    if [[ "$RESUME" == "1" && -d "$STAGE1_EXP_DIR" ]]; then
      STAGE1_RESUME_INFO=$(latest_checkpoint_info "$STAGE1_EXP_DIR" || true)
      if [[ -n "$STAGE1_RESUME_INFO" ]]; then
        read -r STAGE1_RESUME_CKPT STAGE1_RESUME_EPOCH <<< "$STAGE1_RESUME_INFO"
        if [[ "$STAGE1_RESUME_EPOCH" -ge "$STAGE1_EPOCHS" ]]; then
          echo "==== Stage 1 ML-MemAE-SC fold=$fold already reached epoch $STAGE1_RESUME_EPOCH/$STAGE1_EPOCHS; skipping ===="
        else
          RUN_STAGE1_EPOCHS=$((STAGE1_EPOCHS - STAGE1_RESUME_EPOCH))
          echo "==== RESUME Stage 1 fold=$fold from $STAGE1_RESUME_CKPT epoch=$STAGE1_RESUME_EPOCH remaining_epochs=$RUN_STAGE1_EPOCHS ===="
        fi
      else
        echo "==== RESUME Stage 1 fold=$fold requested, but no model.pth-N found; starting fresh in $STAGE1_EXP_DIR ===="
      fi
    elif [[ -d "$STAGE1_EXP_DIR" && "$FORCE" != "1" ]]; then
      echo "Refusing to reuse existing Stage 1 experiment dir: $STAGE1_EXP_DIR" >&2
      echo "Set RESUME=1 to continue, FORCE=1 to reuse anyway, or change STAGE1_RUN_NAME." >&2
      exit 1
    fi

    if [[ "$STAGE1_RESUME_EPOCH" -lt "$STAGE1_EPOCHS" ]]; then
      echo "==== Stage 1 ML-MemAE-SC ShanghaiTech fold=$fold/$((KFOLD - 1)) epochs=$RUN_STAGE1_EPOCHS target_total=$STAGE1_EPOCHS run=$STAGE1_RUN_NAME ===="
      "$PY" - <<PY 2>&1 | tee -a "logs/${STAGE1_RUN_NAME}_fold${fold}.log"
import torch
import yaml

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **({"weights_only": False} | k))

import ml_memAE_sc_train_shanghaitech as s1

cfg_file = "$STAGE1_CFG_FILE"
cfg = yaml.safe_load(open(cfg_file))
cfg["cfg_file"] = cfg_file
cfg["num_epochs"] = int("$RUN_STAGE1_EPOCHS")
cfg["pretrained"] = "$STAGE1_RESUME_CKPT" if "$STAGE1_RESUME_CKPT" else False
train_indices = torch.from_numpy(__import__("numpy").load("$TRAIN_INDEX")).tolist()
s1.train(
    cfg,
    "$TRAIN_DIR",
    "$TEST_CHUNK",
    training_sample_indices=train_indices,
)
PY
    fi
  fi

  if [[ "$STAGES" == "stage1" ]]; then
    continue
  fi

  if [[ "$STAGES" == "all" ]]; then
    ML_MEMAE_FOR_FOLD="ckpt/${STAGE1_RUN_NAME}/fold_${fold}/best.pth"
    if [[ ! -f "$ML_MEMAE_FOR_FOLD" ]]; then
      echo "Missing Stage 1 best checkpoint for fold $fold: $ML_MEMAE_FOR_FOLD" >&2
      exit 1
    fi
  else
    ML_MEMAE_FOR_FOLD="$ML_MEMAE_PRETRAINED"
  fi

  RESUME_CKPT=""
  RESUME_EPOCH=0
  RUN_EPOCHS="$EPOCHS"

  if [[ "$RESUME" == "1" && -d "$EXP_DIR" ]]; then
    RESUME_INFO=$(latest_checkpoint_info "$EXP_DIR" || true)
    if [[ -n "$RESUME_INFO" ]]; then
      read -r RESUME_CKPT RESUME_EPOCH <<< "$RESUME_INFO"
      if [[ "$RESUME_EPOCH" -ge "$EPOCHS" ]]; then
        echo "==== HF2VAD ShanghaiTech fold=$fold already reached epoch $RESUME_EPOCH/$EPOCHS; skipping ===="
        continue
      fi
      RUN_EPOCHS=$((EPOCHS - RESUME_EPOCH))
      echo "==== RESUME fold=$fold from $RESUME_CKPT epoch=$RESUME_EPOCH remaining_epochs=$RUN_EPOCHS ===="
    else
      echo "==== RESUME fold=$fold requested, but no model.pth-N found; starting fresh in $EXP_DIR ===="
    fi
  elif [[ -d "$EXP_DIR" && "$FORCE" != "1" ]]; then
    echo "Refusing to reuse existing experiment dir: $EXP_DIR" >&2
    echo "Set RESUME=1 to continue, FORCE=1 to reuse anyway, or change RUN_NAME." >&2
    exit 1
  fi

  echo "==== Stage 2 CVAE/HF2VAD ShanghaiTech fold=$fold/$((KFOLD - 1)) epochs=$RUN_EPOCHS target_total=$EPOCHS run=$RUN_NAME ===="
  "$PY" - <<PY 2>&1 | tee -a "logs/${RUN_NAME}_fold${fold}.log"
import torch
import yaml

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **({"weights_only": False} | k))

import train as t

cfg_file = "$CFG_FILE"
cfg = yaml.safe_load(open(cfg_file))
cfg["cfg_file"] = cfg_file
cfg["num_epochs"] = int("$RUN_EPOCHS")
cfg["pretrained"] = "$RESUME_CKPT" if "$RESUME_CKPT" else False
cfg["ML_MemAE_SC_pretrained"] = "$ML_MEMAE_FOR_FOLD"
train_indices = torch.from_numpy(__import__("numpy").load("$TRAIN_INDEX")).tolist()
t.train(
    cfg,
    "$TRAIN_DIR",
    "$TEST_CHUNK",
    training_sample_indices=train_indices,
)
PY
done
