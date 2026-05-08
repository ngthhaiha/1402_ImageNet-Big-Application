#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/grouphahieu/imagenet/hf2vad-master}
cd "$ROOT"

export CUDA_HOME=${CUDA_HOME:-/home/grouphahieu/cuda128}
export TORCH_LIB=${TORCH_LIB:-$ROOT/hf2vad_bbox/lib/python3.10/site-packages/torch/lib}
export LD_LIBRARY_PATH=$TORCH_LIB:/home/grouphahieu/cuda128/lib:/home/grouphahieu/cuda128/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:${PYTHONPATH:-}

PY=${PY:-$ROOT/hf2vad_bbox/bin/python}

# Defaults are chosen for fair comparison with a 50-epoch, 5-fold experiment.
BASE_CFG=${BASE_CFG:-cfgs/ped2_resume_cfg.yaml}
EPOCHS=${EPOCHS:-50}
KFOLD=${KFOLD:-5}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}
BATCHSIZE=${BATCHSIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-0}
FOLD=${FOLD:-all}          # all, or one 0-based fold index
FORCE=${FORCE:-0}          # set FORCE=1 to reuse/overwrite an existing experiment dir
SPLIT_ONLY=${SPLIT_ONLY:-0}

DATASET=ped2
SPLIT_ROOT=${SPLIT_ROOT:-data/${DATASET}/kfold${KFOLD}_seed${SEED}}
RUN_NAME=${RUN_NAME:-ped2_ML_MemAE_SC_CVAE_kfold${KFOLD}_e${EPOCHS}_seed${SEED}}
CFG_ROOT=${CFG_ROOT:-cfgs/generated/${RUN_NAME}}
TRAIN_SRC=${TRAIN_SRC:-data/${DATASET}/training/chunked_samples/chunked_samples_00.pkl}
TRAIN_DIR=${TRAIN_DIR:-data/${DATASET}/training/chunked_samples}
TEST_CHUNK=${TEST_CHUNK:-data/${DATASET}/testing/chunked_samples/chunked_samples_00.pkl}

if [[ ! -f "$BASE_CFG" ]]; then
  echo "Missing BASE_CFG: $BASE_CFG" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SRC" ]]; then
  echo "Missing training chunk: $TRAIN_SRC" >&2
  exit 1
fi

if [[ ! -f "$TEST_CHUNK" ]]; then
  echo "Missing testing chunk: $TEST_CHUNK" >&2
  exit 1
fi

mkdir -p logs "$CFG_ROOT"

"$PY" - <<PY
from pathlib import Path
import json
import numpy as np
import joblib
import yaml
from sklearn.model_selection import KFold

base_cfg_path = Path("$BASE_CFG")
train_src = Path("$TRAIN_SRC")
split_root = Path("$SPLIT_ROOT")
cfg_root = Path("$CFG_ROOT")
run_name = "$RUN_NAME"
kfold = int("$KFOLD")
seed = int("$SEED")
epochs = int("$EPOCHS")
device = "$DEVICE"
batchsize = int("$BATCHSIZE")
num_workers = int("$NUM_WORKERS")

data = joblib.load(train_src, mmap_mode="r")
n = len(data["sample_id"])
splitter = KFold(n_splits=kfold, shuffle=True, random_state=seed)
fold_meta = []

split_root.mkdir(parents=True, exist_ok=True)
cfg_root.mkdir(parents=True, exist_ok=True)

base_cfg = yaml.safe_load(base_cfg_path.read_text())

for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.arange(n))):
    fold_dir = split_root / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_index_path = fold_dir / "train_indices.npy"
    val_index_path = fold_dir / "val_indices.npy"
    np.save(train_index_path, train_idx.astype(np.int64))
    np.save(val_index_path, val_idx.astype(np.int64))

    cfg = dict(base_cfg)
    cfg["dataset_name"] = "ped2"
    cfg["dataset_base_dir"] = "./data"
    cfg["exp_name"] = f"{run_name}/fold_{fold_idx}"
    cfg["num_epochs"] = epochs
    cfg["device"] = device
    cfg["batchsize"] = batchsize
    cfg["num_workers"] = num_workers
    cfg["pretrained"] = False
    cfg["ML_MemAE_SC_pretrained"] = "./ckpt/ped2_ML_MemAE_SC/best.pth"

    cfg_path = cfg_root / f"fold_{fold_idx}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    fold_meta.append({
        "fold": fold_idx,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_indices": str(train_index_path),
        "val_indices": str(val_index_path),
        "cfg": str(cfg_path),
    })

meta = {
    "dataset": "ped2",
    "source_chunk": str(train_src),
    "kfold": kfold,
    "seed": seed,
    "epochs": epochs,
    "run_name": run_name,
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

for fold in $FOLDS; do
  if [[ "$fold" -lt 0 || "$fold" -ge "$KFOLD" ]]; then
    echo "Invalid FOLD=$fold. Expected 0..$((KFOLD - 1)) or all." >&2
    exit 1
  fi

  CFG_FILE="$CFG_ROOT/fold_${fold}.yaml"
  TRAIN_INDEX="$SPLIT_ROOT/fold_${fold}/train_indices.npy"
  EXP_DIR="ckpt/${RUN_NAME}/fold_${fold}"

  if [[ -d "$EXP_DIR" && "$FORCE" != "1" ]]; then
    echo "Refusing to reuse existing experiment dir: $EXP_DIR" >&2
    echo "Set FORCE=1 to continue anyway, or change RUN_NAME." >&2
    exit 1
  fi

  echo "==== HF2VAD Ped2 fold=$fold/$((KFOLD - 1)) epochs=$EPOCHS run=$RUN_NAME ===="
  "$PY" - <<PY 2>&1 | tee "logs/${RUN_NAME}_fold${fold}.log"
import torch
import yaml

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **({"weights_only": False} | k))

import train as t

cfg_file = "$CFG_FILE"
cfg = yaml.safe_load(open(cfg_file))
cfg["cfg_file"] = cfg_file
train_indices = torch.from_numpy(__import__("numpy").load("$TRAIN_INDEX")).tolist()
t.train(
    cfg,
    "$TRAIN_DIR",
    "$TEST_CHUNK",
    training_sample_indices=train_indices,
)
PY
done
