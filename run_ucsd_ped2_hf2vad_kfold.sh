#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/grouphahieu/imagenet/hf2vad-master}
cd "$ROOT"

export DATASET=${DATASET:-UCSDped2}
export DATASET_DIR=${DATASET_DIR:-UCSDped2}
export DATASET_LOGIC=${DATASET_LOGIC:-ped2}
export KFOLD=${KFOLD:-5}

exec ./run_ped2_hf2vad_kfold.sh "$@"
