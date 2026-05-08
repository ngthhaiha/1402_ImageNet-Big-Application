#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/grouphahieu/imagenet/hf2vad-master
cd "$ROOT"

export CUDA_HOME=/home/grouphahieu/cuda128
export TORCH_LIB=$ROOT/hf2vad_bbox/lib/python3.10/site-packages/torch/lib
export LD_LIBRARY_PATH=$TORCH_LIB:/home/grouphahieu/cuda128/lib:/home/grouphahieu/cuda128/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT
PY=$ROOT/hf2vad_bbox/bin/python

mkdir -p logs

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

run_bbox() {
  local mode=$1
  log "extract_bboxes.py --dataset_name avenue --mode ${mode}"
  "$PY" -c "import torch, runpy, sys; _orig=torch.load; torch.load=lambda *a, **k: _orig(*a, **({'weights_only': False} | k)); sys.argv=['pre_process/extract_bboxes.py','--proj_root','${ROOT}','--dataset_name','avenue','--mode','${mode}']; runpy.run_path('pre_process/extract_bboxes.py', run_name='__main__')"
}

run_flow() {
  local mode=$1
  log "extract_flows.py --dataset_name avenue --mode ${mode}"
  "$PY" pre_process/extract_flows.py --proj_root "$ROOT" --dataset_name avenue --mode "$mode"
}

run_samples() {
  local mode=$1
  log "extract_samples.py --dataset_name avenue --mode ${mode}"
  "$PY" pre_process/extract_samples.py --proj_root "$ROOT" --dataset_name avenue --mode "$mode"
}

if [[ ! -f data/avenue/avenue_bboxes_train.npy ]]; then
  run_bbox train
else
  log "skip train bboxes"
fi

if [[ ! -f data/avenue/avenue_bboxes_test.npy ]]; then
  run_bbox test
else
  log "skip test bboxes"
fi

if ! find data/avenue/training/flows -name '*.npy' -print -quit | grep -q .; then
  run_flow train
else
  log "skip train flows"
fi

if ! find data/avenue/testing/flows -name '*.npy' -print -quit | grep -q .; then
  run_flow test
else
  log "skip test flows"
fi

if [[ ! -f data/avenue/training/chunked_samples/chunked_samples_00.pkl ]]; then
  run_samples train
else
  log "skip train samples"
fi

if [[ ! -f data/avenue/testing/chunked_samples/chunked_samples_00.pkl ]]; then
  run_samples test
else
  log "skip test samples"
fi

if [[ ! -f ckpt/avenue_ML_MemAE_SC/best.pth ]]; then
  log "ml_memAE_sc_train.py"
  "$PY" ml_memAE_sc_train.py
else
  log "skip stage1"
fi

if [[ ! -f ckpt/avenue_ML_MemAE_SC_CVAE/best.pth ]]; then
  log "train.py"
  "$PY" train.py
else
  log "skip stage2"
fi

if [[ ! -f ckpt/avenue_ML_MemAE_SC_CVAE_finetune/best.pth ]]; then
  log "finetune.py"
  "$PY" finetune.py
else
  log "skip stage3"
fi

log "final eval for avenue finetune best"
"$PY" -c "import yaml; from eval import evaluate; cfg=yaml.safe_load(open('cfgs/finetune_cfg.yaml')); auc=evaluate(cfg, './ckpt/avenue_ML_MemAE_SC_CVAE_finetune/best.pth', './data/avenue/testing/chunked_samples/chunked_samples_00.pkl', './ckpt/avenue_ML_MemAE_SC_CVAE_finetune/training_stats.npy-80', suffix='final_best'); print({'final_auc': auc})"

log "pipeline finished"
