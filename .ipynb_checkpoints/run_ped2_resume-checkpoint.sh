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

exec "$PY" -c "import torch, yaml; _orig=torch.load; torch.load=lambda *a, **k: _orig(*a, **({'weights_only': False} | k)); import train as t; cfg=yaml.safe_load(open('log/ped2_ML_MemAE_SC_CVAE/cfg.yaml')); cfg['pretrained']='./ckpt/ped2_ML_MemAE_SC_CVAE/model.pth-46'; cfg['num_epochs']=34; cfg['num_workers']=0; t.train(cfg, 'data/ped2/training/chunked_samples', 'data/ped2/testing/chunked_samples/chunked_samples_00.pkl')"
