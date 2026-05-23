#!/bin/bash
set -euo pipefail

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate mas_chrcp

cd ~/mas_workspace
mkdir -p logs

# 强制使用 V0 引擎(同 T2,V1 在该环境下子进程会崩)
export VLLM_USE_V1=0

CUDA_VISIBLE_DEVICES=0 vllm serve ~/mas_workspace/models/qwen-7b \
  --served-model-name qwen-7b \
  --quantization gptq_marlin \
  --dtype float16 \
  --gpu-memory-utilization 0.33 \
  --max-model-len 16384 \
  --enforce-eager \
  --enable-prefix-caching \
  --port 8001 \
  --host 0.0.0.0 \
  2>&1 | tee logs/t1_$(date +%Y%m%d_%H%M%S).log
