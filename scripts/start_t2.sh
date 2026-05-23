#!/bin/bash
set -euo pipefail

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate mas_chrcp

cd ~/mas_workspace
mkdir -p logs   # 确保日志目录存在,否则 tee 会失败

# === 关键修复:强制使用 V0 引擎 ===
# V1 引擎在 vLLM 0.10.2 + GPTQ Marlin + 该机器组合下子进程会早崩
export VLLM_USE_V1=0

CUDA_VISIBLE_DEVICES=0 vllm serve ~/mas_workspace/models/qwen-14b \
  --served-model-name qwen-14b \
  --quantization gptq_marlin \
  --dtype float16 \
  --gpu-memory-utilization 0.62 \
  --max-model-len 16384 \
  --enforce-eager \
  --enable-prefix-caching \
  --port 8000 \
  --host 0.0.0.0 \
  2>&1 | tee logs/t2_$(date +%Y%m%d_%H%M%S).log
