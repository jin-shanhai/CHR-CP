#!/bin/bash
# Phase 2 主实验: 5 benchmarks x CHR-CP E2
# E2 = K=5, tau_low=0.10, tau_high=0.50, alpha=0.7 (run_main.py defaults)

set -e
cd /home/aiagent/mas_workspace/code

RESULTS_DIR="results/phase2"
mkdir -p "$RESULTS_DIR"

declare -A BENCHMARKS=(
    ["math"]=300
    ["aime"]=60
    ["humaneval"]=150
    ["mmlu_pro"]=200
    ["gpqa"]=100
)

SEED=42

for benchmark in "${!BENCHMARKS[@]}"; do
    n_samples=${BENCHMARKS[$benchmark]}
    suffix="${benchmark}_chrcp_seed${SEED}"
    output_file="${RESULTS_DIR}/chrcp/${benchmark}_${suffix}.jsonl"
    mkdir -p "$(dirname "$output_file")"

    # Resume
    if [ -f "$output_file" ]; then
        existing=$(wc -l < "$output_file")
        if [ "$existing" -ge "$n_samples" ]; then
            echo "skip: $benchmark ($existing/$n_samples)"
            continue
        else
            echo "resume: $benchmark ($existing/$n_samples)"
        fi
    fi

    echo "=== $benchmark ($n_samples samples) ==="

    python -m experiments.run_main \
        --benchmark "$benchmark" \
        --n_samples "$n_samples" \
        --seed "$SEED" \
        --concurrency 4 \
        --max_cost_usd 30.0 \
        --results_dir "$RESULTS_DIR" \
        --output_suffix "_${suffix}" \
        --resume \
        --method chrcp

    echo "done: $benchmark"
done

echo "=== Phase 2 complete ==="
