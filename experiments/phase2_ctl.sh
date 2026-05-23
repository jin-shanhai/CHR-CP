#!/bin/bash
# Stop or start all Phase 2 benchmark tests
# Usage: bash experiments/phase2_ctl.sh stop
#        bash experiments/phase2_ctl.sh start

MODE="${1:-start}"
cd /home/aiagent/mas_workspace/code

BENCHMARKS=(
    "humaneval:150"
    "aime:60"
    "mmlu_pro:200"
    "gpqa:100"
)

SEED=42

if [ "$MODE" == "stop" ]; then
    pids=$(ps aux | grep "run_main" | grep -v grep | awk '{print $2}')
    if [ -z "$pids" ]; then
        echo "No running tests"
    else
        echo "Killing: $pids"
        kill $pids
        echo "Stopped"
    fi
    exit 0
fi

if [ "$MODE" == "start" ]; then
    for entry in "${BENCHMARKS[@]}"; do
        bm="${entry%%:*}"
        n="${entry##*:}"
        log="phase2_${bm}.log"

        echo "Starting $bm ($n samples) ..."
        nohup python -m experiments.run_main \
            --benchmark "$bm" \
            --n_samples "$n" \
            --seed "$SEED" \
            --concurrency 4 \
            --max_cost_usd 30.0 \
            --results_dir results/phase2 \
            --output_suffix "_${bm}_chrcp_seed${SEED}" \
            --resume \
            --method chrcp \
            > "$log" 2>&1 &
    done
    echo "All 4 benchmarks launched"
    exit 0
fi

echo "Usage: bash experiments/phase2_ctl.sh {start|stop}"
