"""Phase 2 monitor — CHR-CP + baselines across 5 benchmarks.

Usage:
    python -m tests.analyze_phase2
"""

from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

console = Console(width=160)

RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"

BENCHMARK_TARGETS = {
    "math": 300, "aime": 60, "humaneval": 150,
    "mmlu": 200, "gpqa": 100,
}

METHODS = ["chrcp", "single_t1", "single_t4", "static_3agent"]


def discover_run(benchmark: str, method: str) -> Path | None:
    candidates = sorted((RESULTS_DIR / method).glob(f"{benchmark}_*.jsonl"))
    return candidates[-1] if candidates else None


def load_records(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_metrics(records: list) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}

    n_correct = sum(1 for r in records if r.get("correct"))
    total_cost = sum(r.get("cost_usd", 0) for r in records)
    total_calls = sum(r.get("n_calls", 0) for r in records)

    tiers = Counter(r.get("final_tier", "?") for r in records)
    n_t3 = tiers.get("T3", 0) + tiers.get("T4", 0)
    n_t4 = tiers.get("T4", 0)

    n_escalated = sum(1 for r in records if r.get("num_escalate", 0) > 0)

    cache_factors = []
    for r in records:
        for d in r.get("decisions", []):
            cf = d.get("ca2r_cache_factor")
            if cf is not None and cf < 1.0:
                cache_factors.append(cf)

    t2_hits = t2_prompt = t3_hits = t3_prompt = 0
    for r in records:
        for c in r["api_calls"]:
            if c["tier"] == "T2":
                t2_hits += c.get("cache_hit_tokens") or 0
                t2_prompt += c.get("prompt_tokens", 0)
            elif c["tier"] == "T3":
                t3_hits += c.get("cache_hit_tokens") or 0
                t3_prompt += c.get("prompt_tokens", 0)

    return {
        "n": n, "n_correct": n_correct,
        "accuracy": n_correct / n,
        "total_cost": total_cost,
        "cost_per_sample": total_cost / n,
        "avg_calls": total_calls / n,
        "tiers": dict(tiers),
        "t3_rate": n_t3 / n,
        "t4_rate": n_t4 / n,
        "n_escalated": n_escalated,
        "avg_cache": sum(cache_factors) / len(cache_factors) if cache_factors else 1.0,
        "t2_hit": t2_hits / max(1, t2_prompt),
        "t3_hit": t3_hits / max(1, t3_prompt),
    }


def main():
    console.rule("[bold cyan]Phase 2 — CHR-CP vs Baselines[/bold cyan]")

    grand = {}

    for bm in BENCHMARK_TARGETS:
        target = BENCHMARK_TARGETS[bm]
        console.print(f"\n[bold]{bm}[/bold] (target={target})")
        table = Table(title=f"{bm} — Method Comparison")
        table.add_column("Method", style="cyan", width=14)
        table.add_column("N", justify="right", width=5)
        table.add_column("Acc", justify="right", style="green", width=7)
        table.add_column("Cost/s", justify="right", width=10)
        table.add_column("Tiers", justify="center", width=20)
        table.add_column("T3%", justify="right", width=6)
        table.add_column("T4%", justify="right", width=6)
        table.add_column("T2hit", justify="right", width=7)

        bm_data = False
        for method in METHODS:
            path = discover_run(bm, method)
            if not path:
                table.add_row(method, "—", "—", "—", "—", "—", "—", "—")
                continue

            records = load_records(path)
            m = compute_metrics(records)
            if m["n"] == 0:
                table.add_row(method, "0", "—", "—", "—", "—", "—", "—")
                continue

            bm_data = True
            tier_str = " ".join(
                f"{t}:{c/m['n']*100:.0f}%" for t, c in sorted(m["tiers"].items()) if c > 0
            ) or "—"

            table.add_row(
                method,
                str(m["n"]),
                f"{m['accuracy']*100:.1f}%",
                f"\${m['cost_per_sample']:.5f}",
                tier_str,
                f"{m['t3_rate']*100:.0f}%" if m["t3_rate"] > 0 else "—",
                f"{m['t4_rate']*100:.0f}%" if m["t4_rate"] > 0 else "—",
                f"{m['t2_hit']*100:.0f}%" if m["t2_hit"] > 0 else "—",
            )

            # Accumulate
            grand.setdefault(bm, {})[method] = {
                "n": m["n"], "correct": m["n_correct"], "cost": m["total_cost"],
                "acc": m["accuracy"], "cps": m["cost_per_sample"],
            }

        if bm_data:
            console.print(table)

    # Summary
    console.print(f"\n[grey]python -m tests.analyze_phase2[/grey]\n")


if __name__ == "__main__":
    main()
