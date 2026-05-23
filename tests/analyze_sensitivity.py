"""Analyze Phase 1b results and recommend the best configuration."""

from __future__ import annotations
import sys
import json
import statistics
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class ConfigResult:
    name: str
    k: int
    tau_low: float
    tau_high: float
    alpha: float
    n_samples: int = 0
    accuracy: float = 0.0
    total_cost: float = 0.0
    cost_per_sample: float = 0.0
    avg_latency: float = 0.0
    n_stay: int = 0
    n_branch: int = 0
    n_escalate: int = 0
    branch_rate: float = 0.0
    escalate_rate: float = 0.0
    final_tier_dist: dict = field(default_factory=dict)
    escalate_save_rate: float = 0.0
    cache_factor_first_quartile: float = 1.0
    cache_factor_last_quartile: float = 1.0
    composite_score: float = 0.0


GRID_META = {
    "A1": (3, 0.10, 0.45, 0.8),  "A2": (3, 0.10, 0.50, 0.8),  "A3": (3, 0.10, 0.55, 0.8),
    "B1": (3, 0.10, 0.45, 0.7),  "B2": (3, 0.10, 0.50, 0.7),  "C1": (3, 0.10, 0.45, 0.9),
    "D1": (5, 0.10, 0.40, 0.8),  "D2": (5, 0.10, 0.45, 0.8),  "D3": (5, 0.10, 0.50, 0.8),
    "D4": (5, 0.10, 0.55, 0.8),  "E1": (5, 0.10, 0.45, 0.7),  "E2": (5, 0.10, 0.50, 0.7),
    "F1": (5, 0.10, 0.45, 0.9),  "F2": (5, 0.10, 0.50, 0.9),  "F3": (5, 0.10, 0.55, 0.9),
}


def load_p1b_result(name: str, k: int, tau_low: float, tau_high: float, alpha: float) -> Optional[ConfigResult]:
    path = PROJECT_ROOT / "results" / "chrcp" / f"math_grid_{name}_p1b.jsonl"
    if not path.exists():
        return None
    
    with open(path) as f:
        records = [json.loads(line) for line in f]
    
    if not records:
        return None
    
    n = len(records)
    r = ConfigResult(name=name, k=k, tau_low=tau_low, tau_high=tau_high, alpha=alpha,
                    n_samples=n)
    
    r.accuracy = sum(1 for x in records if x.get("correct")) / n
    r.total_cost = sum(x.get("cost_usd", 0) for x in records)
    r.cost_per_sample = r.total_cost / n
    r.avg_latency = sum(x.get("latency_seconds", 0) for x in records) / n
    
    r.n_stay = sum(x.get("num_stay", 0) for x in records)
    r.n_branch = sum(x.get("num_branch", 0) for x in records)
    r.n_escalate = sum(x.get("num_escalate", 0) for x in records)
    total = max(1, r.n_stay + r.n_branch + r.n_escalate)
    r.branch_rate = r.n_branch / total
    r.escalate_rate = r.n_escalate / total
    
    r.final_tier_dist = dict(Counter(x.get("final_tier", "?") for x in records))
    
    esc_samples = [x for x in records if x.get("num_escalate", 0) > 0]
    if esc_samples:
        r.escalate_save_rate = sum(1 for x in esc_samples if x.get("correct")) / len(esc_samples)
    
    cfs = [x.get("avg_cache_factor", 1.0) for x in records if "avg_cache_factor" in x]
    if len(cfs) >= 8:
        q = len(cfs) // 4
        r.cache_factor_first_quartile = statistics.mean(cfs[:q])
        r.cache_factor_last_quartile = statistics.mean(cfs[-q:])
    
    # Composite score
    activeness = min(1.0, (r.branch_rate + r.escalate_rate) / 0.25)
    cost_eff = max(0, 1.0 - r.cost_per_sample / 0.05)
    r.composite_score = (
        0.50 * r.accuracy
        + 0.20 * activeness
        + 0.20 * r.escalate_save_rate
        + 0.10 * cost_eff
    )
    
    return r


def main():
    console.rule("[bold cyan]CHR-CP Sensitivity Analysis: Phase 1b[/bold cyan]")
    
    results: List[ConfigResult] = []
    for name, (k, tau_low, tau_high, alpha) in GRID_META.items():
        r = load_p1b_result(name, k, tau_low, tau_high, alpha)
        if r:
            results.append(r)
    
    if not results:
        console.print("[red]No Phase 1b data found. Run sensitivity_grid_v2.py first.[/red]")
        sys.exit(1)
    
    console.print(f"\n[bold]Loaded {len(results)} configurations from Phase 1b[/bold]\n")
    
    # Full results table
    table = Table(title="Phase 1b Detailed Results (sorted by composite score)")
    table.add_column("Cfg", style="bold")
    table.add_column("K", justify="center")
    table.add_column("τ_low", justify="right")
    table.add_column("τ_high", justify="right")
    table.add_column("α", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Acc", justify="right", style="green")
    table.add_column("$/sample", justify="right")
    table.add_column("S/B/E %", justify="center")
    table.add_column("Tiers", justify="center")
    table.add_column("Save%", justify="right", style="cyan")
    table.add_column("Cache↓", justify="right")
    table.add_column("Score", justify="right", style="bold yellow")
    
    for r in sorted(results, key=lambda x: -x.composite_score):
        stay_pct = r.n_stay / max(1, r.n_stay + r.n_branch + r.n_escalate) * 100
        action_str = f"{stay_pct:.0f}/{r.branch_rate*100:.0f}/{r.escalate_rate*100:.0f}"
        tier_str = ",".join(
            f"{t}:{r.final_tier_dist.get(t, 0)}" for t in ["T1","T2","T3","T4"]
            if r.final_tier_dist.get(t, 0) > 0
        )
        cache_drop = r.cache_factor_first_quartile - r.cache_factor_last_quartile
        cache_str = f"-{cache_drop*100:.1f}%" if cache_drop > 0 else "—"
        save_str = f"{r.escalate_save_rate*100:.0f}%" if r.escalate_rate > 0 else "—"
        
        table.add_row(
            r.name, str(r.k), f"{r.tau_low:.2f}", f"{r.tau_high:.2f}", f"{r.alpha:.1f}",
            str(r.n_samples),
            f"{r.accuracy*100:.1f}%",
            f"${r.cost_per_sample:.4f}",
            action_str, tier_str,
            save_str, cache_str,
            f"{r.composite_score:.3f}",
        )
    
    console.print(table)
    
    # Recommendation
    console.rule("[bold]Recommendation[/bold]")
    
    best_overall = max(results, key=lambda x: x.composite_score)
    best_acc = max(results, key=lambda x: x.accuracy)
    acc_qualified = [r for r in results if r.accuracy >= 0.80]
    best_cost = min(acc_qualified, key=lambda x: x.cost_per_sample) if acc_qualified else results[0]
    most_active = max(results, key=lambda x: x.branch_rate + x.escalate_rate)
    
    console.print(f"\n[bold green]🏆 Best overall (composite score):[/bold green] {best_overall.name}")
    console.print(f"   K={best_overall.k}, τ_low={best_overall.tau_low}, "
                  f"τ_high={best_overall.tau_high}, α={best_overall.alpha}")
    console.print(f"   Accuracy: {best_overall.accuracy*100:.1f}% | "
                  f"Cost/sample: ${best_overall.cost_per_sample:.4f} | "
                  f"Save rate: {best_overall.escalate_save_rate*100:.0f}%")
    console.print(f"   Score: {best_overall.composite_score:.3f}")
    
    console.print(f"\n[bold]Alternative recommendations:[/bold]")
    console.print(f"  🎯 Best accuracy: {best_acc.name} ({best_acc.accuracy*100:.1f}%)")
    console.print(f"  💰 Best cost-eff (acc≥80%): {best_cost.name} (${best_cost.cost_per_sample:.4f}/s)")
    console.print(f"  🔀 Most active routing: {most_active.name} "
                  f"(B+E={(most_active.branch_rate+most_active.escalate_rate)*100:.0f}%)")
    
    console.print(f"\n[bold yellow]→ Recommended for 50-sample main experiment:[/bold yellow] "
                  f"[bold]{best_overall.name}[/bold]")
    console.print(f"\n[bold]Next commands:[/bold]")
    console.print(f"  # Copy Phase 1b data as the start of main experiment")
    console.print(f"  cp results/chrcp/math_grid_{best_overall.name}_p1b.jsonl "
                  f"results/chrcp/math_main_50.jsonl")
    console.print(f"  # Add 10 more samples to reach 50 total")
    console.print(f"  # (Modify run_main.py to support --resume + --output_suffix _main)\n")


if __name__ == "__main__":
    main()