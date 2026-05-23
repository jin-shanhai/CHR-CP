"""Real-time progress monitor for Phase 1b sensitivity grid.

Runs while sensitivity_grid_v2.py is still executing. Reads current
state of each config's JSONL and reports:
  - Per-config progress (X/50 samples)
  - Per-config metrics (accuracy, cost, routing distribution, save rate)
  - Cross-config ranking by composite score
  - Current "leading" config (may change as data accumulates)

Usage:
    python -m tests.analyze_phase1b
"""

from __future__ import annotations
import sys
import json
import statistics
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

console = Console()


# Phase 1b candidates (5 configs)
# Must match GRID_CONFIGS in experiments/sensitivity_grid_v2.py
PHASE1B_CONFIGS = {
    "E2": (5, 0.10, 0.50, 0.7),
    "D3": (5, 0.10, 0.50, 0.8),
    "F3": (5, 0.10, 0.55, 0.9),
    "D2": (5, 0.10, 0.45, 0.8),
    "E1": (5, 0.10, 0.45, 0.7),
}

TARGET_SAMPLES = 50  # Phase 1b target sample count per config


@dataclass
class ConfigProgress:
    """Current state of one Phase 1b configuration."""
    name: str
    k: int
    tau_low: float
    tau_high: float
    alpha: float
    
    # Progress
    n_samples: int = 0
    target_samples: int = TARGET_SAMPLES
    
    # Metrics (None if no data yet)
    accuracy: Optional[float] = None
    total_cost: float = 0.0
    cost_per_sample: float = 0.0
    avg_latency: float = 0.0
    
    n_stay: int = 0
    n_branch: int = 0
    n_escalate: int = 0
    branch_rate: float = 0.0
    escalate_rate: float = 0.0
    
    final_tier_dist: dict = field(default_factory=dict)
    
    n_escalate_samples: int = 0
    escalate_save_rate: float = 0.0
    
    cache_factor_first_third: float = 1.0
    cache_factor_last_third: float = 1.0
    cache_factor_overall: float = 1.0
    
    composite_score: float = 0.0
    
    # Status
    status: str = "pending"  # pending / in-progress / complete
    
    @property
    def progress_pct(self) -> float:
        return self.n_samples / max(1, self.target_samples) * 100


def load_progress(name: str, k: int, tau_low: float, tau_high: float, alpha: float) -> ConfigProgress:
    """Load current state of one Phase 1b configuration from its JSONL."""
    p = ConfigProgress(name=name, k=k, tau_low=tau_low, tau_high=tau_high, alpha=alpha)
    
    path = PROJECT_ROOT / "results" / "chrcp" / f"math_grid_{name}_p1b.jsonl"
    if not path.exists():
        return p
    
    with open(path) as f:
        records = [json.loads(line) for line in f]
    
    p.n_samples = len(records)
    if p.n_samples == 0:
        return p
    
    p.status = "complete" if p.n_samples >= TARGET_SAMPLES else "in-progress"
    
    # Accuracy
    n_correct = sum(1 for r in records if r.get("correct"))
    p.accuracy = n_correct / p.n_samples
    
    # Cost
    p.total_cost = sum(r.get("cost_usd", 0) for r in records)
    p.cost_per_sample = p.total_cost / p.n_samples
    
    # Latency
    p.avg_latency = sum(r.get("latency_seconds", 0) for r in records) / p.n_samples
    
    # Action distribution
    p.n_stay = sum(r.get("num_stay", 0) for r in records)
    p.n_branch = sum(r.get("num_branch", 0) for r in records)
    p.n_escalate = sum(r.get("num_escalate", 0) for r in records)
    total_actions = max(1, p.n_stay + p.n_branch + p.n_escalate)
    p.branch_rate = p.n_branch / total_actions
    p.escalate_rate = p.n_escalate / total_actions
    
    # Final tier distribution
    p.final_tier_dist = dict(Counter(r.get("final_tier", "?") for r in records))
    
    # ESCALATE save rate
    esc_samples = [r for r in records if r.get("num_escalate", 0) > 0]
    p.n_escalate_samples = len(esc_samples)
    if esc_samples:
        n_esc_correct = sum(1 for r in esc_samples if r.get("correct"))
        p.escalate_save_rate = n_esc_correct / len(esc_samples)
    
    # Cache factor trend (split into thirds to see CA²R decay)
    cfs = [r.get("avg_cache_factor", 1.0) for r in records if "avg_cache_factor" in r]
    if len(cfs) >= 6:
        third = len(cfs) // 3
        p.cache_factor_first_third = statistics.mean(cfs[:third])
        p.cache_factor_last_third = statistics.mean(cfs[-third:])
    if cfs:
        p.cache_factor_overall = statistics.mean(cfs)
    
    # Composite score (same weights as analyze_sensitivity.py)
    if p.accuracy is not None:
        activeness = min(1.0, (p.branch_rate + p.escalate_rate) / 0.25)
        cost_eff = max(0, 1.0 - p.cost_per_sample / 0.05)
        p.composite_score = (
            0.50 * p.accuracy
            + 0.20 * activeness
            + 0.20 * p.escalate_save_rate
            + 0.10 * cost_eff
        )
    
    return p


def render_progress_table(progresses: List[ConfigProgress]) -> Table:
    """Main progress table — all configs sorted by current composite score."""
    table = Table(title=f"Phase 1b Real-Time Progress (target: {TARGET_SAMPLES} samples per config)")
    table.add_column("Cfg", style="bold")
    table.add_column("K", justify="center")
    table.add_column("τ_high", justify="right")
    table.add_column("α", justify="right")
    table.add_column("Progress", justify="center")
    table.add_column("Status", style="bold")
    table.add_column("Acc", justify="right", style="green")
    table.add_column("$/sample", justify="right")
    table.add_column("S/B/E %", justify="center")
    table.add_column("Final Tiers", justify="center")
    table.add_column("Save%", justify="right", style="cyan")
    table.add_column("Cache↓", justify="right")
    table.add_column("Score", justify="right", style="bold yellow")
    
    # Sort by composite score (descending), but pending configs go last
    def sort_key(p: ConfigProgress):
        if p.status == "pending":
            return (-1, 0)
        return (1, p.composite_score)
    
    sorted_progresses = sorted(progresses, key=sort_key, reverse=True)
    
    for p in sorted_progresses:
        # Progress bar
        progress_bar = f"{p.n_samples}/{p.target_samples}"
        if p.status == "complete":
            progress_str = f"[green]{progress_bar} ✓[/green]"
        elif p.status == "in-progress":
            pct = p.progress_pct
            color = "yellow" if pct < 50 else "cyan"
            progress_str = f"[{color}]{progress_bar} ({pct:.0f}%)[/{color}]"
        else:
            progress_str = f"[grey]—[/grey]"
        
        # Status color
        status_color = {"complete": "green", "in-progress": "cyan", "pending": "grey"}
        status_str = f"[{status_color.get(p.status, 'white')}]{p.status}[/]"
        
        # If no data, render placeholder row
        if p.n_samples == 0:
            table.add_row(
                p.name, str(p.k), f"{p.tau_high:.2f}", f"{p.alpha:.1f}",
                progress_str, status_str,
                "—", "—", "—", "—", "—", "—", "—",
            )
            continue
        
        stay_pct = p.n_stay / max(1, p.n_stay + p.n_branch + p.n_escalate) * 100
        action_str = f"{stay_pct:.0f}/{p.branch_rate*100:.0f}/{p.escalate_rate*100:.0f}"
        
        tier_str = ",".join(
            f"{t}:{p.final_tier_dist.get(t, 0)}" 
            for t in ["T1", "T2", "T3", "T4"]
            if p.final_tier_dist.get(t, 0) > 0
        ) or "—"
        
        save_str = f"{p.escalate_save_rate*100:.0f}%" if p.n_escalate_samples > 0 else "—"
        
        cache_drop = p.cache_factor_first_third - p.cache_factor_last_third
        cache_str = f"-{cache_drop*100:.1f}%" if cache_drop > 0.005 else "—"
        
        acc_str = f"{p.accuracy*100:.1f}%" if p.accuracy is not None else "—"
        cost_str = f"${p.cost_per_sample:.4f}" if p.n_samples > 0 else "—"
        score_str = f"{p.composite_score:.3f}" if p.accuracy is not None else "—"
        
        table.add_row(
            p.name, str(p.k), f"{p.tau_high:.2f}", f"{p.alpha:.1f}",
            progress_str, status_str,
            acc_str, cost_str,
            action_str, tier_str, save_str, cache_str, score_str,
        )
    
    return table


def render_leader_box(progresses: List[ConfigProgress]) -> None:
    """Print current leading config (if any have data)."""
    have_data = [p for p in progresses if p.accuracy is not None and p.n_samples >= 10]
    if not have_data:
        console.print("[grey]No config has ≥10 samples yet; rankings not stable.[/grey]")
        return
    
    leader = max(have_data, key=lambda p: p.composite_score)
    
    console.print(f"\n[bold yellow]🏆 Current leader (≥10 samples):[/bold yellow] {leader.name}")
    console.print(f"   K={leader.k}, τ_low={leader.tau_low}, "
                  f"τ_high={leader.tau_high}, α={leader.alpha}")
    console.print(f"   Acc: {leader.accuracy*100:.1f}% | "
                  f"Cost/sample: ${leader.cost_per_sample:.4f} | "
                  f"Save: {leader.escalate_save_rate*100:.0f}% | "
                  f"Score: {leader.composite_score:.3f}")
    
    # Show top 3 by accuracy if different from leader
    top_acc = sorted(have_data, key=lambda p: -p.accuracy)[:3]
    if top_acc[0].name != leader.name:
        console.print(f"\n[bold]Top by accuracy:[/bold]")
        for i, p in enumerate(top_acc, 1):
            console.print(
                f"   {i}. {p.name}: {p.accuracy*100:.1f}% "
                f"(${p.cost_per_sample:.4f}/sample)"
            )


def render_summary_stats(progresses: List[ConfigProgress]) -> None:
    """Print aggregate stats across all configs."""
    completed = [p for p in progresses if p.status == "complete"]
    in_progress = [p for p in progresses if p.status == "in-progress"]
    pending = [p for p in progresses if p.status == "pending"]
    
    total_samples = sum(p.n_samples for p in progresses)
    total_target = TARGET_SAMPLES * len(progresses)
    total_cost = sum(p.total_cost for p in progresses)
    
    console.print(f"\n[bold]Overall Phase 1b Progress:[/bold]")
    console.print(f"  Completed: {len(completed)}/{len(progresses)} configs")
    console.print(f"  In-progress: {len(in_progress)} configs")
    console.print(f"  Pending: {len(pending)} configs")
    console.print(f"  Samples: {total_samples}/{total_target} ({total_samples/total_target*100:.0f}%)")
    console.print(f"  Total cost so far: ${total_cost:.3f}")
    
    if in_progress:
        names = ", ".join(p.name for p in in_progress)
        console.print(f"  [cyan]Currently running: {names}[/cyan]")


def main():
    console.rule("[bold cyan]CHR-CP Phase 1b Real-Time Monitor[/bold cyan]")
    
    progresses = []
    for name, (k, tau_low, tau_high, alpha) in PHASE1B_CONFIGS.items():
        p = load_progress(name, k, tau_low, tau_high, alpha)
        progresses.append(p)
    
    # Main table
    table = render_progress_table(progresses)
    console.print(table)
    
    # Summary stats
    render_summary_stats(progresses)
    
    # Leader announcement
    render_leader_box(progresses)
    
    console.print(f"\n[grey]Run again with `python -m tests.analyze_phase1b` to refresh.[/grey]")
    console.print(f"[grey]When all 5 configs are complete, run `python -m tests.analyze_sensitivity` "
                  f"for the final recommendation.[/grey]\n")


if __name__ == "__main__":
    main()