"""Phase 1a + 1b sensitivity grid runner.

Phase 1a: 15 configs × 15 samples (coarse screening)
Phase 1b: kept configs × 40 samples (refined evaluation)

Total estimated: ~600 samples, ~$11-18, ~6-9 hours.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

console = Console()


# ============================================================================
# Grid configuration (15 configs)
# ============================================================================
GRID_CONFIGS: List[Dict] = [
    # Phase 1b: Top-5 configs from Phase 1a cost/latency screening
    {"name": "E2", "k": 5, "tau_low": 0.10, "tau_high": 0.50, "alpha": 0.7},
    {"name": "D3", "k": 5, "tau_low": 0.10, "tau_high": 0.50, "alpha": 0.8},
    {"name": "F3", "k": 5, "tau_low": 0.10, "tau_high": 0.55, "alpha": 0.9},
    {"name": "D2", "k": 5, "tau_low": 0.10, "tau_high": 0.45, "alpha": 0.8},
    {"name": "E1", "k": 5, "tau_low": 0.10, "tau_high": 0.45, "alpha": 0.7},
]

def get_output_path(config_name: str, phase: str, benchmark: str) -> Path:
    """E.g., math_grid_D2_p1a.jsonl, math_grid_D2_p1b.jsonl"""
    return PROJECT_ROOT / "results" / "chrcp" / f"{benchmark}_grid_{config_name}_{phase}.jsonl"


def get_suffix(config_name: str, phase: str) -> str:
    return f"_grid_{config_name}_{phase}"


def run_single_config(
    config: Dict,
    n_samples: int,
    phase: str,
    benchmark: str,
    max_cost: float,
) -> Dict:
    """Run one configuration for a given phase."""
    output_path = get_output_path(config["name"], phase, benchmark)
    suffix = get_suffix(config["name"], phase)
    
    # Resume support
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        if existing >= n_samples:
            console.print(f"[yellow]⏭  {config['name']} {phase}: already complete ({existing}/{n_samples})[/yellow]")
            return {"name": config["name"], "phase": phase, "status": "skipped", "n_samples": existing}
    
    cmd = [
        sys.executable, "-m", "experiments.run_main",
        "--method", "chrcp",
        "--benchmark", benchmark,
        "--n_samples", str(n_samples),
        "--concurrency", "4",
        "--max_cost_usd", str(max_cost),
        "--k_samples", str(config["k"]),
        "--tau_low", str(config["tau_low"]),
        "--tau_high", str(config["tau_high"]),
        "--alpha", str(config["alpha"]),
        "--output_suffix", suffix,
    ]
    
    console.print(f"\n[bold cyan]▶ {config['name']} ({phase})[/bold cyan]: "
                  f"K={config['k']}, τ_low={config['tau_low']}, "
                  f"τ_high={config['tau_high']}, α={config['alpha']}")
    
    t_start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - t_start
        
        if result.returncode != 0:
            console.print(f"[red]✗ FAILED (rc={result.returncode})[/red]")
            console.print(f"[red]stderr: {result.stderr[-300:]}[/red]")
            return {"name": config["name"], "phase": phase, "status": "failed",
                    "elapsed": elapsed, "error": result.stderr[-200:]}
        
        # Parse from stdout
        accuracy, cost = None, None
        for line in result.stdout.split("\n"):
            if "Accuracy:" in line and "=" in line:
                try:
                    parts = line.split("=")[-1].strip().rstrip("%")
                    accuracy = float(parts) / 100
                except (ValueError, IndexError):
                    pass
            if "Total cost:" in line:
                try:
                    cost = float(line.split("$")[-1].strip())
                except (ValueError, IndexError):
                    pass
        
        acc_str = f"acc={accuracy*100:.1f}%" if accuracy else "acc=?"
        cost_str = f"${cost:.3f}" if cost is not None else "$?"
        console.print(f"[green]✓ {config['name']} {phase} done {elapsed:.0f}s | {acc_str} cost={cost_str}[/green]")
        
        return {
            "name": config["name"], "phase": phase, "status": "ok",
            "elapsed": elapsed, "accuracy": accuracy, "cost": cost,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t_start
        console.print(f"[red]✗ TIMEOUT after {elapsed:.0f}s[/red]")
        return {"name": config["name"], "phase": phase, "status": "timeout", "elapsed": elapsed}


def screen_phase1a(p1a_results: List[Dict], benchmark: str) -> List[Dict]:
    """Apply screening rules to Phase 1a results.
    
    Returns the list of configs that pass screening (kept for Phase 1b).
    """
    console.rule("[bold cyan]Phase 1a Screening[/bold cyan]")
    
    kept: List[Dict] = []
    for r in p1a_results:
        if r["status"] != "ok":
            console.print(f"[red]✗ {r['name']}: {r['status']}, excluded[/red]")
            continue
        
        cfg = next((c for c in GRID_CONFIGS if c["name"] == r["name"]), None)
        if not cfg:
            continue
        
        # Load jsonl for detailed metrics
        path = get_output_path(r["name"], "p1a", benchmark)
        if not path.exists():
            console.print(f"[yellow]⚠ {r['name']}: jsonl missing, excluded[/yellow]")
            continue
        
        with open(path) as f:
            records = [json.loads(line) for line in f]
        
        if not records:
            console.print(f"[yellow]⚠ {r['name']}: no records, excluded[/yellow]")
            continue
        
        # Compute screening metrics
        n = len(records)
        n_correct = sum(1 for x in records if x.get("correct"))
        accuracy = n_correct / n
        total_cost = sum(x.get("cost_usd", 0) for x in records)
        cost_per_sample = total_cost / n
        
        n_stay = sum(x.get("num_stay", 0) for x in records)
        n_branch = sum(x.get("num_branch", 0) for x in records)
        n_escalate = sum(x.get("num_escalate", 0) for x in records)
        total = max(1, n_stay + n_branch + n_escalate)
        routing_active = (n_branch + n_escalate) / total
        
        # Screening rules
        reasons = []
        if accuracy < 0.70:
            reasons.append(f"acc {accuracy*100:.0f}% < 70%")
        if routing_active < 0.05:
            reasons.append(f"routing {routing_active*100:.0f}% < 5%")
        if cost_per_sample > 0.15:
            reasons.append(f"cost/sample ${cost_per_sample:.3f} > $0.15")
        
        if reasons:
            console.print(
                f"[red]✗ {r['name']}: acc={accuracy*100:.1f}%, "
                f"route={routing_active*100:.0f}%, ${cost_per_sample:.3f}/s "
                f"→ excluded ({'; '.join(reasons)})[/red]"
            )
        else:
            console.print(
                f"[green]✓ {r['name']}: acc={accuracy*100:.1f}%, "
                f"route={routing_active*100:.0f}%, ${cost_per_sample:.3f}/s "
                f"→ KEPT[/green]"
            )
            kept.append(cfg)
    
    console.print(f"\n[bold]Phase 1a: {len(kept)}/{len(p1a_results)} configs passed[/bold]")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1a_samples", type=int, default=15, help="Phase 1a samples per config")
    parser.add_argument("--p1b_samples", type=int, default=40, help="Phase 1b samples per config")
    parser.add_argument("--benchmark", default="math")
    parser.add_argument("--p1a_max_cost", type=float, default=0.5, help="Phase 1a cost cap per config")
    parser.add_argument("--p1b_max_cost", type=float, default=1.8, help="Phase 1b cost cap per config")
    parser.add_argument("--phase", choices=["1a", "1b", "all"], default="all",
                        help="Run only specified phase, or all")
    parser.add_argument("--start_from", type=str, default=None,
                        help="Resume from config name (e.g., 'D2')")
    args = parser.parse_args()
    
    console.rule("[bold cyan]CHR-CP Sensitivity Grid v2 (Phase 1a + 1b)[/bold cyan]")
    console.print(f"Benchmark: {args.benchmark}")
    console.print(f"Phase 1a: {len(GRID_CONFIGS)} configs × {args.p1a_samples} samples")
    console.print(f"Phase 1b: kept configs × {args.p1b_samples} samples")
    console.print(f"Phase mode: {args.phase}\n")
    
    total_start = time.time()
    
    # ============== Phase 1a ==============
    p1a_results = []
    if args.phase in ("1a", "all"):
        console.rule("[bold]Phase 1a: Coarse Screening[/bold]")
        configs = GRID_CONFIGS
        if args.start_from:
            idx = next((i for i, c in enumerate(GRID_CONFIGS) if c["name"] == args.start_from), None)
            if idx is not None:
                configs = GRID_CONFIGS[idx:]
                console.print(f"[yellow]Resuming from {args.start_from}[/yellow]\n")
        
        for i, config in enumerate(configs, 1):
            console.print(f"\n[bold]== [1a {i}/{len(configs)}] {config['name']} ==[/bold]")
            result = run_single_config(
                config=config,
                n_samples=args.p1a_samples,
                phase="p1a",
                benchmark=args.benchmark,
                max_cost=args.p1a_max_cost,
            )
            p1a_results.append(result)
    else:
        # Load existing Phase 1a results from disk
        for config in GRID_CONFIGS:
            path = get_output_path(config["name"], "p1a", args.benchmark)
            if path.exists():
                with open(path) as f:
                    n = sum(1 for _ in f)
                p1a_results.append({"name": config["name"], "phase": "p1a",
                                    "status": "ok" if n >= args.p1a_samples else "incomplete",
                                    "n_samples": n})
    
    # Screen Phase 1a
    kept_configs = screen_phase1a(p1a_results, args.benchmark)
    
    # ============== Phase 1b ==============
    p1b_results = []
    if args.phase in ("1b", "all") and kept_configs:
        console.rule("[bold]Phase 1b: Refined Evaluation[/bold]")
        for i, config in enumerate(kept_configs, 1):
            console.print(f"\n[bold]== [1b {i}/{len(kept_configs)}] {config['name']} ==[/bold]")
            result = run_single_config(
                config=config,
                n_samples=args.p1b_samples,
                phase="p1b",
                benchmark=args.benchmark,
                max_cost=args.p1b_max_cost,
            )
            p1b_results.append(result)
    elif not kept_configs:
        console.print("[red]Phase 1a screened out all configs. Cannot proceed to 1b.[/red]")
        console.print("[yellow]Consider relaxing screening rules in screen_phase1a().[/yellow]")
    
    # ============== Final summary ==============
    total_elapsed = time.time() - total_start
    console.rule("[bold]Grid Execution Summary[/bold]")
    
    summary_table = Table()
    summary_table.add_column("Phase")
    summary_table.add_column("Configs")
    summary_table.add_column("Samples", justify="right")
    summary_table.add_column("Status")
    
    summary_table.add_row(
        "1a",
        str(len([r for r in p1a_results if r["status"] == "ok"])),
        f"{args.p1a_samples * len([r for r in p1a_results if r['status'] == 'ok'])}",
        f"{sum(1 for r in p1a_results if r['status']=='ok')}/{len(p1a_results)} ok",
    )
    summary_table.add_row(
        "1b",
        str(len(kept_configs)),
        f"{args.p1b_samples * len(p1b_results)}",
        f"{sum(1 for r in p1b_results if r['status']=='ok')}/{len(p1b_results)} ok",
    )
    
    console.print(summary_table)
    console.print(f"\n[bold]Total elapsed: {total_elapsed/60:.1f} min[/bold]")
    
    console.print(f"\n[bold cyan]Next step:[/bold cyan]")
    console.print(f"  python -m tests.analyze_sensitivity")
    console.print(f"  (This will read Phase 1b data and recommend the best config)\n")


if __name__ == "__main__":
    main()