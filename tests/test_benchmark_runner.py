"""Smoke test for benchmark runner: 3 samples through CHR-CP on GSM8K.

Verifies:
- benchmark loader works
- runner executes end-to-end
- checkpoint file is written and resumable
"""

from __future__ import annotations
import sys
import os
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from rich.console import Console

from chr_cp.benchmarks.gsm8k import GSM8KBenchmark
from chr_cp.clients import ClientPool
from chr_cp.utils import CostTracker
from experiments.progress_tracker import ProgressTracker
from experiments.run_main import build_chrcp_method, run_one_sample


console = Console()


def test_gsm8k_loader():
    console.rule("[bold cyan]Test 1: GSM8K Loader[/bold cyan]")
    
    bench = GSM8KBenchmark()
    samples = bench.load(n_samples=3)
    
    if len(samples) != 3:
        console.print(f"[red]✗ Expected 3 samples, got {len(samples)}[/red]")
        return False
    
    for s in samples:
        console.print(f"  {s.sample_id}: ref={s.reference}, prompt[:80]={s.prompt[:80]!r}")
        if s.reference is None:
            console.print(f"[red]✗ Sample {s.sample_id} has None reference[/red]")
            return False
    
    console.print("[green]✓ Loaded 3 samples with valid references[/green]")
    return True


def test_progress_tracker():
    console.rule("[bold cyan]Test 2: Progress Tracker[/bold cyan]")
    
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        temp_path = f.name
    
    try:
        tracker = ProgressTracker(temp_path)
        if tracker.done_count != 0:
            console.print("[red]✗ New tracker should have done_count=0[/red]")
            return False
        
        tracker.write({"sample_id": "test_1", "correct": True})
        tracker.write({"sample_id": "test_2", "correct": False})
        
        if tracker.done_count != 2:
            console.print(f"[red]✗ done_count should be 2, got {tracker.done_count}[/red]")
            return False
        
        # Reload from disk
        tracker2 = ProgressTracker(temp_path)
        if tracker2.done_count != 2:
            console.print(f"[red]✗ Reloaded done_count should be 2, got {tracker2.done_count}[/red]")
            return False
        
        if not tracker2.is_done("test_1") or not tracker2.is_done("test_2"):
            console.print("[red]✗ is_done lookup failed[/red]")
            return False
        
        records = tracker2.all_records()
        if len(records) != 2:
            console.print(f"[red]✗ all_records should return 2, got {len(records)}[/red]")
            return False
        
        console.print("[green]✓ Progress tracker works correctly[/green]")
        return True
    finally:
        os.unlink(temp_path)


def test_runner_3_samples():
    """Run 3 GSM8K samples through CHR-CP end-to-end."""
    console.rule("[bold cyan]Test 3: CHR-CP × GSM8K × 3 samples (live)[/bold cyan]")
    
    bench = GSM8KBenchmark()
    samples = bench.load(n_samples=3)
    
    pool = ClientPool.from_config(PROJECT_ROOT / "configs" / "models.yaml")
    cost_tracker = CostTracker()
    method = build_chrcp_method(pool=pool, cost_tracker=cost_tracker)
    
    n_correct = 0
    total_cost = 0.0
    
    for s in samples:
        console.print(f"\n[bold]Running {s.sample_id}...[/bold]")
        record = run_one_sample(method, bench, s)
        
        status = "[green]✓ CORRECT[/green]" if record["correct"] else "[red]✗ WRONG[/red]"
        console.print(
            f"  {status} | extracted={record.get('extracted_answer')} | "
            f"ref={s.reference} | "
            f"cost=${record['cost_usd']:.6f} | "
            f"latency={record['latency_seconds']:.1f}s | "
            f"actions=(S{record.get('num_stay',0)}/B{record.get('num_branch',0)}/E{record.get('num_escalate',0)})"
        )
        
        if record["correct"]:
            n_correct += 1
        total_cost += record["cost_usd"]
    
    console.print(
        f"\n[bold]Smoke test result: {n_correct}/3 correct, "
        f"${total_cost:.6f} total[/bold]"
    )
    
    # Pass criterion: at least 1 of 3 correct (we're not validating accuracy here,
    # just that the pipeline runs without error)
    return n_correct >= 1


def main():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        console.print(f"[red]ERROR: {env_path} not found[/red]")
        sys.exit(1)
    load_dotenv(env_path)
    
    console.print("[bold green]🚀 Day 7 Benchmark Runner Smoke Test[/bold green]\n")
    
    results = {}
    results["gsm8k_loader"] = test_gsm8k_loader()
    console.print()
    results["progress_tracker"] = test_progress_tracker()
    console.print()
    results["runner_live"] = test_runner_3_samples()
    
    console.rule("[bold]Summary[/bold]")
    for name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"  {name}: {status}")
    
    if all(results.values()):
        console.print("\n[bold green]🎉 Day 7 runner ready! "
                      "You can now run full benchmarks.[/bold green]")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()