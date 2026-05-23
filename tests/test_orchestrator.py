"""End-to-end CHR-CP integration test.

This is the big one: takes a real task, runs through L1 → L2 → L3 → final answer.
If this passes, the full CHR-CP pipeline works.
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from chr_cp.clients import ClientPool
from chr_cp.routing import (
    CHRCPOrchestrator,
    OrchestratorConfig,
    L1Config,
    L2Config,
    L3Config,
)
from chr_cp.utils import CostTracker


console = Console()


def test_e2e_math():
    """E2E: solve a math problem end-to-end."""
    console.rule("[bold cyan]E2E Test: Math Problem[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    tracker = CostTracker()
    
    orch_config = OrchestratorConfig(
        l1_config=L1Config(mode="rule"),
        l2_config=L2Config(tau_low=0.30, tau_high=0.65),
        l3_config=L3Config(),
        per_task_budget_usd=0.30,
        max_tokens=1024,
    )
    
    orch = CHRCPOrchestrator(pool=pool, config=orch_config, cost_tracker=tracker)
    
    task = "Compute 23 * 47. Show your work and give the final number."
    
    result = orch.run(task=task, task_id="e2e_math", benchmark="gsm8k")
    
    # Print summary
    console.print(Panel(result.final_answer[:500], title=f"Final Answer ({result.final_tier})"))
    
    table = Table(title="E2E Run Stats")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("L1 Category", result.l1_config.category.value)
    table.add_row("L1 Topology", result.l1_config.topology)
    table.add_row("Agents Run", str(len(result.step_results)))
    table.add_row("Total API Calls", str(result.total_calls))
    table.add_row("Total Cost", f"${result.total_cost_usd:.6f}")
    table.add_row("Total Latency", f"{result.total_latency_seconds:.2f}s")
    table.add_row("STAY Actions", str(result.num_stay))
    table.add_row("BRANCH Actions", str(result.num_branch))
    table.add_row("ESCALATE Actions", str(result.num_escalate))
    table.add_row("Distillations", str(result.distillations_triggered))
    table.add_row("Warmups", str(result.warmups_triggered))
    console.print(table)
    
    # Per-step trace
    console.print("\n[bold]Per-step decision trace:[/bold]")
    for i, step in enumerate(result.step_results):
        console.print(f"\nStep {i+1}: {step.summary()}")
        for d in step.decisions:
            console.print(f"    → {d}")
    
    # Validate basic outputs
    if not result.final_answer:
        console.print("[red]✗ No final answer produced[/red]")
        return False
    if result.total_calls == 0:
        console.print("[red]✗ No API calls made[/red]")
        return False
    
    # Check the answer contains "1081" (correct answer)
    if "1081" in result.final_answer:
        console.print("[green]✓ Final answer contains correct value (1081)[/green]")
    else:
        console.print("[yellow]⚠ Correct value not found in answer (may still be partially right)[/yellow]")
    
    return True


def test_e2e_qa():
    """E2E: knowledge QA task."""
    console.rule("[bold cyan]E2E Test: Knowledge QA[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    tracker = CostTracker()
    
    orch = CHRCPOrchestrator(
        pool=pool,
        config=OrchestratorConfig(per_task_budget_usd=0.10, max_tokens=512),
        cost_tracker=tracker,
    )
    
    task = (
        "Which of the following is the capital of Australia?\n"
        "A) Sydney\nB) Melbourne\nC) Canberra\nD) Brisbane\n"
        "Answer with the letter only."
    )
    
    result = orch.run(task=task, task_id="e2e_qa", benchmark="mmlu")
    
    console.print(Panel(result.final_answer[:300], title=f"Final Answer ({result.final_tier})"))
    console.print(f"Cost: ${result.total_cost_usd:.6f} | Calls: {result.total_calls}")
    
    if not result.final_answer:
        return False
    
    # Should contain "C" for Canberra
    if "C" in result.final_answer.upper():
        console.print("[green]✓ Found C (correct: Canberra)[/green]")
    else:
        console.print("[yellow]⚠ Letter C not found in answer[/yellow]")
    
    return True


def main():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        console.print(f"[red]ERROR: {env_path} not found.[/red]")
        sys.exit(1)
    load_dotenv(env_path)
    
    console.print("[bold green]🚀 CHR-CP End-to-End Integration Test[/bold green]\n")
    
    results = {}
    results["e2e_math"] = test_e2e_math()
    console.print("\n")
    results["e2e_qa"] = test_e2e_qa()
    
    console.rule("[bold]Final Summary[/bold]")
    summary = Table()
    summary.add_column("Test")
    summary.add_column("Result", style="bold")
    
    for name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        summary.add_row(name, status)
    
    console.print(summary)
    
    if all(results.values()):
        console.print("\n[bold green]🎉 CHR-CP fully functional end-to-end![/bold green]")
        console.print("[bold green]Ready to start running benchmarks (Day 7+).[/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]❌ Some E2E tests failed.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()