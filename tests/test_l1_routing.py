"""Day 5 verification: L1 router classifies tasks correctly."""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from chr_cp.routing import L1Router, L1Config, TaskCategory


console = Console()


def test_rule_classifier():
    """Test 1: rule-based classifier categorizes representative tasks."""
    console.rule("[bold cyan]Test 1: L1 Rule-based Classification[/bold cyan]")
    
    router = L1Router(L1Config(mode="rule"))
    
    test_cases = [
        # (task, expected_category)
        (
            "Compute the integral of x^2 * sin(x) from 0 to pi.",
            TaskCategory.MATH,
        ),
        (
            "Solve: 25 + 17 - 8 = ?",
            TaskCategory.MATH,
        ),
        (
            "Write a Python function `def is_prime(n: int) -> bool` that returns True if n is prime.",
            TaskCategory.CODE,
        ),
        (
            "```python\ndef hello():\n    return 'hi'\n```\nRefactor this function.",
            TaskCategory.CODE,
        ),
        (
            "Which of the following is the capital of France?\n"
            "A) London\nB) Paris\nC) Berlin\nD) Madrid",
            TaskCategory.KNOWLEDGE_QA,
        ),
        (
            "If all dogs are mammals, and all mammals are animals, then we can deduce "
            "that all dogs are animals. Therefore, this argument is valid because "
            "the conclusion follows from the premises.",
            TaskCategory.LOGICAL_REASONING,
        ),
        (
            "Write a short story about a dragon meeting a robot.",
            TaskCategory.OPEN_TEXT,
        ),
    ]
    
    table = Table(title="L1 Rule Classification Results")
    table.add_column("Task (truncated)", style="cyan", max_width=50)
    table.add_column("Expected", style="magenta")
    table.add_column("Predicted", style="yellow")
    table.add_column("Confidence", justify="right")
    table.add_column("Topology")
    table.add_column("Pool Size", justify="right")
    table.add_column("Status", style="bold")
    
    correct = 0
    total = len(test_cases)
    
    for task, expected in test_cases:
        config = router.classify_and_configure(task)
        predicted = config.category
        success = predicted == expected
        if success:
            correct += 1
        
        table.add_row(
            task[:48] + ("..." if len(task) > 48 else ""),
            expected.value,
            predicted.value,
            f"{config.classification_confidence:.2f}",
            config.topology,
            str(len(config.agents)),
            "[green]✓[/green]" if success else "[red]✗[/red]",
        )
    
    console.print(table)
    console.print(f"\n[bold]Accuracy: {correct}/{total} = {correct/total*100:.1f}%[/bold]")
    
    # Allow some tolerance: rule classifier doesn't need 100%
    return correct >= total * 0.7  # at least 70% on these clean cases


def test_pool_config_consistency():
    """Test 2: every category produces a non-empty, valid pool config."""
    console.rule("[bold cyan]Test 2: Pool Config Validity[/bold cyan]")
    
    router = L1Router(L1Config(mode="rule"))
    
    valid_topologies = {"star", "chain", "mesh", "tree"}
    valid_tiers = {"T1", "T2", "T3", "T4"}
    valid_roles = {"solver", "verifier", "aggregator", "compressor", "escalator"}
    
    table = Table(title="Pool Config Per Category")
    table.add_column("Category")
    table.add_column("Topology")
    table.add_column("Agents")
    table.add_column("Status", style="bold")
    
    all_pass = True
    sample_tasks = {
        TaskCategory.MATH: "Compute 5 * 7.",
        TaskCategory.CODE: "Write a Python function `def add(a, b)`.",
        TaskCategory.KNOWLEDGE_QA: "Which is the capital? A) X B) Y C) Z D) W",
        TaskCategory.LOGICAL_REASONING: "If A implies B, and B implies C, then deduce A→C.",
        TaskCategory.OPEN_TEXT: "Tell me about dragons.",
    }
    
    for cat, task in sample_tasks.items():
        config = router.classify_and_configure(task)
        
        valid = (
            config.topology in valid_topologies
            and len(config.agents) >= 1
            and all(role in valid_roles for role, _ in config.agents)
            and all(tier in valid_tiers for _, tier in config.agents)
        )
        
        if not valid:
            all_pass = False
        
        agents_str = ", ".join(f"{role}/{tier}" for role, tier in config.agents)
        table.add_row(
            cat.value,
            config.topology,
            agents_str,
            "[green]✓[/green]" if valid else "[red]✗[/red]",
        )
    
    console.print(table)
    return all_pass


def main():
    console.print("[bold green]🚀 CHR-CP Day 5 Verification (L1)[/bold green]\n")
    
    results = {}
    results["rule_classifier"] = test_rule_classifier()
    console.print()
    results["pool_config_consistency"] = test_pool_config_consistency()
    
    console.rule("[bold]Final Summary[/bold]")
    summary = Table()
    summary.add_column("Test")
    summary.add_column("Result", style="bold")
    
    for name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        summary.add_row(name, status)
    
    console.print(summary)
    
    if all(results.values()):
        console.print("\n[bold green]🎉 Day 5 (L1) done![/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]❌ Some tests failed.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()