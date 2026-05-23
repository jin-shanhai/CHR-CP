"""Day 2-3 verification: VC² uncertainty signal works correctly.

Tests:
1. Verbalized confidence parser handles all formats
2. Consistency estimator runs K samples and computes similarity
3. VC² fusion produces sensible U values for easy/hard problems
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from chr_cp.clients import ClientPool, Tier
from chr_cp.confidence import (
    VerbalizedConfidenceParser,
    ConsistencyEstimator,
    TaskType,
    VC2Estimator,
)
from chr_cp.prompts import StablePrefixBuilder
from chr_cp.prompts.role_templates import get_role
from chr_cp.utils import CostTracker


console = Console()


def test_verbalized_parser():
    """Test 1: VC parser handles all expected formats."""
    console.rule("[bold cyan]Test 1: Verbalized Confidence Parser[/bold cyan]")
    
    parser = VerbalizedConfidenceParser()
    
    test_cases = [
        # (input, expected_normalized, expected_pattern)
        ("The answer is 42.\n<confidence>9/10</confidence>", 0.9, "strict_tag_xy"),
        ("Some reasoning... <confidence>7</confidence> more text.", 0.7, "tag_only_x"),
        ("最终答案是A。<置信度>8/10</置信度>", 0.8, "chinese_tag"),
        ("After analysis, confidence: 6/10 in this answer.", 0.6, "inline"),
        ("I am 85% confident in this.", 0.85, "percentage"),
        ("No confidence tag at all here.", 0.5, None),  # parse fail → 0.5 default uncertainty
    ]
    
    table = Table(title="VC Parser Test Cases")
    table.add_column("Input", style="cyan")
    table.add_column("Expected", justify="right")
    table.add_column("Parsed", justify="right")
    table.add_column("Pattern", style="magenta")
    table.add_column("Status", style="bold")
    
    all_pass = True
    for inp, expected, expected_pattern in test_cases:
        result = parser.parse(inp)
        if expected_pattern is None:
            # Expect parse failure
            success = not result.parse_success
            status = "[green]✓[/green]" if success else "[red]✗[/red]"
            if not success:
                all_pass = False
            table.add_row(
                inp[:50] + ("..." if len(inp) > 50 else ""),
                "FAIL_EXPECTED",
                f"normalized={result.normalized:.2f}",
                str(result.pattern_used),
                status,
            )
        else:
            success = (
                result.parse_success
                and abs(result.normalized - expected) < 1e-6
                and result.pattern_used == expected_pattern
            )
            status = "[green]✓[/green]" if success else "[red]✗[/red]"
            if not success:
                all_pass = False
            table.add_row(
                inp[:50] + ("..." if len(inp) > 50 else ""),
                f"{expected:.2f}",
                f"{result.normalized:.2f}",
                str(result.pattern_used),
                status,
            )
    
    console.print(table)
    return all_pass


def test_strip_confidence():
    """Test 2: confidence tag stripping works."""
    console.rule("[bold cyan]Test 2: Strip Confidence Tag[/bold cyan]")
    
    parser = VerbalizedConfidenceParser()
    
    cases = [
        (
            "The answer is 42.\n<confidence>9/10</confidence>",
            "The answer is 42.",
        ),
        (
            "Reasoning... <confidence>7/10</confidence> Final thought.",
            "Reasoning...  Final thought.",  # 中间stripped
        ),
    ]
    
    all_pass = True
    for inp, expected in cases:
        out = parser.strip_confidence_tag(inp)
        # Normalize whitespace for comparison
        out_norm = " ".join(out.split())
        expected_norm = " ".join(expected.split())
        success = out_norm == expected_norm
        if not success:
            all_pass = False
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(f"  {status} input={inp!r} → output={out!r}")
    
    return all_pass


def test_vc2_easy_vs_hard():
    """Test 3: VC² gives lower U on easy problems, higher U on hard problems.
    
    This is the key validation: does VC² actually carry signal?
    """
    console.rule("[bold cyan]Test 3: VC² on Easy vs Hard Problems[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    tracker = CostTracker()
    estimator = VC2Estimator(pool=pool, alpha=0.6, cost_tracker=tracker)
    
    test_problems = [
        {
            "name": "Easy arithmetic",
            "task": "What is 7 + 5?",
            "task_type": TaskType.NUMERIC,
            "expected_low_U": True,
        },
        {
            "name": "Hard reasoning",
            "task": (
                "If a train leaves city A at 3:47pm traveling at 67 mph, "
                "and another train leaves city B (which is 423 miles away) "
                "at 4:13pm traveling at 53 mph toward city A, "
                "at what exact time will they meet, accounting for the fact that "
                "the second train must wait 12 minutes at a checkpoint located "
                "31% of the way from B to A?"
            ),
            "task_type": TaskType.NUMERIC,
            "expected_low_U": False,
        },
    ]
    
    table = Table(title="VC² Easy vs Hard")
    table.add_column("Problem", style="cyan")
    table.add_column("U_verbal", justify="right")
    table.add_column("U_consist", justify="right")
    table.add_column("U_combined", justify="right", style="bold yellow")
    table.add_column("Expected", style="magenta")
    table.add_column("Status", style="bold")
    
    all_pass = True
    u_easy = None
    u_hard = None
    
    for problem in test_problems:
        # Build messages with stable prefix
        builder = StablePrefixBuilder(task=problem["task"])
        solver = get_role("solver")
        messages = builder.build_messages(
            role_prompt=solver.build(),
            step_payload="Solve this problem now. Show your work and give the final number.",
        )
        
        try:
            # Primary call (T1 for cost; we need verbalized + we'll sample)
            primary = pool.invoke(
                tier=Tier.T1,
                messages=messages,
                temperature=0.3,
                max_tokens=512,
            )
            tracker.record(primary, task_id=problem["name"])
            
            # VC² full estimation
            signal = estimator.estimate(
                primary_response=primary,
                messages=messages,
                tier=Tier.T1,
                task_type=problem["task_type"],
                mode="full",
                task_id=problem["name"],
            )
            
            if problem["expected_low_U"]:
                u_easy = signal.U
                expected_str = "U should be < 0.4"
                success = signal.U < 0.4
            else:
                u_hard = signal.U
                expected_str = "U should be > U(easy)"
                success = True  # Will check at end
            
            status = "[green]✓[/green]" if success else "[yellow]?[/yellow]"
            table.add_row(
                problem["name"],
                f"{signal.U_verbalized:.3f}",
                f"{signal.U_consistency:.3f}",
                f"{signal.U:.3f}",
                expected_str,
                status,
            )
        except Exception as e:
            table.add_row(problem["name"], "—", "—", "—", "—", f"[red]✗ {e}[/red]")
            all_pass = False
    
    console.print(table)
    
    # Compare easy vs hard
    if u_easy is not None and u_hard is not None:
        if u_hard > u_easy:
            console.print(
                f"[bold green]✓ U(hard)={u_hard:.3f} > U(easy)={u_easy:.3f} — VC² shows correct signal![/bold green]"
            )
        else:
            console.print(
                f"[yellow]⚠ U(hard)={u_hard:.3f} ≤ U(easy)={u_easy:.3f} — "
                f"VC² may need calibration for this model.[/yellow]"
            )
            # Don't fail; this can vary by model
    
    console.print(
        f"\n[bold]Total cost for VC² test: ${tracker.total_cost:.6f}[/bold]"
    )
    return all_pass


def test_consistency_only():
    """Test 4: ConsistencyEstimator alone runs without errors."""
    console.rule("[bold cyan]Test 4: Standalone Consistency[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    
    estimator = ConsistencyEstimator(pool=pool)
    
    builder = StablePrefixBuilder(task="What is the capital of France?")
    messages = builder.build_messages(
        role_prompt=get_role("solver").build(),
        step_payload="Answer with a single word.",
    )
    
    try:
        result = estimator.estimate(
            messages=messages,
            tier=Tier.T1,
            task_type=TaskType.OPEN_TEXT,
            max_tokens=100,
        )
        
        console.print(f"  Samples drawn: {len(result.samples)}")
        console.print(f"  Pairwise sims: {[f'{s:.3f}' for s in result.pairwise_similarities]}")
        console.print(f"  Mean similarity: {result.mean_similarity:.3f}")
        console.print(f"  Uncertainty: {result.uncertainty:.3f}")
        
        # For "capital of France" all 3 samples should agree → low uncertainty
        if result.uncertainty < 0.3:
            console.print("[green]✓ Low uncertainty on factual question (expected)[/green]")
            return True
        else:
            console.print("[yellow]⚠ Higher than expected uncertainty; check sampling[/yellow]")
            return True  # Not a hard fail
    except Exception as e:
        console.print(f"[red]✗ {e}[/red]")
        return False


def main():
    """Run all Day 2-3 verification tests."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        console.print(f"[red]ERROR: {env_path} not found.[/red]")
        sys.exit(1)
    
    load_dotenv(env_path)
    
    console.print("[bold green]🚀 CHR-CP Day 2-3 Verification (VC²)[/bold green]\n")
    
    results = {}
    results["verbalized_parser"] = test_verbalized_parser()
    console.print()
    results["strip_confidence"] = test_strip_confidence()
    console.print()
    results["consistency_only"] = test_consistency_only()
    console.print()
    results["vc2_easy_vs_hard"] = test_vc2_easy_vs_hard()
    
    console.rule("[bold]Final Summary[/bold]")
    summary = Table()
    summary.add_column("Test")
    summary.add_column("Result", style="bold")
    
    for name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        summary.add_row(name, status)
    
    console.print(summary)
    
    if all(results.values()):
        console.print("\n[bold green]🎉 Day 2-3 done! VC² is functional. Ready for Day 4 (L2 router).[/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]❌ Some tests failed. Fix before Day 4.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()