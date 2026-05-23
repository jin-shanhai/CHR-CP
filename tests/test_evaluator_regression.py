"""Run all regression cases against the current verifier.

Usage:
    python -m tests.test_evaluator_regression
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from chr_cp.clients import ClientPool
from chr_cp.benchmarks.answer_verify import AnswerVerifier
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    pool = ClientPool.from_config("configs/models.yaml")
    verifier = AnswerVerifier(judge_pool=pool)

    cases_path = PROJECT_ROOT / "tests" / "evaluator_regression_cases.jsonl"
    if not cases_path.exists():
        console.print(f"[red]Not found: {cases_path}[/red]")
        sys.exit(1)

    cases = []
    with open(cases_path) as f:
        for line in f:
            cases.append(json.loads(line))

    console.print(f"\n[bold]Running {len(cases)} regression cases...[/bold]\n")

    table = Table(title="Regression Test Results")
    table.add_column("Source")
    table.add_column("Pred")
    table.add_column("Ref")
    table.add_column("Expected", justify="center")
    table.add_column("Got", justify="center")
    table.add_column("Layer")
    table.add_column("Status")

    pass_count = 0
    fail_count = 0

    for case in cases:
        result = verifier.verify(
            pred=case["pred"],
            ref=case["ref"],
            problem=case.get("problem", ""),
        )
        expected = case["expected_correct"]
        got = result.correct
        passed = expected == got

        if passed:
            pass_count += 1
            status = "[green]PASS[/green]"
        else:
            fail_count += 1
            status = "[red]FAIL[/red]"

        table.add_row(
            case.get("source", "?")[:15],
            case["pred"][:25],
            case["ref"][:25],
            "T" if expected else "F",
            "T" if got else "F",
            result.judge_layer,
            status,
        )

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {pass_count}/{len(cases)} passed, {fail_count} failed\n")

    if fail_count > 0:
        console.print("[red]REGRESSION DETECTED. Fix the verifier first.[/red]")
        sys.exit(1)
    else:
        console.print("[green]All regression cases passed.[/green]")
        sys.exit(0)


if __name__ == "__main__":
    main()
