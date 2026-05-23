"""Quick view of Phase 1a screening status (for monitoring during grid run)."""

from __future__ import annotations
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

console = Console()

GRID_META = {
    "A1": (3, 0.10, 0.45, 0.8),  "A2": (3, 0.10, 0.50, 0.8),  "A3": (3, 0.10, 0.55, 0.8),
    "B1": (3, 0.10, 0.45, 0.7),  "B2": (3, 0.10, 0.50, 0.7),  "C1": (3, 0.10, 0.45, 0.9),
    "D1": (5, 0.10, 0.40, 0.8),  "D2": (5, 0.10, 0.45, 0.8),  "D3": (5, 0.10, 0.50, 0.8),
    "D4": (5, 0.10, 0.55, 0.8),  "E1": (5, 0.10, 0.45, 0.7),  "E2": (5, 0.10, 0.50, 0.7),
    "F1": (5, 0.10, 0.45, 0.9),  "F2": (5, 0.10, 0.50, 0.9),  "F3": (5, 0.10, 0.55, 0.9),
}

console.rule("[bold cyan]Phase 1a Status[/bold cyan]")
table = Table()
table.add_column("Cfg")
table.add_column("K", justify="center")
table.add_column("τ_high", justify="right")
table.add_column("α", justify="right")
table.add_column("Samples", justify="right")
table.add_column("Acc", justify="right")
table.add_column("Cost/sample", justify="right")
table.add_column("Route %", justify="right")
table.add_column("Verdict", style="bold")

for name, (k, tau_low, tau_high, alpha) in GRID_META.items():
    path = PROJECT_ROOT / "results" / "chrcp" / f"math_grid_{name}_p1a.jsonl"
    if not path.exists():
        table.add_row(name, str(k), f"{tau_high:.2f}", f"{alpha:.1f}",
                      "—", "—", "—", "—", "[grey]missing[/grey]")
        continue
    
    with open(path) as f:
        records = [json.loads(line) for line in f]
    
    n = len(records)
    if n == 0:
        continue
    
    n_correct = sum(1 for x in records if x.get("correct"))
    accuracy = n_correct / n
    total_cost = sum(x.get("cost_usd", 0) for x in records)
    cost_per_sample = total_cost / n
    
    n_stay = sum(x.get("num_stay", 0) for x in records)
    n_branch = sum(x.get("num_branch", 0) for x in records)
    n_escalate = sum(x.get("num_escalate", 0) for x in records)
    total = max(1, n_stay + n_branch + n_escalate)
    routing = (n_branch + n_escalate) / total
    
    verdict_color = "green"
    verdict_text = "✓ KEEP"
    if accuracy < 0.70:
        verdict_color = "red"
        verdict_text = "✗ low acc"
    elif routing < 0.05:
        verdict_color = "red"
        verdict_text = "✗ no routing"
    elif cost_per_sample > 0.15:
        verdict_color = "red"
        verdict_text = "✗ too expensive"
    
    table.add_row(
        name, str(k), f"{tau_high:.2f}", f"{alpha:.1f}",
        str(n),
        f"{accuracy*100:.1f}%",
        f"${cost_per_sample:.4f}",
        f"{routing*100:.0f}%",
        f"[{verdict_color}]{verdict_text}[/{verdict_color}]",
    )

console.print(table)