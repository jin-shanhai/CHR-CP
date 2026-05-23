"""For each task with ESCALATE, did escalation actually save the answer?"""

import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from rich.console import Console

console = Console()
records = []
with open(PROJECT_ROOT / "results/chrcp/math.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

escalate_samples = [r for r in records if r.get("num_escalate", 0) > 0]
console.print(f"[bold]Samples with ESCALATE: {len(escalate_samples)}[/bold]")

correct_after_escalate = sum(1 for r in escalate_samples if r.get("correct"))
console.print(f"  Correct after ESCALATE: {correct_after_escalate}/{len(escalate_samples)}")
console.print(f"  Save rate: {correct_after_escalate/max(1,len(escalate_samples))*100:.1f}%")

# Distribution by final_tier
from collections import Counter
final_dist = Counter(r["final_tier"] for r in escalate_samples)
console.print(f"\nFinal tier of escalated tasks: {dict(final_dist)}")

# Per-tier success rate
for tier in ["T2", "T3", "T4"]:
    tier_samples = [r for r in escalate_samples if r["final_tier"] == tier]
    if tier_samples:
        correct = sum(1 for r in tier_samples if r["correct"])
        console.print(f"  {tier}: {correct}/{len(tier_samples)} ({correct/len(tier_samples)*100:.1f}%)")