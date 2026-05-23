"""Diagnose why ESCALATE never reaches T3/T4 in math.jsonl."""

import sys
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console

console = Console()

records = []
with open(PROJECT_ROOT / "results/chrcp/math_no_warmup_fix.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

console.rule("[bold cyan]ESCALATE Path Diagnosis[/bold cyan]")

# Stat 1: Find all decisions with action=ESCALATE
escalate_decisions = []
for r in records:
    for i, d in enumerate(r.get("decisions", [])):
        if d.get("action") == "ESCALATE":
            escalate_decisions.append({
                "sample_id": r["sample_id"],
                "step_idx": i,
                "current_tier": d.get("current_tier"),
                "target_tier": d.get("target_tier"),
                "U": d.get("uncertainty"),
                "final_tier": r.get("final_tier"),
                "n_steps": len(r.get("decisions", [])),
            })

console.print(f"\n[bold]Total ESCALATE decisions: {len(escalate_decisions)}[/bold]\n")

if escalate_decisions:
    console.print("Per-decision analysis:")
    for d in escalate_decisions:
        console.print(
            f"  {d['sample_id']} | step {d['step_idx']+1}/{d['n_steps']}: "
            f"{d['current_tier']} → target={d['target_tier']} | "
            f"final_tier_recorded={d['final_tier']} | U={d['U']:.3f}"
        )
    
    # Aggregate: target_tier distribution
    target_tiers = Counter(d["target_tier"] for d in escalate_decisions)
    console.print(f"\nTarget tier (decided): {dict(target_tiers)}")
    
    # final_tier distribution among ESCALATE samples
    final_tiers = Counter(d["final_tier"] for d in escalate_decisions)
    console.print(f"final_tier (recorded): {dict(final_tiers)}")
    
    # Cross-tabulate
    console.print("\nCross-tab (target vs final):")
    cross = Counter()
    for d in escalate_decisions:
        cross[(d["target_tier"], d["final_tier"])] += 1
    for (target, final), count in sorted(cross.items()):
        marker = "✓" if target == final else "✗ MISMATCH"
        console.print(f"  target={target} → final={final}: {count} {marker}")

# Stat 2: All step responses tier distribution
console.print(f"\n[bold]Per-step tier usage (across all 50 samples):[/bold]\n")
all_tiers = Counter()
for r in records:
    for resp in r.get("step_responses", []):  # if recorded
        all_tiers[resp.get("tier_name", "?")] += 1

if all_tiers:
    console.print(f"Tier invocations: {dict(all_tiers)}")
else:
    console.print("[yellow]No step_responses field in records;[/yellow]")
    console.print("[yellow]checking final_tier and decisions only.[/yellow]")
    
    # Fallback: check final_tier alone
    final_only = Counter(r.get("final_tier") for r in records)
    console.print(f"Final tier (50 samples): {dict(final_only)}")

# Stat 3: Sample-level full trace for the first 3 ESCALATE samples
console.print(f"\n[bold]Detailed traces (first 3 ESCALATE samples):[/bold]\n")
escalate_sample_ids = [d["sample_id"] for d in escalate_decisions[:3]]
for sid in escalate_sample_ids:
    sample = next(r for r in records if r["sample_id"] == sid)
    console.print(f"\n=== {sid} ===")
    console.print(f"  final_tier: {sample.get('final_tier')}")
    console.print(f"  correct: {sample.get('correct')}")
    console.print(f"  decisions:")
    for i, d in enumerate(sample.get("decisions", [])):
        console.print(
            f"    Step {i+1}: {d.get('action')} "
            f"(current={d.get('current_tier')}, target={d.get('target_tier')}, "
            f"U={d.get('uncertainty', 0):.3f})"
        )
    if "step_responses" in sample:
        console.print(f"  step_responses tiers:")
        for j, resp in enumerate(sample["step_responses"]):
            console.print(f"    Resp {j+1}: tier={resp.get('tier_name')}")