"""Diagnose HumanEval 50-sample run."""

import sys
import json
from pathlib import Path
from collections import Counter
import statistics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

console = Console()

# === Load humaneval.jsonl ===
records = []
path = PROJECT_ROOT / "results" / "chrcp" / "humaneval.jsonl"
if not path.exists():
    console.print(f"[red]File not found: {path}[/red]")
    sys.exit(1)

with open(path) as f:
    for line in f:
        records.append(json.loads(line))

n = len(records)
console.rule(f"[bold cyan]HumanEval Diagnosis: {n} samples[/bold cyan]")

# === Basic stats ===
n_correct = sum(1 for r in records if r.get("correct"))
total_cost = sum(r.get("cost_usd", 0.0) for r in records)
avg_latency = sum(r.get("latency_seconds", 0.0) for r in records) / max(n, 1)

console.print(f"\n[bold]Overall:[/bold]")
console.print(f"  Accuracy: {n_correct}/{n} = {n_correct/max(n,1)*100:.2f}%")
console.print(f"  Total cost: ${total_cost:.4f}")
console.print(f"  Avg latency: {avg_latency:.2f}s")

# === Action distribution ===
total_stay = sum(r.get('num_stay', 0) for r in records)
total_branch = sum(r.get('num_branch', 0) for r in records)
total_escalate = sum(r.get('num_escalate', 0) for r in records)
total_actions = max(1, total_stay + total_branch + total_escalate)

console.print(f"\n[bold]Action distribution:[/bold]")
console.print(f"  STAY:     {total_stay} ({total_stay/total_actions*100:.1f}%)")
console.print(f"  BRANCH:   {total_branch} ({total_branch/total_actions*100:.1f}%)")
console.print(f"  ESCALATE: {total_escalate} ({total_escalate/total_actions*100:.1f}%)")

# === L3 mechanism triggers ===
total_distill = sum(r.get('distillations', 0) for r in records)
total_warm = sum(r.get('warmups', 0) for r in records)
console.print(f"\n[bold]L3 mechanism triggers:[/bold]")
console.print(f"  Distillations (M2): {total_distill}")
console.print(f"  Warmups (M3):       {total_warm}")

# === U distribution ===
all_us = []
all_uvs = []
all_ucs = []
for r in records:
    for d in r.get("decisions", []):
        all_us.append(d["uncertainty"])
        all_uvs.append(d["u_verbalized"])
        all_ucs.append(d["u_consistency"])

console.print(f"\n[bold]Total decisions recorded: {len(all_us)}[/bold]")

if all_us:
    console.print(f"\n[bold]U (combined):[/bold]")
    console.print(f"  min={min(all_us):.4f}  max={max(all_us):.4f}  mean={statistics.mean(all_us):.4f}")
    console.print(f"  >0.15:  {sum(1 for u in all_us if u > 0.15)}")
    console.print(f"  >0.30:  {sum(1 for u in all_us if u > 0.30)}")
    console.print(f"  >0.45:  {sum(1 for u in all_us if u > 0.45)}")
    console.print(f"  ==0.0:  {sum(1 for u in all_us if u == 0.0)}")

    console.print(f"\n[bold]U_verbalized:[/bold]")
    console.print(f"  min={min(all_uvs):.4f}  max={max(all_uvs):.4f}  mean={statistics.mean(all_uvs):.4f}")
    console.print(f"  ==0.0:  {sum(1 for u in all_uvs if u == 0.0)}")
    console.print(f"  ==0.5:  {sum(1 for u in all_uvs if abs(u-0.5)<0.01)} (parse failed)")

    console.print(f"\n[bold]U_consistency:[/bold]")
    console.print(f"  min={min(all_ucs):.4f}  max={max(all_ucs):.4f}  mean={statistics.mean(all_ucs):.4f}")
    console.print(f"  ==0.0:  {sum(1 for u in all_ucs if u == 0.0)}")

# === Final tier distribution ===
final_tiers = Counter(r.get('final_tier', '?') for r in records)
console.print(f"\n[bold]Final tier distribution:[/bold] {dict(final_tiers)}")

# === Cache factor ===
cache_factors = []
for r in records:
    if 'avg_cache_factor' in r:
        cache_factors.append(r['avg_cache_factor'])

if cache_factors:
    console.print(f"\n[bold]CA²R avg_cache_factor:[/bold]")
    if len(cache_factors) >= 10:
        first10 = sum(cache_factors[:10]) / 10
        last10 = sum(cache_factors[-10:]) / 10
        console.print(f"  First 10: avg={first10:.4f}")
        console.print(f"  Last 10:  avg={last10:.4f}")
    console.print(f"  Overall:  avg={statistics.mean(cache_factors):.4f}")

# === Error types (parse failures vs runtime errors) ===
errors = [r.get('eval_error') for r in records if not r.get('correct') and r.get('eval_error')]
error_types = Counter()
for e in errors:
    if e is None:
        continue
    if "no python code block" in e:
        error_types["no_code_block"] += 1
    elif "timeout" in e:
        error_types["timeout"] += 1
    elif "non-zero exit" in e:
        error_types["test_failed"] += 1
    else:
        error_types["other"] += 1

if error_types:
    console.print(f"\n[bold]Failure modes ({sum(error_types.values())} failures):[/bold]")
    for err_type, count in error_types.most_common():
        console.print(f"  {err_type}: {count}")

# === Sample-level inspection: 5 examples with non-trivial decisions ===
console.print(f"\n[bold]Sample-level inspection (interesting cases):[/bold]")
interesting = []
for r in records:
    n_b = r.get('num_branch', 0)
    n_e = r.get('num_escalate', 0)
    if n_b > 0 or n_e > 0:
        interesting.append(r)

console.print(f"  Found {len(interesting)} samples with BRANCH or ESCALATE")
for r in interesting[:5]:
    console.print(
        f"\n  {r['sample_id']}: "
        f"correct={r['correct']}, "
        f"actions=(S{r.get('num_stay',0)}/B{r.get('num_branch',0)}/E{r.get('num_escalate',0)}), "
        f"final_tier={r.get('final_tier')}, "
        f"cost=${r.get('cost_usd',0):.4f}"
    )
    for d in r.get('decisions', []):
        console.print(
            f"    → {d['action']} (U={d['uncertainty']:.3f}, "
            f"τ_low={d.get('tau_low','?'):.3f}, "
            f"τ_high={d.get('tau_high','?'):.3f})"
        )