"""Diagnose: U distribution on GSM8K problems."""

import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

records = []
with open(PROJECT_ROOT / "results/chrcp/gsm8k.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

# 提取所有 decision 中的 U 值
all_us = []
all_uvs = []
all_ucs = []
for r in records:
    for d in r.get("decisions", []):
        all_us.append(d["uncertainty"])
        all_uvs.append(d["u_verbalized"])
        all_ucs.append(d["u_consistency"])

import statistics
print(f"Total decisions: {len(all_us)}")
print(f"\nU (combined):")
print(f"  min={min(all_us):.4f}  max={max(all_us):.4f}  mean={statistics.mean(all_us):.4f}")
print(f"  >0.15: {sum(1 for u in all_us if u > 0.15)}")
print(f"  >0.30: {sum(1 for u in all_us if u > 0.30)}")
print(f"  >0.45: {sum(1 for u in all_us if u > 0.45)}")
print(f"  ==0:   {sum(1 for u in all_us if u == 0.0)}")

print(f"\nU_verbalized:")
print(f"  min={min(all_uvs):.4f}  max={max(all_uvs):.4f}  mean={statistics.mean(all_uvs):.4f}")
print(f"  ==0.0: {sum(1 for u in all_uvs if u == 0.0)}")
print(f"  ==0.5: {sum(1 for u in all_uvs if abs(u-0.5)<0.01)}  (parse failed)")

print(f"\nU_consistency:")
print(f"  min={min(all_ucs):.4f}  max={max(all_ucs):.4f}  mean={statistics.mean(all_ucs):.4f}")
print(f"  ==0.0: {sum(1 for u in all_ucs if u == 0.0)}")