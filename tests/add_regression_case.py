"""Add a new regression case after discovering a false negative.

Usage:
    python -m tests.add_regression_case \
        --pred "5" --ref "x=5" --expected_correct true \
        --source "math_0023" --note "variable prefix"
"""

import argparse
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "tests" / "evaluator_regression_cases.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--expected_correct", required=True,
                        type=lambda x: x.lower() == "true")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    case = {
        "pred": args.pred,
        "ref": args.ref,
        "expected_correct": args.expected_correct,
        "source": args.source,
        "note": args.note,
    }

    with open(CASES_PATH, "a") as f:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")

    lines = sum(1 for _ in open(CASES_PATH))
    print(f"Added: {case}")
    print(f"Total cases: {lines}")

    print("\nRunning regression tests...")
    subprocess.run(["python", "-m", "tests.test_evaluator_regression"])


if __name__ == "__main__":
    main()
