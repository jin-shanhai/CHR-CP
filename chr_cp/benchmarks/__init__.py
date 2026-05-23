"""Benchmark loaders + evaluators for CHR-CP experiments."""

from chr_cp.benchmarks.base import (
    Benchmark,
    BenchmarkSample,
    EvaluationResult,
)
from chr_cp.benchmarks.math500 import MATHBenchmark
from chr_cp.benchmarks.gsm8k import GSM8KBenchmark
from chr_cp.benchmarks.humaneval import HumanEvalBenchmark
from chr_cp.benchmarks.mmlu import MMLUBenchmark
from chr_cp.benchmarks.mmlu_pro import MMLUProBenchmark
from chr_cp.benchmarks.gpqa import GPQABenchmark
from chr_cp.benchmarks.aime import AIMEBenchmark

__all__ = [
    "Benchmark", "BenchmarkSample", "EvaluationResult",
    "MATHBenchmark", "GSM8KBenchmark", "HumanEvalBenchmark",
    "MMLUBenchmark", "MMLUProBenchmark", "GPQABenchmark", "AIMEBenchmark",
]

# Convenience registry for get_benchmark()
BENCHMARK_CLASSES = {
    "math": MATHBenchmark,
    "gsm8k": GSM8KBenchmark,
    "humaneval": HumanEvalBenchmark,
    "mmlu": MMLUBenchmark,
    "mmlu_pro": MMLUProBenchmark,
    "gpqa": GPQABenchmark,
    "aime": AIMEBenchmark,
}


def get_benchmark(name: str, **kwargs):
    """Get a benchmark instance by name."""
    if name not in BENCHMARK_CLASSES:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_CLASSES.keys())}")
    return BENCHMARK_CLASSES[name](**kwargs)