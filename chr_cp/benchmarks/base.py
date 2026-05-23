"""Benchmark abstract base class + unified data structures.

Each benchmark subclass implements:
- load(n_samples): produce a list of BenchmarkSample
- evaluate(sample, prediction): produce EvaluationResult

The runner doesn't need to know benchmark-specific details.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path
import json


@dataclass
class BenchmarkSample:
    """Single test sample, benchmark-agnostic."""
    
    sample_id: str               # unique within benchmark, e.g., "gsm8k_42"
    benchmark: str               # benchmark name
    
    # Task content
    prompt: str                  # the task description fed to CHR-CP
    
    # Ground truth (used by evaluate(), not given to CHR-CP)
    reference: Any               # could be a number / letter / code / text
    
    # Optional metadata for analysis
    difficulty: Optional[str] = None      # e.g., "easy"/"hard" or MATH level 1-5
    category: Optional[str] = None        # e.g., subject for MMLU
    extra: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Sample({self.sample_id}, ref={self.reference!r})"


@dataclass
class EvaluationResult:
    """Result of evaluating a model's prediction against the reference."""
    
    correct: bool
    score: float = 0.0           # 1.0 if correct, can be partial (e.g., ROUGE)
    extracted_answer: Optional[Any] = None  # what we parsed from prediction
    error: Optional[str] = None  # parse error / runtime error if any
    metadata: dict = field(default_factory=dict)


class Benchmark(ABC):
    """Abstract benchmark base class."""
    
    name: str = "base"
    
    @abstractmethod
    def load(self, n_samples: Optional[int] = None) -> list[BenchmarkSample]:
        """Load (a subset of) test samples.
        
        Args:
            n_samples: If given, return only the first n samples.
        """
        ...
    
    @abstractmethod
    def evaluate(
        self,
        sample: BenchmarkSample,
        prediction: str,
    ) -> EvaluationResult:
        """Evaluate a single prediction against the sample's reference."""
        ...
    
    def evaluate_batch(
        self,
        samples: list[BenchmarkSample],
        predictions: list[str],
    ) -> list[EvaluationResult]:
        """Convenience batch evaluator."""
        if len(samples) != len(predictions):
            raise ValueError(
                f"Mismatched lengths: {len(samples)} samples vs "
                f"{len(predictions)} predictions"
            )
        return [
            self.evaluate(s, p)
            for s, p in zip(samples, predictions)
        ]
    
    @staticmethod
    def cache_dir() -> Path:
        """Default location for cached benchmark data."""
        # Hard-coded relative to project root
        # ~/mas_workspace/code/mas_chrcp/cache/benchmarks
        from pathlib import Path
        root = Path.home() / "mas_workspace" / "code" / "mas_chrcp"
        cache = root / "cache" / "benchmarks"
        cache.mkdir(parents=True, exist_ok=True)
        return cache