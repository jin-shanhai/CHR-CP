"""MATH benchmark (Hendrycks et al.) - difficulty-stratified math problems.

Source: https://github.com/hendrycks/math
Levels 1-5, where Level 5 is competition-level (AMC/AIME).

We use the 'MATH-500' subset (500 representative problems) for tractability.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import json
import re

from loguru import logger

from chr_cp.benchmarks.base import (
    Benchmark,
    BenchmarkSample,
    EvaluationResult,
)
from chr_cp.utils.text_sim import (
    extract_final_number, numeric_match, normalize_latex_answer,
)


# MATH answers are in \boxed{...}
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")


class MATHBenchmark(Benchmark):
    """MATH benchmark with difficulty levels 1-5."""
    
    name = "math"
    
    def __init__(
        self,
        cache_path: Optional[Path] = None,
        levels: Optional[list[int]] = None,
    ):
        self.cache_path = cache_path or (self.cache_dir() / "math500.jsonl")
        # Filter to specific difficulty levels (None = all)
        self.levels = levels
    
    def load(self, n_samples: Optional[int] = None) -> list[BenchmarkSample]:
        if not self.cache_path.exists():
            self._download_to_cache()
        
        all_rows = []
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                # Filter by difficulty level if specified
                if self.levels is not None:
                    level_str = row.get("level", "")
                    # MATH uses "Level X" format
                    m = re.search(r"Level\s*(\d+)", level_str)
                    if m:
                        level_num = int(m.group(1))
                        if level_num not in self.levels:
                            continue
                all_rows.append(row)
        
        if n_samples is not None:
            all_rows = all_rows[:n_samples]
        
        samples = [self._row_to_sample(r, i) for i, r in enumerate(all_rows)]
        logger.info(f"MATH loaded {len(samples)} samples (levels={self.levels})")
        return samples
    
    def _download_to_cache(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")
        
        logger.info("Downloading MATH-500 (HuggingFaceH4/MATH-500)...")
        try:
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        except Exception:
            # Fallback to original MATH dataset if MATH-500 unavailable
            logger.warning("MATH-500 unavailable, falling back to lighteval/MATH")
            ds = load_dataset("lighteval/MATH", "all", split="test")
        
        with open(self.cache_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"MATH cached at {self.cache_path}")
    
    def _row_to_sample(self, row: dict, idx: int) -> BenchmarkSample:
        problem = row.get("problem", row.get("question", ""))
        solution = row.get("solution", "")
        level = row.get("level", "")
        subject = row.get("subject", row.get("type", "unknown"))
        
        # Extract reference answer from solution
        ref_answer = row.get("answer", None)
        if ref_answer is None:
            # Try to extract from solution
            m = BOXED_PATTERN.search(solution)
            if m:
                ref_answer = m.group(1).strip()
        
        prompt = (
            f"Solve the following math problem. Show your reasoning step by step, "
            f"then state the final answer in \\boxed{{}}.\n\n"
            f"Problem: {problem}\n\n"
            f"End your response with: \\boxed{{ANSWER}}"
        )
        
        return BenchmarkSample(
            sample_id=f"math_{idx:04d}",
            benchmark=self.name,
            prompt=prompt,
            reference=ref_answer,
            difficulty=level,
            category=subject,
            extra={"problem": problem, "solution": solution},
        )
    
    def evaluate(
        self,
        sample: BenchmarkSample,
        prediction: str,
    ) -> EvaluationResult:
        if sample.reference is None:
            return EvaluationResult(
                correct=False, score=0.0,
                error="reference is None",
            )
        
        # Try to extract \boxed{} answer first
        m = BOXED_PATTERN.search(prediction)
        if m:
            pred_answer = m.group(1).strip()
        else:
            # Fallback: extract last number
            num = extract_final_number(prediction)
            pred_answer = str(num) if num is not None else None
        
        if pred_answer is None:
            return EvaluationResult(
                correct=False, score=0.0,
                error="failed to extract answer",
            )
        
        ok = self._answers_match(pred_answer, str(sample.reference))
        return EvaluationResult(
            correct=ok,
            score=1.0 if ok else 0.0,
            extracted_answer=pred_answer,
            metadata={
                "reference": sample.reference,
                "level": sample.difficulty,
                "subject": sample.category,
            },
        )
    
    @staticmethod
    def _answers_match(pred: str, ref: str) -> bool:
        """Match MATH answers, tolerating common formatting variations."""
        pred_norm = pred.strip().replace(" ", "")
        ref_norm = ref.strip().replace(" ", "")
        
        if pred_norm == ref_norm:
            return True
        
        # Try numeric comparison
        try:
            pred_num = float(pred_norm)
            ref_num = float(ref_norm)
            return abs(pred_num - ref_num) < 1e-6
        except ValueError:
            pass
        
        # Normalize LaTeX and compare
        return normalize_latex_answer(pred_norm) == normalize_latex_answer(ref_norm)
    



