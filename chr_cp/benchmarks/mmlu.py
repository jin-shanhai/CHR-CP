"""MMLU multiple-choice knowledge benchmark.

Source: https://github.com/hendrycks/test
We use the test split of `cais/mmlu` (all subjects mixed).
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import json
import random

from loguru import logger

from chr_cp.benchmarks.base import (
    Benchmark,
    BenchmarkSample,
    EvaluationResult,
)
from chr_cp.utils.text_sim import extract_mc_answer
from chr_cp.benchmarks.answer_verify import AnswerVerifier, extract_boxed


CHOICE_LETTERS = ["A", "B", "C", "D"]


class MMLUBenchmark(Benchmark):
    """MMLU benchmark.
    
    By default samples uniformly across all subjects (set seed for reproducibility).
    """
    
    name = "mmlu"
    
    def __init__(
        self,
        cache_path: Optional[Path] = None,
        random_seed: int = 42,
        judge_pool=None,
    ):
        self.cache_path = cache_path or (self.cache_dir() / "mmlu_test.jsonl")
        self.random_seed = random_seed
        self.verifier = AnswerVerifier(judge_pool=judge_pool)
    
    def load(self, n_samples: Optional[int] = None) -> list[BenchmarkSample]:
        if not self.cache_path.exists():
            self._download_to_cache()
        
        # Load all rows then sample
        all_rows = []
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                all_rows.append(json.loads(line))
        
        if n_samples is not None and n_samples < len(all_rows):
            rng = random.Random(self.random_seed)
            all_rows = rng.sample(all_rows, n_samples)
        
        samples = [self._row_to_sample(r, i) for i, r in enumerate(all_rows)]
        logger.info(f"MMLU loaded {len(samples)} samples")
        return samples
    
    def _download_to_cache(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")
        
        logger.info("Downloading MMLU test split (all subjects)...")
        ds = load_dataset("cais/mmlu", "all", split="test")
        with open(self.cache_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"MMLU cached at {self.cache_path}")
    
    def _row_to_sample(self, row: dict, idx: int) -> BenchmarkSample:
        question = row["question"]
        choices = row["choices"]    # list of 4 strings
        answer_idx = row["answer"]   # 0-3
        subject = row.get("subject", "unknown")
        
        ref_letter = CHOICE_LETTERS[answer_idx]
        
        choices_text = "\n".join(
            f"{CHOICE_LETTERS[i]}) {c}" for i, c in enumerate(choices)
        )
        
        prompt = (
            f"Answer the following multiple-choice question. "
            f"Reply with ONLY the letter of the correct answer (A, B, C, or D).\n\n"
            f"Question: {question}\n\n"
            f"{choices_text}\n\n"
            f"Answer:"
        )
        
        return BenchmarkSample(
            sample_id=f"mmlu_{idx:04d}",
            benchmark=self.name,
            prompt=prompt,
            reference=ref_letter,
            category=subject,
            extra={"choices": choices},
        )
    
    def evaluate(
        self,
        sample: BenchmarkSample,
        prediction: str,
    ) -> EvaluationResult:
        if sample.reference is None:
            return EvaluationResult(correct=False, score=0.0, error="reference is None")

        pred_answer = extract_boxed(prediction) or extract_mc_answer(prediction)
        if pred_answer is None:
            return EvaluationResult(
                correct=False, score=0.0,
                error="failed to extract answer",
                metadata={"judge_layer": "extract_fail"},
            )

        result = self.verifier.verify(
            pred=pred_answer, ref=str(sample.reference),
        )
        return EvaluationResult(
            correct=result.correct,
            score=1.0 if result.correct else 0.0,
            extracted_answer=pred_answer,
            metadata={
                "reference": sample.reference, "subject": sample.category,
                "judge_layer": result.judge_layer,
            },
        )