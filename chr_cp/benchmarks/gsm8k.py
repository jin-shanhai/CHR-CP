"""GSM8K math benchmark loader and evaluator.

Source: https://github.com/openai/grade-school-math
Test set: 1319 problems, exact-match evaluation on the final number.

Loading strategy:
1. Try huggingface datasets (`gsm8k`, split='test')
2. Fall back to a local cache file if downloaded once
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
from chr_cp.utils.text_sim import extract_final_number
from chr_cp.benchmarks.answer_verify import AnswerVerifier, extract_boxed


# GSM8K reference answers are formatted as "...#### NUMBER"
GSM8K_ANSWER_PATTERN = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


class GSM8KBenchmark(Benchmark):
    """GSM8K grade-school math benchmark."""

    name = "gsm8k"

    def __init__(self, cache_path: Optional[Path] = None, judge_pool=None):
        self.cache_path = cache_path or (self.cache_dir() / "gsm8k_test.jsonl")
        self.verifier = AnswerVerifier(judge_pool=judge_pool)
    
    def load(self, n_samples: Optional[int] = None) -> list[BenchmarkSample]:
        """Load GSM8K test split."""
        if not self.cache_path.exists():
            self._download_to_cache()
        
        samples: list[BenchmarkSample] = []
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if n_samples is not None and i >= n_samples:
                    break
                row = json.loads(line)
                samples.append(self._row_to_sample(row, i))
        
        logger.info(f"GSM8K loaded {len(samples)} samples from {self.cache_path}")
        return samples
    
    def _download_to_cache(self) -> None:
        """Download GSM8K test split via HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install datasets: pip install datasets"
            )
        
        logger.info("Downloading GSM8K test split via HuggingFace...")
        ds = load_dataset("gsm8k", "main", split="test")
        
        with open(self.cache_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        logger.info(f"GSM8K cached at {self.cache_path}")
    
    def _row_to_sample(self, row: dict, idx: int) -> BenchmarkSample:
        """Convert raw GSM8K row to BenchmarkSample."""
        question = row["question"]
        full_answer = row["answer"]  # contains "#### NUMBER" at the end
        
        # Extract the final numeric answer
        m = GSM8K_ANSWER_PATTERN.search(full_answer)
        if m is None:
            logger.warning(f"GSM8K row {idx}: cannot extract reference number")
            ref_number = None
        else:
            ref_number = float(m.group(1))
        
        # Build the prompt CHR-CP will see
        prompt = (
            f"Solve the following math word problem. Show your reasoning step by step, "
            f"then state the final numeric answer.\n\n"
            f"Problem: {question}\n\n"
            f"End your response with: \\boxed{{ANSWER}} where ANSWER is the final number."
        )
        
        return BenchmarkSample(
            sample_id=f"gsm8k_{idx:04d}",
            benchmark=self.name,
            prompt=prompt,
            reference=ref_number,
            extra={
                "raw_answer": full_answer,
                "question": question,
            },
        )
    
    def evaluate(
        self,
        sample: BenchmarkSample,
        prediction: str,
    ) -> EvaluationResult:
        """Evaluate by extracting the final number and comparing."""
        if sample.reference is None:
            return EvaluationResult(
                correct=False, score=0.0,
                error="reference is None (parse error in dataset)",
            )
        
        pred_answer = extract_boxed(prediction)
        if pred_answer is None:
            pred_number = extract_final_number(prediction)
            pred_answer = str(pred_number) if pred_number is not None else None

        if pred_answer is None:
            return EvaluationResult(
                correct=False, score=0.0,
                error="failed to extract answer",
                metadata={"judge_layer": "extract_fail"},
            )

        result = self.verifier.verify(
            pred=pred_answer, ref=str(sample.reference),
            problem=sample.prompt,
        )
        return EvaluationResult(
            correct=result.correct,
            score=1.0 if result.correct else 0.0,
            extracted_answer=pred_answer,
            metadata={
                "reference": sample.reference,
                "judge_layer": result.judge_layer,
                "answer_type": result.answer_type.value if result.answer_type else None,
            },
        )