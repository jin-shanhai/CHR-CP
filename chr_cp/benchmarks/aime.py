"""AIME benchmark loader (2024 + 2025)."""

from __future__ import annotations
import re
from typing import Optional

from loguru import logger

from chr_cp.benchmarks.base import Benchmark, BenchmarkSample, EvaluationResult
from chr_cp.benchmarks.answer_verify import AnswerVerifier, extract_boxed


AIME_SYSTEM_PROMPT = """You are an expert mathematician solving an AIME problem.
AIME answers are always integers from 0 to 999. Reason step by step,
and provide your final integer answer in \\boxed{N}.
End with: <confidence>X.X/10</confidence>"""


class AIMEBenchmark(Benchmark):
    name = "aime"

    def __init__(self, years: tuple[str, ...] = ("2024", "2025"), n_samples: int = 60,
                 judge_pool=None):
        self.years = years
        self.n_samples = n_samples
        self.verifier = AnswerVerifier(judge_pool=judge_pool)

    def load(self, n_samples: Optional[int] = None) -> list[BenchmarkSample]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        n = n_samples if n_samples is not None else self.n_samples
        all_items = []
        for year in self.years:
            try:
                ds_name = (
                    "Maxwell-Jia/AIME_2024" if year == "2024"
                    else "yentinglin/aime_2025"
                )
                ds = load_dataset(ds_name, split="train")
                for item in ds:
                    all_items.append({
                        "problem": item.get("Problem") or item.get("problem", ""),
                        "answer": str(item.get("Answer") or item.get("answer", "")),
                        "year": year,
                    })
            except Exception as e:
                logger.warning(f"AIME {year} load failed: {e}")

        all_items = all_items[:n]
        samples = []
        for i, item in enumerate(all_items):
            samples.append(BenchmarkSample(
                sample_id=f"aime_{item['year']}_{i:04d}",
                benchmark="aime",
                prompt=item["problem"],
                reference=item["answer"],
                difficulty="extreme",
                category="math_olympiad",
                extra={"year": item["year"]},
            ))

        logger.info(f"AIME loaded {len(samples)} samples (years={self.years})")
        return samples

    def evaluate(self, sample: BenchmarkSample, prediction: str) -> EvaluationResult:
        if sample.reference is None:
            return EvaluationResult(correct=False, score=0.0, error="reference is None")

        pred_answer = extract_boxed(prediction)
        if pred_answer is None:
            tail = prediction[-200:]
            numbers = re.findall(r"\b(\d{1,3})\b", tail)
            pred_answer = numbers[-1] if numbers else None

        if pred_answer is None:
            return EvaluationResult(correct=False, score=0.0,
                                    error="no integer extracted",
                                    metadata={"judge_layer": "extract_fail"})

        result = self.verifier.verify(pred=pred_answer, ref=str(sample.reference))
        return EvaluationResult(
            correct=result.correct, score=1.0 if result.correct else 0.0,
            extracted_answer=pred_answer,
            metadata={"judge_layer": result.judge_layer},
        )
