"""MMLU-Pro benchmark loader."""

from __future__ import annotations
import re
from typing import Optional, Iterator

from loguru import logger

from chr_cp.benchmarks.base import Benchmark, BenchmarkSample, EvaluationResult
from chr_cp.benchmarks.answer_verify import AnswerVerifier, extract_boxed


MMLU_PRO_SYSTEM_PROMPT = """You are an expert in answering multiple-choice questions.
Read the question carefully and select the best answer from the options.
Respond with the letter (A-J) inside \\boxed{}, e.g., \\boxed{C}.
End with: <confidence>X.X/10</confidence>"""


class MMLUProBenchmark(Benchmark):
    name = "mmlu_pro"

    def __init__(self, n_samples: int = 200, subjects: Optional[list[str]] = None,
                 judge_pool=None):
        self.n_samples = n_samples
        self.subjects = subjects or ["math", "physics", "biology", "philosophy"]
        self.verifier = AnswerVerifier(judge_pool=judge_pool)

    def load(self, n_samples: Optional[int] = None) -> list[BenchmarkSample]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        n = n_samples if n_samples is not None else self.n_samples
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

        per_subject = max(1, n // len(self.subjects))
        selected = []
        for subj in self.subjects:
            subj_items = [x for x in ds if x["category"] == subj][:per_subject]
            selected.extend(subj_items)

        all_selected = selected[:n]
        samples = []
        for i, item in enumerate(all_selected):
            options_text = "\n".join(
                f"{chr(65+j)}. {opt}" for j, opt in enumerate(item["options"])
            )
            prompt = f"{item['question']}\n\nOptions:\n{options_text}"
            samples.append(BenchmarkSample(
                sample_id=f"mmlupro_{i:04d}",
                benchmark="mmlu_pro",
                prompt=prompt,
                reference=chr(65 + item["answer_index"]),
                difficulty="",
                category=item.get("category", ""),
                extra={"question_id": item.get("question_id", "")},
            ))

        logger.info(f"MMLU-Pro loaded {len(samples)} samples")
        return samples

    def evaluate(self, sample: BenchmarkSample, prediction: str) -> EvaluationResult:
        if sample.reference is None:
            return EvaluationResult(correct=False, score=0.0, error="reference is None")

        pred_answer = extract_boxed(prediction)
        if pred_answer is None:
            tail = prediction[-200:].upper()
            letters = re.findall(r"\b([A-J])\b", tail)
            pred_answer = letters[-1] if letters else None

        if pred_answer is None:
            return EvaluationResult(correct=False, score=0.0,
                                    error="no answer extracted",
                                    metadata={"judge_layer": "extract_fail"})

        result = self.verifier.verify(pred=pred_answer, ref=str(sample.reference))
        return EvaluationResult(
            correct=result.correct, score=1.0 if result.correct else 0.0,
            extracted_answer=pred_answer,
            metadata={"judge_layer": result.judge_layer},
        )
