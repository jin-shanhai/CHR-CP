"""GPQA Diamond benchmark loader.

GPQA is a gated dataset on HuggingFace. You must:
1. Run `huggingface-cli login` with a token from https://huggingface.co/settings/tokens
2. Accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa
"""

from __future__ import annotations
import json
import random
import re
from pathlib import Path
from typing import Optional

from loguru import logger

from chr_cp.benchmarks.base import Benchmark, BenchmarkSample, EvaluationResult
from chr_cp.benchmarks.answer_verify import AnswerVerifier, extract_boxed


GPQA_SYSTEM_PROMPT = """You are an expert in answering graduate-level scientific questions.
This question requires deep knowledge in physics, chemistry, or biology.
Read carefully, reason step by step, and select the best answer.
Respond with the letter (A-D) inside \\boxed{}, e.g., \\boxed{B}.
End with: <confidence>X.X/10</confidence>"""


class GPQABenchmark(Benchmark):
    name = "gpqa"

    def __init__(self, split: str = "diamond", n_samples: int = 100,
                 cache_path: Optional[Path] = None, judge_pool=None):
        self.split = split
        self.n_samples = n_samples
        self.cache_path = cache_path or (self.cache_dir() / "gpqa_diamond.jsonl")
        self.verifier = AnswerVerifier(judge_pool=judge_pool)

    def load(self, n_samples: Optional[int] = None) -> list[BenchmarkSample]:
        if not self.cache_path.exists():
            self._download_to_cache()

        n = n_samples if n_samples is not None else self.n_samples
        rows = []
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

        rows = rows[:n]
        samples = []
        for i, item in enumerate(rows):
            options = [
                item["correct_answer"],
                item["incorrect_1"],
                item["incorrect_2"],
                item["incorrect_3"],
            ]
            rng = random.Random(42 + i)
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [options[j] for j in order]
            correct_letter = chr(65 + order.index(0))

            options_text = "\n".join(
                f"{chr(65+j)}. {opt}" for j, opt in enumerate(shuffled)
            )
            prompt = f"{item['question']}\n\nOptions:\n{options_text}"

            samples.append(BenchmarkSample(
                sample_id=f"gpqa_{i:04d}",
                benchmark="gpqa",
                prompt=prompt,
                reference=correct_letter,
                difficulty="diamond",
                category=item.get("subdomain", "science"),
                extra={"high_level": item.get("high_level_domain", "")},
            ))

        logger.info(f"GPQA loaded {len(samples)} samples ({self.split})")
        return samples

    def _download_to_cache(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        logger.info("Downloading GPQA Diamond (Idavidrein/gpqa)...")
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GPQA: {e}\n\n"
                "GPQA is a gated dataset. Please:\n"
                "  1. Run: huggingface-cli login\n"
                "  2. Accept terms at: https://huggingface.co/datasets/Idavidrein/gpqa\n"
            )

        with open(self.cache_path, "w", encoding="utf-8") as f:
            for item in ds:
                row = {
                    "question": item.get("Question", ""),
                    "correct_answer": item.get("Correct Answer", ""),
                    "incorrect_1": item.get("Incorrect Answer 1", ""),
                    "incorrect_2": item.get("Incorrect Answer 2", ""),
                    "incorrect_3": item.get("Incorrect Answer 3", ""),
                    "subdomain": item.get("Subdomain", ""),
                    "high_level_domain": item.get("High-level domain", ""),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(f"GPQA cached at {self.cache_path}")

    def evaluate(self, sample: BenchmarkSample, prediction: str) -> EvaluationResult:
        if sample.reference is None:
            return EvaluationResult(correct=False, score=0.0, error="reference is None")

        pred_answer = extract_boxed(prediction)
        if pred_answer is not None:
            # Strip full answer text: "\text{D. } (cos(θ/2), sin(θ/2))" → "D"
            m = re.match(r'\\text\{([A-D])\b', pred_answer)
            if m:
                pred_answer = m.group(1)
            else:
                m = re.match(r'([A-D])[.\)\s]', pred_answer)
                if m:
                    pred_answer = m.group(1)
        if pred_answer is None:
            tail = prediction[-200:].upper()
            letters = re.findall(r"\b([A-D])\b", tail)
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
