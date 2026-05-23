"""HumanEval code generation benchmark.

Source: https://github.com/openai/human-eval
Test set: 164 problems, Pass@1 evaluation via execution.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import json
import re
import subprocess
import tempfile
import os

from loguru import logger

from chr_cp.benchmarks.base import (
    Benchmark,
    BenchmarkSample,
    EvaluationResult,
)


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class HumanEvalBenchmark(Benchmark):
    """HumanEval code generation benchmark with Pass@1 evaluation."""
    
    name = "humaneval"
    
    def __init__(
        self,
        cache_path: Optional[Path] = None,
        execution_timeout: float = 10.0,
    ):
        self.cache_path = cache_path or (self.cache_dir() / "humaneval.jsonl")
        self.execution_timeout = execution_timeout
    
    def load(self, n_samples: Optional[int] = None) -> list[BenchmarkSample]:
        """Load HumanEval problems."""
        if not self.cache_path.exists():
            self._download_to_cache()
        
        samples: list[BenchmarkSample] = []
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if n_samples is not None and i >= n_samples:
                    break
                row = json.loads(line)
                samples.append(self._row_to_sample(row, i))
        
        logger.info(f"HumanEval loaded {len(samples)} samples")
        return samples
    
    def _download_to_cache(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        logger.info("Downloading HumanEval...")
        ds = load_dataset("openai_humaneval", split="test")
        with open(self.cache_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"HumanEval cached at {self.cache_path}")
    
    def _row_to_sample(self, row: dict, idx: int) -> BenchmarkSample:
        task_id = row["task_id"]
        prompt_code = row["prompt"]   # function signature + docstring
        canonical_solution = row["canonical_solution"]
        test_code = row["test"]
        entry_point = row["entry_point"]
        
        prompt = (
            f"Complete the following Python function. Provide the complete function "
            f"definition (including the signature) in a single ```python ... ``` block.\n\n"
            f"```python\n{prompt_code}```\n\n"
            f"Provide your implementation:"
        )
        
        return BenchmarkSample(
            sample_id=f"humaneval_{idx:03d}",
            benchmark=self.name,
            prompt=prompt,
            reference=None,  # not needed; we evaluate via execution
            extra={
                "task_id": task_id,
                "prompt_code": prompt_code,
                "canonical_solution": canonical_solution,
                "test_code": test_code,
                "entry_point": entry_point,
            },
        )
    
    def evaluate(
        self,
        sample: BenchmarkSample,
        prediction: str,
    ) -> EvaluationResult:
        """Pass@1: execute the prediction against the test cases."""
        # Extract code block
        code = self._extract_code(prediction)
        if not code:
            return EvaluationResult(
                correct=False, score=0.0,
                error="no python code block found in prediction",
            )
        
        # Build the execution program: the function definition + test cases
        prompt_code = sample.extra["prompt_code"]
        test_code = sample.extra["test_code"]
        entry_point = sample.extra["entry_point"]
        
        # Decide what to execute:
        # - if `code` already includes "def {entry_point}(", use it directly
        # - else prepend prompt_code (the signature + docstring) and let the
        #   model's implementation be the body
        if f"def {entry_point}(" in code:
            program = code
        else:
            program = prompt_code + "\n" + code
        
        full = program + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
        
        # Execute
        passed, error = self._safe_execute(full)
        return EvaluationResult(
            correct=passed,
            score=1.0 if passed else 0.0,
            extracted_answer=code[:200],
            error=error,
            metadata={"task_id": sample.extra["task_id"]},
        )
    
    def _extract_code(self, prediction: str) -> Optional[str]:
        """Extract Python code from the prediction with multiple fallback strategies.
        
        Tried in order:
        1. ```python ... ``` fenced block
        2. ``` ... ``` (any language) fenced block
        3. Full prediction if it contains 'def ' (likely raw code)
        4. Indented block following the function signature (CHR-CP common output)
        5. Strip <confidence>...</confidence> tags and try again
        """
        if not prediction:
            return None
        
        text = prediction
        
        # Strategy 0: Strip <confidence>...</confidence> tags (they pollute parsing)
        text = re.sub(
            r"<confidence>.*?</confidence>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        
        # Strategy 1: ```python ... ``` (preferred)
        m = CODE_BLOCK_RE.search(text)
        if m:
            code = m.group(1).strip()
            if code:
                return code
        
        # Strategy 2: any ``` ... ``` block
        any_fence = re.search(r"```\s*\n?(.*?)```", text, flags=re.DOTALL)
        if any_fence:
            code = any_fence.group(1).strip()
            if code:
                # Strip language label if present on first line
                first_newline = code.find("\n")
                if first_newline != -1:
                    first_line = code[:first_newline].strip()
                    if first_line.lower() in (
                        "python", "py", "python3",
                    ) or len(first_line) < 12:
                        code = code[first_newline + 1:]
                return code.strip()
        
        # Strategy 3: full text if it contains a def
        if "def " in text:
            # Take from the first 'def ' onwards (skip natural-language preamble)
            first_def = text.find("def ")
            return text[first_def:].strip()
        
        # Strategy 4: try to detect indented body (e.g. CHR-CP gives "    return ..."
        # without the def line because the prompt already had it)
        lines = text.strip().split("\n")
        code_like_lines = [
            ln for ln in lines
            if ln.strip().startswith(("return ", "if ", "for ", "while ", "import ",
                                    "from ", "x =", "result", "    "))
        ]
        if len(code_like_lines) >= 2:
            # Looks like raw code body
            return text.strip()
        
        return None
    
    def _safe_execute(self, program: str) -> tuple[bool, Optional[str]]:
        """Run the program in a subprocess with timeout. Returns (passed, error)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(program)
            tmp_path = tmp.name
        
        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                timeout=self.execution_timeout,
                text=True,
            )
            if result.returncode == 0:
                return True, None
            err_msg = (result.stderr or "")[-500:]
            return False, f"non-zero exit ({result.returncode}): {err_msg}"
        except subprocess.TimeoutExpired:
            return False, f"timeout (>{self.execution_timeout}s)"
        except Exception as e:
            return False, f"execution error: {type(e).__name__}: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass