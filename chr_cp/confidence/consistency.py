"""Multi-sample self-consistency uncertainty signal.

Strategy:
1. Sample the same prompt K times at different temperatures
2. Compare the K answers pairwise
3. Average pairwise similarity → consistency score
4. Uncertainty = 1 - consistency
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
from itertools import combinations

from loguru import logger

from chr_cp.clients import ClientPool, Tier
from chr_cp.clients.base_client import CompletionResponse
from chr_cp.confidence.verbalized import VerbalizedConfidenceParser
from chr_cp.utils.text_sim import (
    extract_final_number,
    numeric_match,
    rouge_l_f1,
    extract_mc_answer,
)
from chr_cp.utils.ast_diff import code_similarity


class TaskType(str, Enum):
    """Task type determines the similarity metric used in consistency."""
    NUMERIC = "numeric"          # GSM8K, MATH → exact number match
    MULTIPLE_CHOICE = "mc"       # MMLU → letter match
    CODE = "code"                # HumanEval, MBPP → AST similarity
    OPEN_TEXT = "open_text"      # general → ROUGE-L
    
    @classmethod
    def from_benchmark(cls, benchmark: str) -> "TaskType":
        """Infer task type from benchmark name."""
        b = benchmark.lower()
        if b in ("gsm8k", "math", "math500", "svamp"):
            return cls.NUMERIC
        if b in ("mmlu", "bbh"):
            return cls.MULTIPLE_CHOICE
        if b in ("humaneval", "mbpp", "humaneval-x"):
            return cls.CODE
        return cls.OPEN_TEXT


@dataclass
class ConsistencyResult:
    """Result of a multi-sample consistency estimation."""

    samples: list[str]                    # raw text outputs
    extracted_answers: list               # extracted canonical answers (depends on task type)
    pairwise_similarities: list[float]    # K*(K-1)/2 pairwise scores
    mean_similarity: float                # mean of pairwise similarities
    responses: list = field(default_factory=list)  # CompletionResponse objects

    @property
    def uncertainty(self) -> float:
        """U_consistency = 1 - mean_similarity, clipped to [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.mean_similarity))


# Default sampling temperatures for K=3 self-consistency.
# Three temperatures span exploration-vs-greedy spectrum:
# - 0.3: near-greedy, baseline answer
# - 0.7: moderate diversity
# - 0.9: high diversity, surfaces alternative paths
# Default sampling temperatures for K=3 self-consistency.
# Three temperatures span exploration-vs-greedy spectrum.
DEFAULT_TEMPERATURES = [0.3, 0.7, 0.9]


def temperatures_for_k(k: int) -> list[float]:
    """Generate K temperatures spanning the greedy → diverse spectrum.
    
    K=3: [0.3, 0.7, 0.9]      (default, near-greedy + moderate + diverse)
    K=5: [0.2, 0.5, 0.7, 0.85, 1.0]
    K=7: [0.2, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0]
    
    For other K values, we linearly interpolate between 0.3 and 1.0.
    """
    if k == 3:
        return [0.3, 0.7, 0.9]
    if k == 5:
        return [0.2, 0.5, 0.7, 0.85, 1.0]
    if k == 7:
        return [0.2, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0]
    
    # Fallback: linear interpolation
    if k <= 1:
        return [0.3]
    if k == 2:
        return [0.3, 0.9]
    step = (1.0 - 0.3) / (k - 1)
    return [round(0.3 + i * step, 3) for i in range(k)]


class ConsistencyEstimator:
    """Estimates uncertainty via multi-sample consistency.
    
    Usage:
        estimator = ConsistencyEstimator(pool=pool)
        result = estimator.estimate(
            messages=messages,
            tier=Tier.T2,
            task_type=TaskType.NUMERIC,
        )
        print(f"Uncertainty: {result.uncertainty:.3f}")
    """
    
    # Similarity functions for each task type
    SIMILARITY_FUNCS: dict[TaskType, Callable[[str, str], float]] = {
        TaskType.OPEN_TEXT: rouge_l_f1,
        TaskType.CODE: lambda a, b: code_similarity(a, b, language="python"),
    }
    
    def __init__(
        self,
        pool: ClientPool,
        temperatures: Optional[list[float]] = None,
        k_samples: Optional[int] = None,
        cost_tracker=None,
        vc_parser: Optional[VerbalizedConfidenceParser] = None,
    ):
        """
        Args:
            pool: Client pool
            temperatures: Explicit temperature list (overrides k_samples)
            k_samples: Number of consistency samples (3, 5, 7). If provided
                    and temperatures is None, temperatures_for_k(k) is used.
            cost_tracker: Optional CostTracker
            vc_parser: Optional confidence tag parser
        """
        self.pool = pool
        
        # Resolve temperatures: explicit list > k_samples > default
        if temperatures is not None:
            self.temperatures = temperatures
        elif k_samples is not None:
            self.temperatures = temperatures_for_k(k_samples)
        else:
            self.temperatures = DEFAULT_TEMPERATURES
        
        self.k_samples = len(self.temperatures)  # convenience attribute
        self.cost_tracker = cost_tracker
        self.vc_parser = vc_parser or VerbalizedConfidenceParser()
        
        logger.debug(
            f"ConsistencyEstimator: K={self.k_samples}, "
            f"temperatures={self.temperatures}"
        )
    
    def estimate(
        self,
        messages: list[dict],
        tier: str | Tier,
        task_type: TaskType,
        max_tokens: int = 2048,
        task_id: Optional[str] = None,
        anchor_answer: Optional[str] = None,
    ) -> ConsistencyResult:
        """Run K samples and compute consistency-based uncertainty.

        Args:
            messages: Prompt as message list
            tier: Which tier to sample from
            task_type: Determines similarity metric
            max_tokens: Per-sample completion limit
            task_id: For cost tracking
            anchor_answer: If provided, K verifier samples (short, 128t)
                are used instead of full reasoning samples. Each sample
                votes YES/NO on the anchor. Saves ~91% completion tokens.

        Returns:
            ConsistencyResult
        """
        # 1. Build verify messages if anchor_answer is provided
        if anchor_answer is not None:
            verify_messages = self._build_verify_messages(messages, anchor_answer)
            verify_max_tokens = 128
        else:
            verify_messages = messages
            verify_max_tokens = max_tokens

        # 1. Sample K times at different temperatures
        # T1 (qwen): parallel — no prefix cache, no benefit from serial
        # T2/T3 (deepseek): serial — each call warms cache for the next,
        #   hitting primary call's recently-written prefix at 1/10 token cost
        tier_str = tier.value if isinstance(tier, Tier) else str(tier)
        use_parallel = tier_str == "T1"

        def _sample_one(temp: float) -> tuple[float, Optional[CompletionResponse]]:
            try:
                resp = self.pool.invoke(
                    tier=tier,
                    messages=verify_messages,
                    temperature=temp,
                    max_tokens=verify_max_tokens,
                )
                if self.cost_tracker:
                    self.cost_tracker.record(
                        resp,
                        task_id=task_id,
                        step_id="consistency_sample",
                        routing_action="BRANCH",
                    )
                return temp, resp
            except Exception as e:
                logger.warning(f"Consistency sample failed at temp={temp}: {e}")
                return temp, None

        responses: list[CompletionResponse] = []
        results_by_temp: dict[float, Optional[CompletionResponse]] = {}

        if use_parallel:
            with ThreadPoolExecutor(max_workers=len(self.temperatures)) as executor:
                futures = {executor.submit(_sample_one, t): t for t in self.temperatures}
                for future in as_completed(futures):
                    temp, resp = future.result()
                    results_by_temp[temp] = resp
        else:
            # Serial: primary call's cache is warm; each call feeds the next
            for temp in self.temperatures:
                t, resp = _sample_one(temp)
                if resp is not None:
                    results_by_temp[t] = resp

        # Preserve original temperature order
        for temp in self.temperatures:
            resp = results_by_temp.get(temp)
            if resp is not None:
                responses.append(resp)
        
        if len(responses) < 2:
            # Not enough samples to compute consistency
            return ConsistencyResult(
                samples=[r.content for r in responses],
                extracted_answers=[],
                pairwise_similarities=[],
                mean_similarity=0.5,
                responses=responses,
            )
        
        # 2. Strip <confidence>...</confidence> tags before similarity
        # (we don't want score variance to be driven by the confidence tag)
        cleaned_samples = [
            self.vc_parser.strip_confidence_tag(r.content) for r in responses
        ]
        
        # 3. Extract canonical answers based on task type
        extracted = self._extract_answers(cleaned_samples, task_type)
        
        # 4. Pairwise similarity
        pairwise = self._pairwise_similarity(cleaned_samples, extracted, task_type)
        
        mean_sim = sum(pairwise) / len(pairwise) if pairwise else 0.5
        
        return ConsistencyResult(
            samples=cleaned_samples,
            extracted_answers=extracted,
            pairwise_similarities=pairwise,
            mean_similarity=mean_sim,
            responses=responses,
        )
    
    def _extract_answers(
        self,
        samples: list[str],
        task_type: TaskType,
    ) -> list:
        """Extract canonical answer from each sample."""
        if task_type == TaskType.NUMERIC:
            return [extract_final_number(s) for s in samples]
        elif task_type == TaskType.MULTIPLE_CHOICE:
            return [extract_mc_answer(s) for s in samples]
        else:
            # For CODE and OPEN_TEXT, the "answer" is the full text
            # (similarity will be computed on text directly)
            return list(samples)
    
    @staticmethod
    def _build_verify_messages(
        original_messages: list[dict],
        anchor_answer: str,
    ) -> list[dict]:
        """Build verify prompt: original problem + candidate answer + verdict request."""
        # Extract problem: prefer system message (TASK-ANCHOR), fallback to user
        problem_text = ""
        for m in original_messages:
            if m["role"] == "system":
                # Extract TASK-ANCHOR section from system message
                import re
                m_task = re.search(r'=== TASK ===\n(.*?)(?:\n===|\Z)', m["content"], re.DOTALL)
                if m_task:
                    problem_text = m_task.group(1).strip()
                    break
        if not problem_text:
            for m in original_messages:
                if m["role"] == "user":
                    problem_text = m["content"]
                    break

        verify_user = (
            f"<problem>\n{problem_text}\n</problem>\n\n"
            f"<candidate_answer>\n{anchor_answer}\n</candidate_answer>\n\n"
            f"Is this candidate answer correct? Be skeptical — only agree if it is genuinely right.\n"
            f"Reply with EXACTLY:\n"
            f"YES — if the candidate is correct\n"
            f"NO <answer>X</answer> — if wrong, provide your corrected answer"
        )

        return [
            {"role": "system", "content": (
                "You are a verifier. Review a candidate answer. Be skeptical. "
                "Only agree if the answer is genuinely correct. "
                "Reply YES or NO <answer>X</answer>."
            )},
            {"role": "user", "content": verify_user},
        ]

    @staticmethod
    def _parse_verify_response(content: str, anchor_answer: str) -> str:
        """Extract the voted answer from a verify response.
        - "YES" → anchor_answer
        - "NO <answer>X</answer>" → X
        - Otherwise → anchor_answer (default to anchor if unparseable)
        """
        import re
        content_upper = content.strip().upper()
        if content_upper.startswith("YES") or content_upper == "YES":
            return anchor_answer
        m = re.search(r'<answer>\s*(.+?)\s*</answer>', content, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r'NO\s+(.+)', content_upper)
        if m:
            return m.group(1).strip()
        return anchor_answer

    def _pairwise_similarity(
        self,
        samples: list[str],
        extracted: list,
        task_type: TaskType,
    ) -> list[float]:
        """Compute pairwise similarity scores."""
        scores = []
        n = len(samples)
        
        for i, j in combinations(range(n), 2):
            if task_type == TaskType.NUMERIC:
                if extracted[i] is not None and extracted[j] is not None:
                    sim = 1.0 if numeric_match(extracted[i], extracted[j]) else 0.0
                else:
                    # Non-numeric answer (e.g. name, symbol): fall back to
                    # string match so \"Evelyn\" vs \"Evelyn\" is not misjudged
                    # as disagreement.
                    sim = 1.0 if samples[i] == samples[j] else 0.0
            elif task_type == TaskType.MULTIPLE_CHOICE:
                if extracted[i] is None or extracted[j] is None:
                    sim = 0.0
                else:
                    sim = 1.0 if extracted[i] == extracted[j] else 0.0
            elif task_type == TaskType.CODE:
                sim = code_similarity(samples[i], samples[j], language="python")
            else:
                sim = rouge_l_f1(samples[i], samples[j])
            scores.append(sim)
        
        return scores