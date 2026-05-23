"""L1 Difficulty Probe — single T1 call to assess task difficulty before routing.

Returns a DifficultyProbeResult that L1 uses to decide start_tier and whether
to skip the cascade entirely (direct dispatch to T4).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re
import json

from loguru import logger


DIFFICULTY_PROBE_PROMPT = """You are a task difficulty assessor, NOT a solver.
Read the following problem and evaluate its difficulty.

CRITICAL: Be brutally honest about your limits. If you can only guess or the problem
requires graduate-level knowledge, say "unsure" or "cannot_solve".
It is BETTER to admit uncertainty than to pretend you can solve it.

Output in this EXACT format:
<assessment>
domain: <math|physics|chemistry|biology|code|commonsense|other>
self_assessment: <can_solve|unsure|cannot_solve>
reasoning_depth: <shallow|medium|deep>
needs_expert_knowledge: <true|false>
</assessment>

<tentative_answer>
Give a quick answer (may be incomplete or uncertain, just your best guess).
If you truly cannot even guess, write "NONE".
</tentative_answer>

Problem:
---
{problem}
---"""


@dataclass
class DifficultyProbeResult:
    """Result of a single T1 difficulty probe call."""
    domain: str = "other"
    t1_self_assessment: str = "unsure"  # can_solve / unsure / cannot_solve
    reasoning_depth: str = "medium"     # shallow / medium / deep
    needs_expert_knowledge: bool = False
    t1_tentative_answer: str = ""
    difficulty_score: int = 5
    probe_cost_usd: float = 0.0

    @property
    def is_direct_candidate(self) -> bool:
        """cannot_solve + (medium or deep) → direct T4."""
        return (
            self.t1_self_assessment == "cannot_solve"
            and self.reasoning_depth in ("medium", "deep")
        )

    @property
    def start_tier(self) -> str:
        """Determine starting tier from difficulty probe."""
        a = self.t1_self_assessment
        d = self.reasoning_depth
        e = self.needs_expert_knowledge

        if a == "can_solve":
            return "T2"
        if a == "unsure":
            if d == "deep" or e:
                return "T3"
            return "T2"
        if a == "cannot_solve":
            if d == "deep":
                return "T4"
            if d == "medium":
                return "T3"
            return "T2"  # shallow — probe may be wrong, conservative
        return "T2"


def compute_difficulty_score(
    self_assessment: str,
    reasoning_depth: str,
    needs_expert: bool,
) -> int:
    """Weighted score 1-10 from probe signals."""
    base = {"can_solve": 3, "unsure": 6, "cannot_solve": 8}.get(self_assessment, 6)
    if reasoning_depth == "deep":
        base += 2
    elif reasoning_depth == "shallow":
        base -= 1
    if needs_expert:
        base += 1
    return max(1, min(10, base))


def parse_probe_response(text: str) -> DifficultyProbeResult:
    """Parse structured fields from probe model output. Conservative defaults on failure."""
    result = DifficultyProbeResult()

    # Extract <assessment> block
    m = re.search(r'<assessment>(.*?)</assessment>', text, re.DOTALL)
    if m:
        block = m.group(1)
        for line in block.splitlines():
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                val = val.strip()
                if key == 'domain':
                    result.domain = val
                elif key == 'self_assessment':
                    if val in ('can_solve', 'unsure', 'cannot_solve'):
                        result.t1_self_assessment = val
                elif key == 'reasoning_depth':
                    if val in ('shallow', 'medium', 'deep'):
                        result.reasoning_depth = val
                elif key == 'needs_expert_knowledge':
                    result.needs_expert_knowledge = val.lower() == 'true'

    # Extract <tentative_answer> block
    m = re.search(r'<tentative_answer>(.*?)</tentative_answer>', text, re.DOTALL)
    if m:
        ans = m.group(1).strip()
        if ans and ans.upper() != "NONE":
            result.t1_tentative_answer = ans

    result.difficulty_score = compute_difficulty_score(
        result.t1_self_assessment,
        result.reasoning_depth,
        result.needs_expert_knowledge,
    )
    return result
