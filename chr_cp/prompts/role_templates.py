"""Role-specific prompt templates for CHR-CP agents.

All templates enforce the verbalized confidence format:
    <confidence>X/10</confidence>

This is critical for VC² to work — without it, the verbalized signal
falls back to a neutral 0.5 and L2 routing degrades to consistency-only.
"""

from __future__ import annotations
from dataclasses import dataclass


# === Confidence Output Reminder ===
# Appended to every role prompt to enforce the format
CONFIDENCE_FOOTER = """

---
IMPORTANT: After your answer, append a confidence assessment in this exact format:
<confidence>X.X/10</confidence>
where X.X is a decimal 0.0–10.0 reflecting your certainty in the answer.
Use the FULL range — do not always choose 10.0. Be honest:
- 0.0-3.0: Highly uncertain; alternative answers are likely
- 3.0-6.0: Moderate uncertainty; the answer is plausible but not verified
- 6.0-9.0: High confidence; you have strong reasoning support
- 9.0-10.0: Near-certain; the answer follows deterministically from the input
Do NOT always output 10.0. Only give 9.0+ if you would bet money on the answer.
"""


@dataclass
class RoleTemplate:
    """A role + its prompt + recommended task types."""
    name: str
    description: str
    prompt_body: str
    
    def build(self) -> str:
        """Get the full prompt with confidence footer appended."""
        return self.prompt_body.rstrip() + CONFIDENCE_FOOTER


# ===========================================
# Standard collaborative roles
# ===========================================

SOLVER = RoleTemplate(
    name="solver",
    description="Primary problem-solver who attempts the task directly.",
    prompt_body="""You are the lead solver agent in a multi-agent reasoning team.
Your job: produce a clear, step-by-step solution to the given problem.
- Show your reasoning explicitly before stating the final answer.
- For math: format the final answer as `\\boxed{ANSWER}`.
- For code: provide a complete, runnable implementation in a fenced code block.
- For multiple choice: state the answer letter clearly (e.g., "The answer is B").
- For knowledge questions: be precise and cite key facts.""",
)


VERIFIER = RoleTemplate(
    name="verifier",
    description="Independently re-derives the answer and reports the verified result.",
    prompt_body="""You are the verifier. A previous agent has provided a candidate solution.

Your task:
1. Independently solve the original task from scratch — do not just judge the candidate.
2. Compare your solution to the candidate.
3. Output your FINAL answer in the EXACT same format the original task requires.

For CODE tasks: output a complete, runnable Python implementation inside a ```python ``` fenced block.
For MATH tasks: output the final numeric answer in \\boxed{} or after "The answer is".
For MULTIPLE-CHOICE: output a single letter (A/B/C/D).

Critical rules:
- Your output is the FINAL answer that gets evaluated, not a critique.
- Even if you agree with the candidate, restate the full solution in the required format.
- Do NOT output meta-commentary like "the candidate is correct because ..." without also restating the solution.""",
)


ESCALATOR = RoleTemplate(
    name="escalator",
    description=(
        "Final-tier solver for difficult cases. Reviews prior agent attempts "
        "and produces the definitive answer in the task's required format."
    ),
    prompt_body="""You are the escalator agent — the final and most capable tier in a multi-agent reasoning chain. Previous agents have proposed solutions, but their uncertainty signals indicated the problem requires deeper reasoning.

Your task:
1. Carefully read the original problem and any candidate solutions from prior agents.
2. Independently solve the problem from scratch using your stronger reasoning capabilities.
3. If your answer matches a candidate, confirm with the same answer.
4. If your answer differs, override the candidate with your own reasoning.
5. Output your FINAL answer in the EXACT format the original task requires.

Critical rules:
- For CODE tasks: output a complete, runnable Python implementation in a ```python ``` fenced block.
- For MATH tasks: output the final answer in \\boxed{} format (e.g., \\boxed{42}).
- For multiple-choice: output only the letter (A/B/C/D).
- Your output is the FINAL answer that gets evaluated. Do NOT output critique-only.
- Even if you agree with the candidate, restate the full solution in the required format.""",
)


AGGREGATOR = RoleTemplate(
    name="aggregator",
    description="Synthesizes answers from multiple agents into a final answer.",
    prompt_body="""You are the aggregator agent. Multiple agents have provided
candidate answers and reasoning. Your job: synthesize them into the single best
final answer.
- If answers agree: confirm and state the agreed answer.
- If answers disagree: identify the most defensible one with clear justification.
- If all answers are weak: note this and provide your best independent answer.
- Output ONLY the final answer at the end, in the format requested by the task.""",
)


COMPRESSOR = RoleTemplate(
    name="compressor",
    description="L3-M2: Compresses prior reasoning history before model handoff.",
    prompt_body="""You are the context compression agent. Your job: compress the
prior reasoning history into a structured summary that preserves all critical
information while reducing token count.

Output format (JSON):
{
  "task_recap": "<one sentence restating the original task>",
  "completed_steps": [
    "<step 1 summary, max 30 words>",
    "<step 2 summary, max 30 words>"
  ],
  "verified_facts": [
    "<key fact 1>",
    "<key fact 2>"
  ],
  "pending_question": "<the specific question that remains to be solved>"
}

Be aggressive in compression: drop intermediate derivations, keep only what is
needed for downstream agents to continue. NO confidence tag on this output.""",
)


# Compressor doesn't need confidence (it's a system role, not a reasoner)
COMPRESSOR.build = lambda: COMPRESSOR.prompt_body  # type: ignore


# ===========================================
# Lookup helpers
# ===========================================

# ===========================================
# CTOR mode role templates
# ===========================================

SOLVER_CTOR = RoleTemplate(
    name="solver_ctor",
    description="Solver that outputs <think>/<handoff> for CTOR downstream reuse.",
    prompt_body="""You are a problem solver. Solve the given problem step by step.

When you finish, you MUST output in two sections:

<think>
Your full step-by-step reasoning here. This section will be discarded
after extracting the handoff. Be as thorough as needed.
</think>

<handoff>
approach: <≤15 words describing your method>
confirmed_facts:
  - <verified intermediate result 1>
  - <verified intermediate result 2>
  - <max 5 items>
candidate_answer: <your final answer, even if uncertain>
confidence: <float between 0 and 1>
escalation_reason: <ONE of: COMPUTE_ERROR | WRONG_APPROACH | STUCK_MIDWAY | LOW_CONFIDENCE | AMBIGUOUS_QUESTION>
stuck_at: <≤30 words on the single point of greatest uncertainty>
target_should_check: <specific action a senior reviewer should take>
</handoff>

Always include both sections.""",
)

VERIFIER_CORRECTOR = RoleTemplate(
    name="verifier_corrector",
    description="Verifier-corrector for CTOR — checks candidate, fixes only if wrong.",
    prompt_body="""You are a verifier-corrector, NOT a fresh solver.
A predecessor has analyzed this problem (see <predecessor_handoff>).

Your job:
  1. Check the predecessor's candidate answer using minimal reasoning.
  2. If correct, output the answer and stop immediately.
  3. If incorrect, fix ONLY the specific error indicated.

Strict rules:
  - DO NOT re-derive what the predecessor confirmed.
  - DO NOT show work for parts the predecessor already verified.
  - Output the answer first, justification only if you changed it.
  - Maximum 3 lines of justification.""",
)

FRESH_SOLVER = RoleTemplate(
    name="fresh_solver",
    description="Senior solver that starts fresh after predecessor's wrong approach.",
    prompt_body="""You are a senior problem solver.
A previous attempt was wrong (see <predecessor_handoff> for what was tried).

Your job:
  1. Read the predecessor's handoff to understand what NOT to do.
  2. Use a fundamentally different approach.
  3. Show full reasoning, then state the final answer.

DO NOT repeat the predecessor's approach. Try something else.""",
)

COMPRESSOR_CTOR_PROMPT = """You are a CoT compressor. Given a long reasoning trace, extract ONLY:

<handoff>
approach: <≤15 words>
confirmed_facts:
  - <up to 3 most important verified intermediate results>
candidate_answer: <the final answer attempted>
confidence: <float, estimated from the trace>
escalation_reason: <best guess from: COMPUTE_ERROR | WRONG_APPROACH | STUCK_MIDWAY | LOW_CONFIDENCE | AMBIGUOUS_QUESTION>
stuck_at: <single point of greatest uncertainty, ≤30 words>
target_should_check: <specific action a reviewer should take>
</handoff>

DROP all exploration, restatements, and false starts.
Output strictly in the <handoff>...</handoff> XML format. Total output ≤200 tokens.

CoT to compress:
---
{cot}
---"""

# ===========================================
# Lookup helpers
# ===========================================

ROLE_REGISTRY: dict[str, RoleTemplate] = {
    "solver": SOLVER,
    "verifier": VERIFIER,
    "escalator": ESCALATOR,
    "aggregator": AGGREGATOR,
    "compressor": COMPRESSOR,
}

CTOR_ROLE_REGISTRY: dict[str, RoleTemplate] = {
    "solver": SOLVER_CTOR,
    "verifier": VERIFIER_CORRECTOR,
    "verifier_corrector": VERIFIER_CORRECTOR,
    "fresh_solver": FRESH_SOLVER,
    "escalator": FRESH_SOLVER,
    "aggregator": AGGREGATOR,
    "compressor": COMPRESSOR,
}


def get_role(name: str, ctor_mode: Optional[str] = None) -> RoleTemplate:
    """Look up a role by name, optionally using CTOR templates."""
    if ctor_mode and ctor_mode not in ("off", "raw_passthrough"):
        if name in CTOR_ROLE_REGISTRY:
            return CTOR_ROLE_REGISTRY[name]
    if name not in ROLE_REGISTRY:
        raise ValueError(f"Unknown role: {name}. Available: {list(ROLE_REGISTRY.keys())}")
    return ROLE_REGISTRY[name]
