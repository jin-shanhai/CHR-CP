"""Robust answer verification using a multi-layer cascade.

Layers:
  1. Exact match + type-specific normalization (fast, catches ~80%)
  2. math-verify  (HF 2024, symbolic equivalence)
  3. SymPy        (fallback for symbolic expressions)
  4. LLM-as-judge (final fallback for semantic equivalence)

Usage:
    verifier = AnswerVerifier(judge_pool=pool)
    result = verifier.verify(
        pred="5",
        ref="x=5",
        problem="Solve for x: ...",
    )
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger


class AnswerType(str, Enum):
    SCALAR = "scalar"
    FRACTION = "fraction"
    EXPRESSION = "expression"
    SET = "set"
    INTERVAL = "interval"
    TUPLE = "tuple"
    MATRIX = "matrix"
    LETTER = "letter"
    INTEGER = "integer"
    STRING = "string"
    UNKNOWN = "unknown"


@dataclass
class VerifyResult:
    correct: bool
    judge_layer: str
    confidence: float = 1.0
    answer_type: Optional[AnswerType] = None
    pred_normalized: Optional[str] = None
    ref_normalized: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ============================================================================
# Answer Extraction
# ============================================================================

def extract_boxed(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{...} in text (nested-brace safe)."""
    idx = 0
    last_match = None
    while True:
        m = re.search(r'\\boxed\{', text[idx:])
        if not m:
            break
        start = idx + m.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            last_match = text[start:i - 1]
        idx = i
    return last_match.strip() if last_match else None


# ============================================================================
# Answer Type Detection
# ============================================================================

def detect_answer_type(ref: str) -> AnswerType:
    """Detect the type of the reference answer to route verification."""
    r = ref.strip()

    if len(r) == 1 and r.upper() in "ABCDEFGHIJ":
        return AnswerType.LETTER

    if re.match(r'^-?\d+$', r):
        return AnswerType.INTEGER

    if re.match(r'^-?\d+\.\d+$', r):
        return AnswerType.SCALAR

    if re.match(r'.*\\frac\{', r) or re.match(r'^-?\d+/\d+$', r):
        return AnswerType.FRACTION

    if re.match(r'^[\(\[].*[\)\]]$', r) and (',' in r or 'infty' in r.lower()):
        return AnswerType.TUPLE if not ('infty' in r or '...' in r) else AnswerType.INTERVAL

    if ',' in r and not r.startswith('('):
        return AnswerType.SET

    if 'begin{pmatrix}' in r or 'begin{bmatrix}' in r:
        return AnswerType.MATRIX

    if any(op in r for op in ['\\sqrt', '\\sin', '\\cos', '\\log', '\\pi', '^', '_']):
        return AnswerType.EXPRESSION

    return AnswerType.UNKNOWN


# ============================================================================
# Layer 1: Exact + Type-specific normalization
# ============================================================================

def _normalize_set(s: str) -> Optional[frozenset]:
    """Parse 'a, b, c' as a set."""
    if ',' not in s:
        return None
    parts = [p.strip() for p in re.split(r'[,;]', s)]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    return frozenset(parts)


def _type_specific_match(pred: str, ref: str, ans_type: AnswerType) -> Optional[bool]:
    """Type-aware equivalence. Returns None if not applicable."""
    # Strip variable prefix (x=5 → 5)
    def strip_var(s):
        m = re.match(r'^[a-zA-Z]\s*=\s*(.+)$', s.strip())
        return m.group(1) if m else s

    # Exact match after normalization (handles most cases)
    if pred == ref:
        return True

    if ans_type == AnswerType.LETTER:
        return pred.strip().upper() == ref.strip().upper()

    if ans_type == AnswerType.INTEGER:
        try:
            return int(pred.strip()) == int(ref.strip())
        except ValueError:
            pass

    if ans_type == AnswerType.SCALAR:
        try:
            return abs(float(pred) - float(ref)) < 1e-6
        except (ValueError, TypeError):
            pass

    if ans_type == AnswerType.SET:
        p_set = _normalize_set(pred)
        r_set = _normalize_set(ref)
        if p_set is not None and r_set is not None:
            try:
                return frozenset(float(x) for x in p_set) == frozenset(float(x) for x in r_set)
            except ValueError:
                return p_set == r_set

    # Variable prefix normalization (x=5 ≡ 5)
    p_stripped = strip_var(pred)
    r_stripped = strip_var(ref)
    if p_stripped != pred or r_stripped != ref:
        if _type_specific_match(p_stripped, r_stripped, ans_type):
            return True

    # LaTeX normalization for exact match
    from chr_cp.utils.text_sim import normalize_latex_answer
    if normalize_latex_answer(pred) == normalize_latex_answer(ref):
        return True

    return None


# ============================================================================
# Layer 2: math-verify
# ============================================================================

def _math_verify_layer(pred: str, ref: str) -> Optional[bool]:
    """Use math-verify library for symbolic equivalence."""
    try:
        from math_verify import parse, verify
        pred_latex = f"${pred}$" if not pred.startswith('$') else pred
        ref_latex = f"${ref}$" if not ref.startswith('$') else ref
        pred_parsed = parse(pred_latex)
        ref_parsed = parse(ref_latex)
        return bool(verify(ref_parsed, pred_parsed))
    except ImportError:
        return None
    except Exception:
        return None


# ============================================================================
# Layer 3: SymPy
# ============================================================================

def _sympy_layer(pred: str, ref: str) -> Optional[bool]:
    """SymPy-based symbolic equivalence."""
    try:
        from sympy import simplify, Eq, S
        from sympy.parsing.latex import parse_latex

        def clean(s):
            s = s.strip().strip('$').strip()
            s = re.sub(r'\\left\s*', '', s)
            s = re.sub(r'\\right\s*', '', s)
            s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
            m = re.match(r'^[a-zA-Z]\s*=\s*(.+)$', s)
            if m:
                s = m.group(1)
            return s.strip()

        p_clean = clean(pred)
        r_clean = clean(ref)

        p_expr = parse_latex(p_clean)
        r_expr = parse_latex(r_clean)

        diff = simplify(p_expr - r_expr)
        if diff == 0 or diff == S.Zero:
            return True

        try:
            return abs(float(p_expr) - float(r_expr)) < 1e-6
        except (TypeError, ValueError):
            return False
    except ImportError:
        return None
    except Exception:
        return None


# ============================================================================
# Layer 4: LLM-as-Judge
# ============================================================================

_LLM_JUDGE_CACHE: dict = {}

LLM_JUDGE_PROMPT = """Determine if a student's answer is mathematically equivalent to a reference answer.

Problem: {problem}

Reference answer: {ref}
Student's answer: {pred}

Are they mathematically equivalent? Consider:
- Different but equivalent forms (e.g., "x=5" ≡ "5", "1/2" ≡ "0.5" ≡ "\\frac{{1}}{{2}}")
- Order-independent collections (e.g., "1,-2" ≡ "-2,1" for solution sets)
- Algebraic simplification (e.g., "\\sqrt{{8}}" ≡ "2\\sqrt{{2}}")

Reply with EXACTLY one word:
YES - if equivalent
NO  - if not equivalent

Reply:"""


def _llm_layer(pred: str, ref: str, problem: str, judge_pool) -> Optional[bool]:
    """LLM-as-judge. Returns None if pool not available."""
    if judge_pool is None:
        return None

    cache_key = (pred.strip(), ref.strip())
    if cache_key in _LLM_JUDGE_CACHE:
        return _LLM_JUDGE_CACHE[cache_key]

    prompt = LLM_JUDGE_PROMPT.format(
        problem=problem[:400],
        ref=ref,
        pred=pred,
    )

    try:
        response = judge_pool.invoke(
            tier="T1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        verdict = response.content.strip().upper()
        result = verdict.startswith("YES")
        _LLM_JUDGE_CACHE[cache_key] = result
        return result
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return None


# ============================================================================
# Main Verifier Class
# ============================================================================

class AnswerVerifier:
    """Robust answer verifier with multi-layer cascade and audit trail."""

    def __init__(
        self,
        judge_pool=None,
        use_math_verify: bool = True,
        use_sympy: bool = True,
        use_llm: bool = True,
        regression_test_path: Optional[str] = None,
    ):
        self.judge_pool = judge_pool
        self.use_math_verify = use_math_verify
        self.use_sympy = use_sympy
        self.use_llm = use_llm
        self.regression_cases: dict = {}
        if regression_test_path:
            self._load_regression_cases(regression_test_path)

        # Startup sanity: verify math-verify is actually functional
        if self.use_math_verify:
            ok = _math_verify_layer("2", "2")
            if ok is None:
                logger.error("math-verify unavailable! Format errors will NOT be caught.")
                self.use_math_verify = False

    def _load_regression_cases(self, path: str):
        p = Path(path)
        if not p.exists():
            logger.warning(f"Regression test file not found: {path}")
            return
        with open(p) as f:
            for line in f:
                case = json.loads(line)
                key = (case["pred"].strip(), case["ref"].strip())
                self.regression_cases[key] = case["expected_correct"]
        logger.info(f"Loaded {len(self.regression_cases)} regression cases")

    def verify(
        self,
        pred: str,
        ref: str,
        problem: str = "",
        answer_type: Optional[AnswerType] = None,
    ) -> VerifyResult:
        """Run the full verification cascade."""
        if pred is None or ref is None:
            return VerifyResult(correct=False, judge_layer="extract_fail",
                                error=f"pred={pred!r} ref={ref!r}")

        pred = pred.strip()
        ref = ref.strip()

        if not pred or not ref:
            return VerifyResult(correct=False, judge_layer="extract_fail",
                                error="empty answer")

        if answer_type is None:
            answer_type = detect_answer_type(ref)

        # Regression test override (highest authority)
        reg_key = (pred, ref)
        if reg_key in self.regression_cases:
            expected = self.regression_cases[reg_key]
            return VerifyResult(correct=expected, judge_layer="regression",
                                answer_type=answer_type,
                                metadata={"source": "regression"})

        # Exact match
        if pred == ref:
            return VerifyResult(correct=True, judge_layer="exact",
                                answer_type=answer_type)

        # Layer 1: Type-specific normalization
        type_result = _type_specific_match(pred, ref, answer_type)
        if type_result is True:
            return VerifyResult(correct=True, judge_layer="type_norm",
                                answer_type=answer_type)
        if type_result is False and answer_type in (AnswerType.LETTER, AnswerType.INTEGER):
            return VerifyResult(correct=False, judge_layer="type_norm",
                                answer_type=answer_type)

        # Layer 2: math-verify
        if self.use_math_verify:
            r = _math_verify_layer(pred, ref)
            logger.debug(f"math-verify: pred={pred[:60]!r} ref={ref[:60]!r} type={answer_type.value} result={r}")
            if r is True:
                return VerifyResult(correct=True, judge_layer="math_verify",
                                    answer_type=answer_type)

        # Layer 3: SymPy
        if self.use_sympy:
            r = _sympy_layer(pred, ref)
            logger.debug(f"SymPy: pred={pred[:60]!r} ref={ref[:60]!r} result={r}")
            if r is True:
                return VerifyResult(correct=True, judge_layer="sympy",
                                    answer_type=answer_type)

        # Layer 4: LLM judge
        if self.use_llm and self.judge_pool is not None:
            r = _llm_layer(pred, ref, problem, self.judge_pool)
            if r is True:
                return VerifyResult(correct=True, judge_layer="llm",
                                    answer_type=answer_type)
            if r is False:
                return VerifyResult(correct=False, judge_layer="llm_rejected",
                                    answer_type=answer_type)

        return VerifyResult(correct=False, judge_layer="rejected",
                            answer_type=answer_type)
