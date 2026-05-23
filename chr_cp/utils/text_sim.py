"""Text and numeric similarity utilities for self-consistency estimation."""

from __future__ import annotations
import re
from collections import Counter
from typing import Optional, FrozenSet


# ============================================================
# LaTeX answer normalization
# ============================================================

# LaTeX commands whose name carries NO mathematical meaning.
# After removing the backslash, their plain-text name is stripped so that
# e.g. "\left( 3 \right)" and "(3)" both normalize to "(3)".
#
# Categories:
#   Delimiters:  \left \right \big \Big \bigg \Bigg \bigl \bigr ...
#   Text mode:   \text \mbox \textrm \mathrm ...
#   Font/style:  \mathbf \mathcal \mathbb \boldsymbol \mathsf ...
#   Display:     \displaystyle \textstyle \limits \nolimits
#   Decorations: \overline \underline \widehat \widetilde
#   Environments:\begin \end
#   Spacing:     \qquad \quad \, \!  (\, and \! are single chars, handled by \ removal)
#   Old-style:   \rm \bf \it \sf \tt

_LATEX_FORMAT_CMDS: FrozenSet[str] = frozenset({
    # Delimiter sizing
    "left", "right",
    "big", "Big", "bigg", "Bigg",
    "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl", "Biggr",
    # Text / inline text
    "text", "mbox",
    "textrm", "mathrm", "mathit", "mathbf", "mathcal", "mathbb",
    "boldsymbol", "mathsf", "mathtt", "textsf", "texttt",
    "textbf", "textit", "textmd", "textup", "textnormal",
    # Display / limits
    "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
    "limits", "nolimits", "displaylimits",
    # Decorations (appearance only — \bar \hat \dot \vec etc. carry meaning, kept)
    "overline", "underline", "widehat", "widetilde",
    "overbrace", "underbrace",
    # Environments
    "begin", "end",
    # Spacing
    "qquad", "quad", "enspace", "thinspace", "medspace", "thickspace",
    # Negation (just "not", \not is a prefix; the combined symbol e.g. \not= or \ne is separate)
    # Legacy font commands
    "rm", "bf", "it", "sf", "tt",
})

# Commands to normalize (not strip) — map to canonical forms
_LATEX_NORMALIZE: dict = {
    "\\dfrac": "\\frac",
    "\\tfrac": "\\frac",
}


def normalize_latex_answer(s: str) -> str:
    """Normalize a LaTeX math answer for string comparison.

    Strategy:
    1. Normalize fraction commands (\\dfrac, \\tfrac → \\frac)
    2. Strip formatting \\commands (left/right/text/begin/end/mathbf/…)
    3. Remove remaining backslashes (semantic commands become plain text)
    4. Strip braces, spaces; lowercase
    5. Handle degree marker (^circ → "")

    This is symmetric (same transform on both prediction and reference),
    so only format *differences* cause mismatches. Semantic commands like
    \\frac, \\sqrt, \\sin, \\pi survive as plain text and match symmetrically.
    """
    # Step 1: Normalize fractions before stripping backslash
    for cmd, replacement in _LATEX_NORMALIZE.items():
        s = s.replace(cmd, replacement)

    # Step 2: Strip formatting commands (with backslash)
    for cmd in _LATEX_FORMAT_CMDS:
        s = s.replace("\\" + cmd, "")

    # Step 3: Remove remaining backslashes
    s = s.replace("\\", "")

    # Step 4: Remove braces and dollar signs
    s = s.replace("{", "").replace("}", "").replace("$", "")

    # Step 5: Strip degree marker (unit, not a value)
    s = s.replace("^circ", "")

    # Step 6: Remove spaces and lowercase
    return s.replace(" ", "").lower()


# === Numeric answer extraction & matching ===

# Match numbers including:
# - integers: 42, -5
# - decimals: 3.14, -0.5
# - fractions: 1/2 (treated as separate numbers, see _is_numeric_answer)
# - scientific notation: 1.5e-3
NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def extract_final_number(text: str) -> Optional[float]:
    """Extract the final numeric answer from text.
    
    Strategy:
    1. Look for "answer is X" or "= X" patterns first
    2. If not found, take the last number in the text
    """
    if not text:
        return None
    
    # Pattern 1: explicit answer markers
    answer_patterns = [
        r"(?:答案是|答案为|the answer is|final answer:?|结果是)\s*[\$]?(-?\d+(?:\.\d+)?)",
        r"(?:=|equals?)\s*\$?(-?\d+(?:\.\d+)?)\s*[.\$]?\s*$",
        r"\\boxed\{(-?\d+(?:\.\d+)?)\}",
    ]
    for pattern in answer_patterns:
        m = re.findall(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m[-1])
            except ValueError:
                continue
    
    # Pattern 2: last number in text (fallback)
    numbers = NUMERIC_PATTERN.findall(text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def numeric_match(a: Optional[float], b: Optional[float], tol: float = 1e-4) -> bool:
    """Check if two numeric answers match within tolerance."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# === Lexical similarity (word-overlap based ROUGE-L approximation) ===

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer (good enough for similarity)."""
    return re.findall(r"\w+", text.lower())


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity over word tokens."""
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def rouge_l_f1(a: str, b: str) -> float:
    """Approximate ROUGE-L F1 score using LCS over word tokens.
    
    Uses dynamic programming, O(m*n). Fine for short outputs (<2000 tokens each).
    """
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    
    m, n = len(ta), len(tb)
    # DP table for LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ta[i - 1] == tb[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    
    p = lcs / n
    r = lcs / m
    return 2 * p * r / (p + r)


# === Multiple-choice answer extraction ===

def extract_mc_answer(text: str, num_options: int = 5) -> Optional[str]:
    """Extract a multiple-choice answer letter from text."""
    if not text:
        return None

    # Strip <confidence> tags that can break end-of-line patterns
    cleaned = re.sub(r'<confidence>.*?</confidence>', '', text, flags=re.DOTALL).strip()
    letters = "ABCDEFGHIJ"[:num_options]

    patterns = [
        r"(?:答案是|答案为|the answer is|final answer:?)\s*[\(\[]?\s*([" + letters + r"])\s*[\)\]]?",
        r"\\boxed\{([" + letters + r"])\}",
        r"\b([" + letters + r"])\b\s*[.\)]?\s*$",  # last standalone letter
    ]
    for pattern in patterns:
        m = re.findall(pattern, cleaned, flags=re.IGNORECASE)
        if m:
            return m[-1].upper()
    # Fallback: find last standalone letter in last 200 chars
    tail = cleaned[-200:].upper()
    m = re.findall(r"\b([" + letters + r"])\b", tail)
    if m:
        return m[-1]
    return None