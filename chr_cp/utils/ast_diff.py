"""AST-based code similarity for self-consistency on code generation tasks.

Uses Python's built-in `ast` module for Python code. For other languages,
falls back to lexical similarity (text_sim.rouge_l_f1).
"""

from __future__ import annotations
import ast
import re
from collections import Counter
from typing import Optional

from chr_cp.utils.text_sim import rouge_l_f1


def extract_code_block(text: str, language: str = "python") -> Optional[str]:
    """Extract the first ```language ... ``` block from text.
    
    If no fenced block found, returns None.
    """
    if not text:
        return None
    
    # Try language-specific fence first
    pattern = rf"```{language}\s*\n(.*?)```"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    # Fallback to any fenced block
    m = re.search(r"```\s*\n?(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    
    return None


def _ast_node_signatures(tree: ast.AST) -> Counter:
    """Collect a Counter of AST node type signatures.
    
    Each node contributes its type name. We ignore line numbers and
    column offsets for similarity comparison.
    """
    counter = Counter()
    for node in ast.walk(tree):
        counter[type(node).__name__] += 1
    return counter


def python_ast_similarity(code_a: str, code_b: str) -> float:
    """Compute structural similarity between two Python code snippets.
    
    Uses cosine similarity over node-type counts.
    Returns 0.0 if either snippet fails to parse.
    
    Note: This is structure-only; two snippets with same structure but
    different identifiers will get high similarity. For CHR-CP self-consistency,
    this is a feature (we want to detect logical equivalence, not surface form).
    """
    try:
        tree_a = ast.parse(code_a)
        tree_b = ast.parse(code_b)
    except SyntaxError:
        # Parse failed; fall back to lexical similarity
        return rouge_l_f1(code_a, code_b)
    
    sig_a = _ast_node_signatures(tree_a)
    sig_b = _ast_node_signatures(tree_b)
    
    # Cosine similarity over node-type counts
    keys = set(sig_a) | set(sig_b)
    if not keys:
        return 1.0
    
    dot = sum(sig_a[k] * sig_b[k] for k in keys)
    norm_a = sum(v * v for v in sig_a.values()) ** 0.5
    norm_b = sum(v * v for v in sig_b.values()) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def code_similarity(text_a: str, text_b: str, language: str = "python") -> float:
    """Compute similarity between two code-containing responses.
    
    Process:
    1. Extract code blocks
    2. If both extracted and language=python, use AST similarity
    3. Otherwise, fall back to lexical similarity on raw text
    """
    code_a = extract_code_block(text_a, language)
    code_b = extract_code_block(text_b, language)
    
    if code_a and code_b and language == "python":
        return python_ast_similarity(code_a, code_b)
    
    # Fallback: lexical
    return rouge_l_f1(code_a or text_a, code_b or text_b)