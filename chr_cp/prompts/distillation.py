"""L3 Mechanism 2: Cross-Vendor Context Distillation.

Prompt templates and helpers for compressing reasoning history when crossing
vendors (DeepSeek ↔ Qwen). The compression is performed by T1 (cheap) and
output is structured JSON consumable by the next-tier agent.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json
import re

from loguru import logger


# Distillation prompt template (paired with COMPRESSOR role from role_templates)
DISTILLATION_INSTRUCTION = """Compress the following reasoning history into a structured summary.

Output ONLY valid JSON in this exact format:
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
  "pending_question": "<the specific question that remains>"
}

Compression rules:
- Drop intermediate derivations, keep only verified results.
- Preserve all numbers, identifiers, and names exactly as written.
- If no verified facts exist, use empty array [].
- pending_question must be ONE sentence.

History to compress:
"""


@dataclass
class DistilledContext:
    """Parsed structured summary."""
    
    task_recap: str
    completed_steps: list[str]
    verified_facts: list[str]
    pending_question: str
    
    # Token count metrics (for paper analysis)
    original_tokens: int = 0
    compressed_tokens: int = 0
    
    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens
    
    def to_text(self) -> str:
        """Render as a clean text block for the next-tier agent."""
        lines = [
            f"Task: {self.task_recap}",
            "",
            "Completed steps:",
        ]
        for i, step in enumerate(self.completed_steps, 1):
            lines.append(f"  {i}. {step}")
        
        if self.verified_facts:
            lines.append("")
            lines.append("Verified facts:")
            for fact in self.verified_facts:
                lines.append(f"  - {fact}")
        
        lines.append("")
        lines.append(f"Current question: {self.pending_question}")
        return "\n".join(lines)


def build_distillation_messages(history_text: str) -> list[dict]:
    """Build the messages for a distillation call.
    
    Uses a simple structure since this is invoked on T1 (no need for
    fancy CHR-CP-style stable prefix here).
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a context compression assistant. "
                "You output ONLY valid JSON, no extra commentary, no markdown fences."
            ),
        },
        {
            "role": "user",
            "content": DISTILLATION_INSTRUCTION + history_text,
        },
    ]
def parse_distilled(text: str) -> Optional[DistilledContext]:
    """Parse the JSON output from the compressor agent.
    
    Robust to common errors:
    - JSON wrapped in ```json fences
    - Trailing commentary after the JSON
    - Missing optional fields
    - **Truncated JSON** (best-effort field recovery)
    """
    if not text:
        return None
    
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", text)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    
    # Find the first '{' 
    start = cleaned.find("{")
    if start == -1:
        logger.debug(f"Distillation parse: no '{{' in text: {text[:200]!r}")
        return None
    
    # Try strict JSON parse first
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    
    # === Path A: well-formed JSON ===
    if end != -1:
        json_str = cleaned[start : end + 1]
        try:
            data = json.loads(json_str)
            return DistilledContext(
                task_recap=data.get("task_recap", ""),
                completed_steps=list(data.get("completed_steps", [])),
                verified_facts=list(data.get("verified_facts", [])),
                pending_question=data.get("pending_question", ""),
            )
        except json.JSONDecodeError as e:
            logger.debug(f"Distillation strict JSON failed: {e}")
            # fall through to best-effort
    
    # === Path B: truncated/malformed JSON, best-effort field extraction ===
    logger.debug("Distillation: attempting best-effort field recovery from truncated JSON")
    return _best_effort_extract(cleaned[start:])


def _best_effort_extract(text: str) -> Optional[DistilledContext]:
    """Best-effort extraction of fields from possibly truncated JSON.
    
    Strategy: regex-extract each known field independently. Even if the
    closing brace is missing, fields written before truncation can be
    recovered. Returns None only if NO field could be extracted.
    """
    fields = {}
    
    # task_recap: "task_recap": "..."
    m = re.search(
        r'"task_recap"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text,
        flags=re.DOTALL,
    )
    if m:
        fields["task_recap"] = _unescape(m.group(1))
    
    # pending_question: "pending_question": "..."
    m = re.search(
        r'"pending_question"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text,
        flags=re.DOTALL,
    )
    if m:
        fields["pending_question"] = _unescape(m.group(1))
    
    # completed_steps: "completed_steps": [ ... ]
    fields["completed_steps"] = _extract_string_array(text, "completed_steps")
    
    # verified_facts: "verified_facts": [ ... ]
    fields["verified_facts"] = _extract_string_array(text, "verified_facts")
    
    # If we got nothing useful, give up
    has_any = (
        fields.get("task_recap")
        or fields.get("pending_question")
        or fields.get("completed_steps")
        or fields.get("verified_facts")
    )
    if not has_any:
        logger.debug("Distillation best-effort: no fields recoverable")
        return None
    
    logger.debug(
        f"Distillation best-effort recovered: "
        f"task_recap={'yes' if fields.get('task_recap') else 'no'}, "
        f"steps={len(fields.get('completed_steps', []))}, "
        f"facts={len(fields.get('verified_facts', []))}, "
        f"pending={'yes' if fields.get('pending_question') else 'no'}"
    )
    
    return DistilledContext(
        task_recap=fields.get("task_recap", ""),
        completed_steps=fields.get("completed_steps", []),
        verified_facts=fields.get("verified_facts", []),
        pending_question=fields.get("pending_question", ""),
    )


def _extract_string_array(text: str, key: str) -> list[str]:
    """Extract entries from a JSON string array, tolerating truncation.
    
    Looks for patterns like:
        "key": [
            "step 1",
            "step 2",
            ...
        ]
    
    Returns whatever complete strings can be parsed before truncation.
    """
    # Find the array opening
    pat = rf'"{re.escape(key)}"\s*:\s*\['
    m = re.search(pat, text)
    if not m:
        return []
    
    # Scan for string entries until we hit ']' or run out of text
    entries: list[str] = []
    i = m.end()
    while i < len(text):
        # Skip whitespace
        while i < len(text) and text[i] in " \t\n\r,":
            i += 1
        if i >= len(text):
            break
        # End of array?
        if text[i] == "]":
            break
        # Expect a string
        if text[i] != '"':
            break  # malformed; stop
        # Read string with escape handling
        j = i + 1
        buf = []
        complete = False
        while j < len(text):
            c = text[j]
            if c == "\\" and j + 1 < len(text):
                # escape sequence
                next_c = text[j + 1]
                if next_c == "n":
                    buf.append("\n")
                elif next_c == "t":
                    buf.append("\t")
                elif next_c == '"':
                    buf.append('"')
                elif next_c == "\\":
                    buf.append("\\")
                else:
                    buf.append(next_c)
                j += 2
                continue
            if c == '"':
                complete = True
                j += 1
                break
            buf.append(c)
            j += 1
        if complete:
            entries.append("".join(buf))
            i = j
        else:
            # truncated mid-string, stop
            break
    
    return entries


def _unescape(s: str) -> str:
    """Minimal JSON string unescape (only the basics)."""
    return (
        s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )



def estimate_tokens(text: str) -> int:
    """Rough token count (1 token ≈ 4 chars for English, 1.5 chars for Chinese).
    
    For precise counts, use the model's tokenizer; for cost estimation
    during distillation, this approximation is sufficient.
    """
    if not text:
        return 0
    # Mixed-language heuristic: take avg of English/Chinese rates
    return max(1, int(len(text) / 3))