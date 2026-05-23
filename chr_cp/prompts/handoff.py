"""CTOR (Cross-Tier Output Reuse) — structured handoff packet.

When T_k escalates, its reasoning trace is compressed into a HandoffPacket.
T_{k+1} receives this as <predecessor_handoff> and operates in a
verifier-corrector role rather than starting from scratch.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re

from chr_cp.prompts.distillation import estimate_tokens


class EscalationReason(str, Enum):
    COMPUTE_ERROR = "COMPUTE_ERROR"
    WRONG_APPROACH = "WRONG_APPROACH"
    STUCK_MIDWAY = "STUCK_MIDWAY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS_QUESTION = "AMBIGUOUS_QUESTION"


@dataclass
class HandoffPacket:
    """Structured handoff from T_k to T_{k+1}."""
    approach: str = ""
    confirmed_facts: list[str] = field(default_factory=list)
    candidate_answer: str = ""
    confidence: float = 0.5
    stuck_at: str = ""
    escalation_reason: Optional[EscalationReason] = None
    target_action_hint: str = ""

    # Metadata
    source_tier: str = ""
    target_tier: str = ""
    raw_token_count: int = 0
    packet_token_count: int = 0

    @property
    def compression_ratio(self) -> float:
        return self.raw_token_count / max(1, self.packet_token_count)


# ===== XML Parsing =====

_HANDOFF_RE = re.compile(r'<handoff>(.*?)</handoff>', re.DOTALL)
_FIELD_RE = re.compile(r'^(\w[\w\s]*?)\s*:\s*(.*)$')
_BULLET_RE = re.compile(r'^\s*[-*]\s+(.+)$')


def parse_handoff_xml(text: str) -> Optional[HandoffPacket]:
    """Parse <handoff>...</handoff> block from model output."""
    m = _HANDOFF_RE.search(text)
    if not m:
        return None
    block = m.group(1).strip()

    fields: dict[str, str] = {}
    facts: list[str] = []
    in_facts = False

    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Check for confirmed_facts section
        if 'confirmed_facts' in stripped.lower() and ':' in stripped:
            in_facts = True
            continue
        if in_facts:
            if ':' in stripped and not stripped.startswith('-'):
                in_facts = False  # next field
            else:
                bm = _BULLET_RE.match(stripped)
                if bm:
                    facts.append(bm.group(1).strip())
                continue
        fm = _FIELD_RE.match(stripped)
        if fm:
            key = fm.group(1).strip()
            val = fm.group(2).strip()
            fields[key] = val

    if 'candidate_answer' not in fields:
        return None

    conf = 0.5
    try:
        conf = float(fields.get('confidence', '0.5'))
    except ValueError:
        pass

    def _parse_reason(s: Optional[str]) -> Optional[EscalationReason]:
        if not s:
            return None
        s = s.strip().upper()
        for r in EscalationReason:
            if r.value in s:
                return r
        return None

    return HandoffPacket(
        approach=fields.get('approach', ''),
        confirmed_facts=facts,
        candidate_answer=fields['candidate_answer'],
        confidence=conf,
        stuck_at=fields.get('stuck_at', ''),
        escalation_reason=_parse_reason(fields.get('escalation_reason')),
        target_action_hint=fields.get('target_should_check', ''),
        packet_token_count=estimate_tokens(block),
    )


# ===== Truncation =====

_FIELD_PRIORITY = [
    "candidate_answer",
    "escalation_reason",
    "target_action_hint",
    "confirmed_facts",
    "stuck_at",
    "approach",
]


def truncate_packet(packet: HandoffPacket, max_tokens: int) -> HandoffPacket:
    """Hard-cap a handoff packet by progressively trimming low-priority fields."""
    current = estimate_tokens(format_handoff_for_prompt(packet))
    if current <= max_tokens:
        return packet

    # Trim confirmed_facts first
    while len(packet.confirmed_facts) > 2 and current > max_tokens:
        packet.confirmed_facts.pop()
        current = estimate_tokens(format_handoff_for_prompt(packet))

    # Trim stuck_at
    if current > max_tokens and len(packet.stuck_at) > 80:
        packet.stuck_at = packet.stuck_at[:80] + "..."
        current = estimate_tokens(format_handoff_for_prompt(packet))

    # Trim approach
    if current > max_tokens and len(packet.approach) > 60:
        packet.approach = packet.approach[:60] + "..."
        current = estimate_tokens(format_handoff_for_prompt(packet))

    return packet


# ===== Prompt formatting =====

def format_handoff_for_prompt(packet: HandoffPacket) -> str:
    """Serialize HandoffPacket into prompt-insertable XML block."""
    conf_str = f"{packet.confidence:.2f}"
    if packet.confidence < 0.3:
        conf_str = "uncertain (review carefully)"
    elif packet.confidence > 0.85:
        conf_str = "0.85"

    reason = packet.escalation_reason.value if packet.escalation_reason else "LOW_CONFIDENCE"

    facts_str = "\n".join(f"  - {f}" for f in packet.confirmed_facts) if packet.confirmed_facts else "  (none)"

    return (
        f"approach: {packet.approach}\n"
        f"confirmed_facts:\n{facts_str}\n"
        f"candidate_answer: {packet.candidate_answer}\n"
        f"confidence: {conf_str}\n"
        f"escalation_reason: {reason}\n"
        f"target_should_check: {packet.target_action_hint}\n"
    )
