"""Infer escalation reason from VC² signal + T_k response.

Used by CTOR to decide target tier's reasoning effort and role.
"""

from __future__ import annotations
from typing import Optional

from chr_cp.confidence.vc2 import UncertaintySignal
from chr_cp.prompts.handoff import EscalationReason, HandoffPacket


def infer_escalation_reason(
    uncertainty: UncertaintySignal,
    finish_reason: Optional[str] = None,
    packet_in_response: Optional[HandoffPacket] = None,
) -> EscalationReason:
    """Infer why escalation happened.

    Priority:
      1. T_k self-declared in <handoff> (if available)
      2. VC² signal heuristic
      3. Default LOW_CONFIDENCE
    """
    # Priority 1: self-declared
    if packet_in_response is not None and packet_in_response.escalation_reason is not None:
        return packet_in_response.escalation_reason

    # Priority 2: signal heuristic
    u_cons = uncertainty.U_consistency
    u_verb = uncertainty.U_verbalized

    if u_cons >= 0.8:
        return EscalationReason.WRONG_APPROACH

    if 0.4 <= u_cons <= 0.6 and u_verb >= 0.3:
        return EscalationReason.COMPUTE_ERROR

    if 0.2 <= u_cons <= 0.4 and u_verb < 0.3:
        return EscalationReason.LOW_CONFIDENCE

    if finish_reason == "length":
        return EscalationReason.STUCK_MIDWAY

    return EscalationReason.LOW_CONFIDENCE
