"""L3: Cache-Preserved Switching with CADS + CTOR.

Three sub-mechanisms:
- M1 (Stable Prefix): handled by chr_cp.prompts.stable_prefix
- M2 (Cross-Vendor Distillation) with CADS — Cost-Adaptive Distillation Strategy
- M3 (Cross-Tier Output Reuse) — CTOR: extract structured handoff from current
  tier output and pass to target tier as verifier-corrector task

CADS: distillation tier dynamically selected based on history length:
    history < threshold → cheaper tier (T1)
    history ≥ threshold → stronger tier (T4) for reliability

CTOR: when ESCALATE, extract candidate_answer, confidence, and key context
    from the current tier's response. Build a structured handoff block and
    inject it into the target tier's prompt, so the target acts as a
    verifier-corrector rather than starting from scratch.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re
import time

from loguru import logger

from chr_cp.clients import ClientPool, Tier
from chr_cp.clients.base_client import CompletionResponse
from chr_cp.prompts.distillation import (
    build_distillation_messages,
    parse_distilled,
    estimate_tokens,
    DistilledContext,
)
from chr_cp.prompts.handoff import (
    EscalationReason,
    HandoffPacket,
    parse_handoff_xml,
    truncate_packet,
    format_handoff_for_prompt,
)
from chr_cp.prompts.role_templates import COMPRESSOR_CTOR_PROMPT


@dataclass
class L3Config:
    """Configuration for L3 cache mechanisms (with CADS + UD-SCW)."""
    
    # === M1: Stable Prefix ===
    enable_m1_stable_prefix: bool = True   # actually controlled in prompts.stable_prefix
    
    # === M2: Distillation (CADS) ===
    enable_m2_distillation: bool = True
    
    # CADS tier selection
    distillation_mode: str = "adaptive"
    # "adaptive" : T2 if history<threshold else T4
    # "fixed_t1" / "fixed_t2" / "fixed_t4" : for ablation
    distillation_tier_short: str = "T1"
    distillation_tier_long: str = "T1"
    distillation_length_threshold: int = 3000  # tokens
    distillation_min_tokens: int = 500          # skip distillation below this
    
    # === M3: Cross-Tier Output Reuse (CTOR) ===
    enable_m3_ctor: bool = True
    ctor_mode: str = "self_compress"
        # "self_compress": T_k outputs <handoff> block inline (preferred)
        # "external_compress": external T1 qwen-turbo compressor
        # "raw_passthrough": no compression (ablation baseline)
        # "off": no CTOR (target starts from scratch)
    ctor_compressor_tier: str = "T1"
    ctor_max_handoff_tokens: int = 400
    ctor_target_role: str = "verifier_corrector"
        # "verifier_corrector" or "fresh_solver"
    # === Provider mapping ===
    tier_to_provider: dict[str, str] = field(
        default_factory=lambda: {
            "T1": "qwen",       # qwen-turbo
            "T2": "deepseek",
            "T3": "deepseek",
            "T4": "openai",
        }
    )
    
    def select_distillation_tier(self, history_tokens: int) -> str:
        """CADS: select distillation tier based on history length."""
        if self.distillation_mode == "fixed_t1":
            return "T1"
        if self.distillation_mode == "fixed_t2":
            return "T2"
        if self.distillation_mode == "fixed_t4":
            return "T4"
        # adaptive (default)
        if history_tokens < self.distillation_length_threshold:
            return self.distillation_tier_short
        return self.distillation_tier_long


@dataclass
class HandoffResult:
    """Result of a cache-preserved tier handoff."""
    
    target_response: CompletionResponse
    distilled: Optional[DistilledContext] = None
    distillation_tier_used: Optional[str] = None
    ctor_used: bool = False
    handoff_overhead_seconds: float = 0.0
    distillation_cost_usd: float = 0.0


class L3CacheManager:
    """L3 cache-preserved switching manager (CADS + CTOR)."""

    def __init__(
        self,
        pool: ClientPool,
        config: Optional[L3Config] = None,
        cost_tracker=None,
        budget_tracker=None,
    ):
        self.pool = pool
        self.config = config or L3Config()
        self.cost_tracker = cost_tracker
        self.budget_tracker = budget_tracker

        logger.info(
            f"L3CacheManager: M1={self.config.enable_m1_stable_prefix}, "
            f"M2={self.config.enable_m2_distillation} (mode={self.config.distillation_mode}), "
            f"M3(CTOR)={self.config.enable_m3_ctor}"
        )
    
    # -------- Cross-vendor detection --------
    
    def is_cross_vendor(self, tier_a: str, tier_b: str) -> bool:
        prov_a = self.config.tier_to_provider.get(tier_a)
        prov_b = self.config.tier_to_provider.get(tier_b)
        if prov_a is None or prov_b is None:
            logger.warning(
                f"Unknown provider for {tier_a} or {tier_b}; assuming cross-vendor"
            )
            return True
        return prov_a != prov_b
    
    # -------- M3: CTOR — Cross-Tier Output Reuse --------

    def build_handoff_packet(
        self,
        current_response: CompletionResponse,
        source_tier: str,
        target_tier: str,
        escalation_reason: EscalationReason,
    ) -> HandoffPacket:
        """Build a structured handoff packet from T_k output."""
        raw_content = current_response.content
        raw_token_count = estimate_tokens(raw_content)

        mode = self.config.ctor_mode
        if mode == "self_compress":
            packet = parse_handoff_xml(raw_content)
            if packet is not None:
                packet.raw_token_count = raw_token_count
            else:
                logger.debug("self_compress parse failed, falling back to heuristic")
                packet = self._heuristic_handoff(raw_content, raw_token_count)
        elif mode == "external_compress":
            packet = self._external_compress(raw_content, source_tier, target_tier, raw_token_count)
        elif mode == "raw_passthrough":
            packet = HandoffPacket(
                approach="(not compressed)",
                candidate_answer=self._extract_boxed(raw_content) or "",
                confidence=0.5,
                escalation_reason=escalation_reason,
                target_action_hint="Review full prior solution.",
                raw_token_count=raw_token_count,
                packet_token_count=raw_token_count,
            )
        else:
            return None  # mode == "off"

        # Safety net
        if not packet.candidate_answer:
            packet.candidate_answer = self._extract_boxed(raw_content) or ""

        # Hard cap
        packet = truncate_packet(packet, self.config.ctor_max_handoff_tokens)
        packet.source_tier = source_tier
        packet.target_tier = target_tier
        packet.escalation_reason = escalation_reason
        return packet

    def _heuristic_handoff(self, raw_content: str, raw_token_count: int) -> HandoffPacket:
        """Heuristic extraction when self_compress parse fails."""
        packet = HandoffPacket(raw_token_count=raw_token_count)
        packet.candidate_answer = self._extract_boxed(raw_content) or ""
        m = re.search(r'<confidence>\s*(\d+(?:\.\d+)?)\s*/\s*10\s*</confidence>', raw_content)
        if m:
            packet.confidence = float(m.group(1)) / 10.0
        tail = raw_content[-500:]
        tail = re.sub(r'<confidence>.*?</confidence>', '', tail, flags=re.DOTALL).strip()
        packet.stuck_at = tail[:100]
        return packet

    @staticmethod
    def _extract_boxed(text: str) -> Optional[str]:
        m = re.search(r'\\boxed\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', text)
        return m.group(1).strip() if m else None

    def _external_compress(
        self, raw_content: str, source_tier: str, target_tier: str, raw_token_count: int
    ) -> HandoffPacket:
        """Use T1 qwen-turbo to compress CoT into handoff packet."""
        compressor_prompt = COMPRESSOR_CTOR_PROMPT.format(cot=raw_content[:8000])
        try:
            response = self.pool.invoke(
                tier=self.config.ctor_compressor_tier,
                messages=[{"role": "user", "content": compressor_prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            packet = parse_handoff_xml(response.content)
            if packet is not None:
                packet.raw_token_count = raw_token_count
                packet.packet_token_count = estimate_tokens(response.content)
                return packet
        except Exception as e:
            logger.warning(f"CTOR external compressor failed: {e}")

        # Fallback to heuristic
        logger.debug("external_compress failed, falling back to heuristic")
        return self._heuristic_handoff(raw_content, raw_token_count)

    # -------- Reasoning effort & output cap decisions --------

    @staticmethod
    def decide_reasoning_effort(
        target_tier: str,
        escalation_reason: EscalationReason,
        packet: Optional[HandoffPacket] = None,
    ) -> str:
        """Decide target tier reasoning effort based on escalation reason."""
        if escalation_reason == EscalationReason.LOW_CONFIDENCE and packet and packet.confidence >= 0.5:
            return "low"  # DeepSeek accepts: high/low/medium/max/xhigh
        if escalation_reason == EscalationReason.COMPUTE_ERROR:
            return "low"
        if escalation_reason == EscalationReason.STUCK_MIDWAY:
            return "medium"
        if escalation_reason == EscalationReason.WRONG_APPROACH:
            return "high"
        if packet and packet.confidence < 0.3:
            return "medium"
        return "medium"

    @staticmethod
    def decide_max_tokens(
        escalation_reason: EscalationReason,
        target_role: str = "verifier_corrector",
    ) -> int:
        """Decode output token cap based on scenario.

        Note: DeepSeek thinking mode consumes a large portion of max_tokens
        for internal reasoning. Floor raised to 2048 to leave room for
        visible output after thinking tokens are consumed.
        """
        if target_role == "verifier_corrector":
            if escalation_reason == EscalationReason.LOW_CONFIDENCE:
                return 2048  # was 512 — too small for thinking mode overhead
            if escalation_reason == EscalationReason.COMPUTE_ERROR:
                return 2048
            if escalation_reason == EscalationReason.STUCK_MIDWAY:
                return 2048
        if target_role == "fresh_solver":
            return 4096
        return 2048

    # -------- Target messages construction --------

    @staticmethod
    def build_target_messages(
        base_messages: list[dict],
        handoff_packet: Optional[HandoffPacket],
        distilled_history: Optional[DistilledContext],
        target_role: str,
        escalation_reason: EscalationReason,
    ) -> list[dict]:
        """Construct target tier messages with CTOR handoff + M2 distillation."""
        user_parts = []

        # Problem
        problem_msg = base_messages[1] if len(base_messages) >= 2 else base_messages[0]
        user_parts.append("<problem>")
        if problem_msg["role"] == "user":
            user_parts.append(problem_msg["content"])
        user_parts.append("</problem>")

        # M2: distilled history
        if distilled_history is not None:
            user_parts.append("<prior_context>")
            user_parts.append(distilled_history.to_text())
            user_parts.append("</prior_context>")

        # M3: CTOR handoff
        if handoff_packet is not None:
            user_parts.append("<predecessor_handoff>")
            user_parts.append(format_handoff_for_prompt(handoff_packet))
            user_parts.append("</predecessor_handoff>")

        # Answer-first directive
        user_parts.append(_build_answer_first_directive(handoff_packet, escalation_reason))

        return [
            {"role": "system", "content": base_messages[0]["content"]},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    # -------- M2: Distillation handoff (CADS) --------
    
    def handoff(
        self,
        messages: list[dict],
        current_tier: str,
        target_tier: str,
        history_text: Optional[str] = None,
        task_id: Optional[str] = None,
        max_tokens: int = 2048,
        current_response_text: Optional[str] = None,
    ) -> HandoffResult:
        """L3 handoff: switch from current_tier to target_tier with CTOR + CADS."""
        t_start = time.time()
        result = HandoffResult(target_response=None)  # type: ignore
        escalation_reason = EscalationReason.LOW_CONFIDENCE

        # === M3 (CTOR): Build structured handoff ===
        handoff_packet: Optional[HandoffPacket] = None

        if (self.config.enable_m3_ctor
                and self.config.ctor_mode != "off"
                and current_response_text):
            handoff_packet = self.build_handoff_packet(
                current_response=CompletionResponse(
                    content=current_response_text,
                    tier_name=current_tier, model_id="", provider="",
                ),
                source_tier=current_tier,
                target_tier=target_tier,
                escalation_reason=escalation_reason,
            )
            result.ctor_used = True

        # === M2 (CADS): Distillation (runs after CTOR, provides data only) ===
        history_token_estimate = estimate_tokens(history_text) if history_text else 0

        if (self.config.enable_m2_distillation
                and history_text is not None
                and history_token_estimate >= self.config.distillation_min_tokens):
            distill_tier = self.config.select_distillation_tier(history_token_estimate)
            logger.info(
                f"L3-M2 (CADS): distilling {history_token_estimate} tokens "
                f"using tier={distill_tier} for {current_tier}→{target_tier}"
            )
            distilled = self._distill_history(
                history_text=history_text,
                distillation_tier=distill_tier,
                task_id=task_id,
            )
            if distilled is not None:
                result.distilled = distilled
                result.distillation_tier_used = distill_tier
                logger.info(
                    f"L3-M2: compressed {history_token_estimate} → "
                    f"~{estimate_tokens(distilled.to_text())} tokens"
                )

        # === Build final target messages (CTOR + M2 combined) ===
        target_messages = self.build_target_messages(
            base_messages=messages,
            handoff_packet=handoff_packet,
            distilled_history=result.distilled,
            target_role=self.config.ctor_target_role,
            escalation_reason=escalation_reason,
        )

        # === Decide reasoning effort & output cap ===
        target_effort = self.decide_reasoning_effort(
            target_tier=target_tier,
            escalation_reason=escalation_reason,
            packet=handoff_packet,
        )
        target_max = self.decide_max_tokens(
            escalation_reason=escalation_reason,
            target_role=self.config.ctor_target_role,
        )
        max_tokens = min(max_tokens, target_max)

        # === Main target call ===
        invoke_kwargs: dict = {
            "tier": target_tier,
            "messages": target_messages,
            "max_tokens": max_tokens,
        }
        if result.ctor_used:
            invoke_kwargs["reasoning_effort"] = target_effort
        target_response = self.pool.invoke(**invoke_kwargs)
        result.target_response = target_response
        result.handoff_overhead_seconds = time.time() - t_start

        if self.cost_tracker:
            self.cost_tracker.record(
                target_response,
                task_id=task_id,
                step_id="l3_target",
                routing_action="ESCALATE",
            )

        # CA²R feedback
        if self.budget_tracker is not None:
            self.budget_tracker.record_cache_event(
                tier=target_tier,
                cache_hit_tokens=target_response.cache_hit_tokens or 0,
                total_input_tokens=target_response.prompt_tokens,
            )

        return result
    
    # -------- Internal --------
    
    def _distill_history(
        self,
        history_text: str,
        distillation_tier: str,
        task_id: Optional[str] = None,
    ) -> Optional[DistilledContext]:
        """Run distillation via CADS-selected tier."""
        try:
            messages = build_distillation_messages(history_text)
            response = self.pool.invoke(
                tier=distillation_tier,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
            )
            
            if self.cost_tracker:
                self.cost_tracker.record(
                    response,
                    task_id=task_id,
                    step_id=f"l3_m2_distill_{distillation_tier}",
                    routing_action="ESCALATE",
                )
            
            # Also feed back into budget tracker
            if self.budget_tracker is not None:
                self.budget_tracker.record_cache_event(
                    tier=distillation_tier,
                    cache_hit_tokens=response.cache_hit_tokens or 0,
                    total_input_tokens=response.prompt_tokens,
                )
            
            distilled = parse_distilled(response.content)
            if distilled is None:
                logger.warning(
                    f"L3-M2: distillation parse failed (tier={distillation_tier}); "
                    f"raw output: {response.content[:200]!r}"
                )
                return None
            
            distilled.original_tokens = estimate_tokens(history_text)
            distilled.compressed_tokens = estimate_tokens(distilled.to_text())
            return distilled
        except Exception as e:
            logger.warning(f"L3-M2 distillation failed: {e}")
            return None
    
    def _inject_distilled(
        self,
        original_messages: list[dict],
        distilled: DistilledContext,
    ) -> list[dict]:
        """Replace verbose history in messages with distilled summary.
        
        Strategy: keep system message (cacheable prefix) intact,
        replace user/assistant turns with one user message containing distilled.
        """
        new_messages = []
        for msg in original_messages:
            if msg["role"] == "system":
                new_messages.append(msg)
        
        new_messages.append({
            "role": "user",
            "content": (
                f"=== PRIOR CONTEXT (compressed) ===\n{distilled.to_text()}\n\n"
                f"=== YOUR TASK ===\nContinue from the current question above."
            ),
        })
        return new_messages
    


def _build_answer_first_directive(
    packet: Optional[HandoffPacket],
    escalation_reason: EscalationReason,
) -> str:
    """Build answer-first behavioral constraints for target tier."""
    parts = []

    if packet and packet.confirmed_facts:
        parts.append("Take these as verified axioms (DO NOT re-derive):")
        for f in packet.confirmed_facts[:5]:
            parts.append(f"  - {f}")

    if packet and escalation_reason in (EscalationReason.LOW_CONFIDENCE, EscalationReason.COMPUTE_ERROR):
        parts.append(f"Focus only on: {packet.target_action_hint or 'verify the candidate answer'}")
    elif escalation_reason == EscalationReason.WRONG_APPROACH:
        parts.append("The previous approach was wrong. Use a DIFFERENT strategy.")
        parts.append("Do NOT repeat the predecessor's method.")
    elif escalation_reason == EscalationReason.STUCK_MIDWAY:
        parts.append(f"Continue from where predecessor stopped: {packet.target_action_hint if packet else ''}")

    parts.append("")
    parts.append("Output format (strict):")
    parts.append("\\boxed{answer}")
    parts.append("")
    parts.append("Do not include preamble, summary, or postamble.")

    return "\n".join(parts)