"""L2 Step-level Router: the decision brain of CHR-CP.

Coupled to L3 via:
- ESCALATE branch invokes l3.handoff() for cache-preserved switching
- All API calls record cache events into budget_tracker for CA²R feedback

For each agent step:
1. Get primary response (current tier)
2. Compute VC² uncertainty
3. Apply verbalized-failure strategy
4. Compute CA²R-adjusted thresholds (uses target_tier cache state)
5. Decide STAY / BRANCH / ESCALATE
6. Execute action (ESCALATE → L3 handoff with M2/M3 effects)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import time

from loguru import logger

from chr_cp.clients import ClientPool, Tier
from chr_cp.clients.base_client import CompletionResponse
from chr_cp.confidence.vc2 import VC2Estimator, UncertaintySignal
from chr_cp.confidence.consistency import TaskType
from chr_cp.routing.decisions import RoutingAction, RoutingDecision, StepResult
from chr_cp.routing.budget import BudgetTracker, AdaptiveThresholds
from chr_cp.routing.l3_cache import L3CacheManager


DEFAULT_TIER_LADDER = ["T1", "T2", "T3", "T4"]


@dataclass
class L2Config:
    tau_low: float = 0.30
    tau_high: float = 0.65
    alpha: float = 0.6
    branch_k: int = 3
    
    verbalized_failure_strategy: str = "consistency_only"
    tier_ladder: list[str] = field(default_factory=lambda: DEFAULT_TIER_LADDER.copy())
    
    enable_branch_cascade: bool = True
    cascade_threshold: float = 0.3
    
    max_actions_per_step: int = 5
    
    # NEW: For ablation, allow disabling BRANCH (collapse to ESCALATE)
    disable_branch: bool = False
    # NEW: VC² signal configuration
    alpha: float = 0.8           # U = alpha * U_cons + (1-alpha) * U_verb
    k_samples: int = 3           # K for consistency sampling

class L2Router:
    """L2 step-level router with CA²R thresholds and L3-coupled ESCALATE."""
    
    def __init__(
        self,
        pool: ClientPool,
        vc2_estimator: VC2Estimator,
        budget: BudgetTracker,
        config: Optional[L2Config] = None,
        cost_tracker=None,
        l3_manager: Optional[L3CacheManager] = None,  # NEW: for cache-preserved ESCALATE
        history_provider: Optional[Callable[[], str]] = None,  # NEW: for M2 to get history
    ):
        """
        Args:
            pool: Client pool
            vc2_estimator: VC² uncertainty estimator
            budget: Budget tracker (must be the same instance shared with L3)
            config: L2 configuration
            cost_tracker: Optional cost tracking
            l3_manager: L3 cache manager. If provided, ESCALATE invokes
                        l3_manager.handoff() instead of plain pool.invoke().
            history_provider: Callable returning current task's history text
                              (used by L3-M2 distillation). If None, no
                              distillation will be triggered.
        """
        self.pool = pool
        self.vc2 = vc2_estimator
        self.budget = budget
        self.config = config or L2Config()
        self.cost_tracker = cost_tracker
        self.l3 = l3_manager
        self.history_provider = history_provider
        self.diff_probe = None  # set by orchestrator for difficulty-aware routing

        for tier in self.config.tier_ladder:
            if tier not in self.pool.list_tiers():
                logger.warning(
                    f"Tier {tier} in ladder but not in pool. "
                    f"Available: {self.pool.list_tiers()}"
                )
    
    # -------- Public API --------
    
    def execute_step(
        self,
        messages: list[dict],
        current_tier: str,
        task_type: TaskType,
        task_id: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> StepResult:
        """Execute one L2-routed step."""
        result = StepResult(
            final_response=None,  # type: ignore
            final_tier=current_tier,
        )
        
        # === Step A: Primary call ===
        primary = self._invoke(
            tier=current_tier,
            messages=messages,
            task_id=task_id,
            step_id="primary",
            max_tokens=max_tokens,
        )
        result.all_responses.append(primary)
        result.total_cost_usd += primary.cost_usd
        result.total_latency_seconds += primary.latency_seconds
        
        # === Step B: VC² uncertainty ===
        # Cap consistency max_tokens: T3/T4 consistency needs comparable answers,
        # not full 16K reasoning. 4096 gives enough room for thinking+answer.
        consistency_max = max_tokens
        if current_tier in ("T3", "T4"):
            consistency_max = min(max_tokens, 4096)
        signal = self.vc2.estimate(
            primary_response=primary,
            messages=messages,
            tier=current_tier,
            task_type=task_type,
            mode="full",
            max_tokens=consistency_max,
            task_id=task_id,
        )
        result.final_uncertainty = signal

        # Record consistency sampling calls for full traceability
        if signal.consistency is not None and signal.consistency.responses:
            for resp in signal.consistency.responses:
                result.all_responses.append(resp)
                result.total_cost_usd += resp.cost_usd
                result.total_latency_seconds += resp.latency_seconds
                # Feed consistency cache hits into CA²R budget tracker
                if self.budget is not None and resp.cache_hit_tokens is not None:
                    self.budget.record_cache_event(
                        tier=resp.tier_name,
                        cache_hit_tokens=resp.cache_hit_tokens,
                        total_input_tokens=resp.prompt_tokens,
                    )

        # === Step C: Apply verbalized-failure strategy ===
        u_effective, vc_mode_used = self._apply_failure_strategy(signal)
        
        # === Step D: CA²R-adjusted thresholds ===
        # Use the next-tier-up as target (the tier we'd ESCALATE to)
        target_tier_for_adjust = self._next_tier_above(current_tier)
        thresholds = self.budget.adjust_thresholds(
            tau_low=self.config.tau_low,
            tau_high=self.config.tau_high,
            current_tier=current_tier,  
            target_tier=target_tier_for_adjust,
        )

        # === Step D2: Difficulty-adaptive thresholds (overlays on CA²R) ===
        if self.diff_probe:
            score = self.diff_probe.difficulty_score
            if score >= 6:
                thresholds.tau_low = min(thresholds.tau_low, 0.03)
                thresholds.tau_high = max(thresholds.tau_high, 0.60)
            elif score >= 4:
                thresholds.tau_low = min(thresholds.tau_low, 0.05)
                thresholds.tau_high = max(thresholds.tau_high, 0.60)

        # === Step D3: T1 cross-check (auxiliary signal for STAY) ===
        t1_cross = "skipped"
        if (self.diff_probe and self.diff_probe.t1_tentative_answer
                and signal.consistency is not None
                and signal.consistency.samples):
            from chr_cp.benchmarks.answer_verify import AnswerVerifier
            # Lightweight check: does tier answer agree with T1 guess?
            tier_answer = signal.consistency.samples[0]  # primary answer text
            try:
                v = getattr(self, '_cross_verifier', None)
                if v is None:
                    v = AnswerVerifier(judge_pool=None, use_llm=False)
                    self._cross_verifier = v
                result_cross = v.verify(
                    pred=tier_answer, ref=self.diff_probe.t1_tentative_answer
                )
                if result_cross.correct:
                    t1_cross = "agree"
                    u_effective = max(0.0, u_effective - 0.05)  # slight STAY boost
                else:
                    t1_cross = "disagree"
                    u_effective = min(1.0, u_effective + 0.10)  # slight ESCALATE push
            except Exception:
                pass

        # === Step E: Decision ===
        decision = self._decide(
            u_effective=u_effective,
            thresholds=thresholds,
            current_tier=current_tier,
            signal=signal,
            vc_mode_used=vc_mode_used,
            cascade_check=signal.consistency,
        )
        result.decisions.append(decision)
        
        # === Step F: Execute action ===
        if decision.action == RoutingAction.STAY:
            result.final_response = primary
            result.final_tier = current_tier
        elif decision.action == RoutingAction.BRANCH:
            branch_response = self._aggregate_branch_samples(
                primary=primary,
                signal=signal,
                tier=current_tier,
                messages=messages,
                task_type=task_type,
                task_id=task_id,
            )
            result.final_response = branch_response
            result.final_tier = current_tier
        elif decision.action == RoutingAction.ESCALATE:
            target_tier = decision.target_tier
            escalated = self._do_escalate(
                target_tier=target_tier,
                messages=messages,
                current_tier=current_tier,
                task_id=task_id,
                max_tokens=max_tokens,
                current_response_text=primary.content,
            )
            result.all_responses.append(escalated)
            result.total_cost_usd += escalated.cost_usd
            result.total_latency_seconds += escalated.latency_seconds
            result.final_response = escalated
            result.final_tier = target_tier
        
        # Record total cost into budget
        self.budget.add_cost(result.total_cost_usd)
        
        # CA²R feedback: record cache events for ALL responses in this step
        for resp in result.all_responses:
            self.budget.record_cache_event(
                tier=resp.tier_name,
                cache_hit_tokens=resp.cache_hit_tokens or 0,
                total_input_tokens=resp.prompt_tokens,
            )
        
        logger.info(f"L2 step complete: {result.summary()}")
        return result
    
    # -------- Internal --------
    
    def _invoke(
        self,
        tier: str,
        messages: list[dict],
        task_id: Optional[str],
        step_id: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> CompletionResponse:
        resp = self.pool.invoke(
            tier=tier,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self.cost_tracker:
            self.cost_tracker.record(resp, task_id=task_id, step_id=step_id)
        # Feed cache hit data into CA²R budget tracker for adaptive thresholds
        if self.budget is not None:
            self.budget.record_cache_event(
                tier=tier,
                cache_hit_tokens=resp.cache_hit_tokens or 0,
                total_input_tokens=resp.prompt_tokens,
            )
        return resp
    
    def _do_escalate(
        self,
        target_tier: str,
        messages: list[dict],
        current_tier: str,
        task_id: Optional[str],
        max_tokens: int,
        current_response_text: Optional[str] = None,
    ) -> CompletionResponse:
        """Execute ESCALATE: route through L3 if available, else direct invoke."""
        if self.l3 is None:
            return self._invoke(
                tier=target_tier,
                messages=messages,
                task_id=task_id,
                step_id="escalate",
                max_tokens=max_tokens,
            )

        # With L3 manager: cache-preserved handoff + CTOR
        history_text = self.history_provider() if self.history_provider else None
        handoff = self.l3.handoff(
            messages=messages,
            current_tier=current_tier,
            target_tier=target_tier,
            history_text=history_text,
            task_id=task_id,
            max_tokens=max_tokens,
            current_response_text=current_response_text,
        )
        return handoff.target_response
    
    def _apply_failure_strategy(
        self,
        signal: UncertaintySignal,
    ) -> tuple[float, str]:
        if signal.verbalized is None:
            return signal.U, "lite"
        if signal.verbalized.parse_success:
            return signal.U, "full"
        if self.config.verbalized_failure_strategy == "consistency_only":
            return signal.U_consistency, "consistency_only"
        else:
            return signal.U, "neutral_fallback"
    
    def _decide(
        self,
        u_effective: float,
        thresholds: AdaptiveThresholds,
        current_tier: str,
        signal: UncertaintySignal,
        vc_mode_used: str,
        cascade_check=None,
    ) -> RoutingDecision:
        target_tier = self._next_tier_above(current_tier)
        can_escalate = target_tier is not None

        # === Difficulty-aware: skip escalation (U ≥ 0.8 → jump to T4) ===
        if u_effective >= 0.8 and can_escalate:
            # Jump directly to T4: current tier is completely lost
            tier_ladder = self.config.tier_ladder
            if "T4" in tier_ladder:
                target_tier = "T4"
                can_escalate = True

        # === Difficulty-aware: BRANCH gating ===
        disable_branch = self.config.disable_branch
        if not disable_branch and self.diff_probe:
            if self.diff_probe.difficulty_score >= 6:
                disable_branch = True

        cascade_triggered = False
        if (
            self.config.enable_branch_cascade
            and cascade_check is not None
            and cascade_check.mean_similarity < self.config.cascade_threshold
            and can_escalate
        ):
            cascade_triggered = True
        
        if u_effective < thresholds.tau_low:
            action = RoutingAction.STAY
            reason = f"U={u_effective:.3f} < τ_low={thresholds.tau_low:.3f}"
            target = None
        elif u_effective < thresholds.tau_high:
            if cascade_triggered:
                action = RoutingAction.ESCALATE
                reason = (
                    f"BRANCH→ESCALATE cascade: samples disagreed "
                    f"(mean_sim={cascade_check.mean_similarity:.3f} "
                    f"< {self.config.cascade_threshold})"
                )
                target = target_tier
            elif disable_branch:
                # Ablation: collapse BRANCH zone into ESCALATE
                if can_escalate:
                    action = RoutingAction.ESCALATE
                    reason = (
                        f"BRANCH disabled (ablation), τ_low ≤ U < τ_high "
                        f"→ forced ESCALATE"
                    )
                    target = target_tier
                else:
                    action = RoutingAction.STAY
                    reason = "BRANCH disabled and no upper tier; falling back to STAY"
                    target = None
            else:
                action = RoutingAction.BRANCH
                reason = (
                    f"τ_low={thresholds.tau_low:.3f} ≤ U={u_effective:.3f} "
                    f"< τ_high={thresholds.tau_high:.3f}"
                )
                target = None
        else:
            if can_escalate:
                action = RoutingAction.ESCALATE
                reason = f"U={u_effective:.3f} ≥ τ_high={thresholds.tau_high:.3f}"
                target = target_tier
            else:
                action = RoutingAction.BRANCH
                reason = (
                    f"U={u_effective:.3f} ≥ τ_high but already at top tier "
                    f"{current_tier}, falling back to BRANCH"
                )
                target = None
        
        return RoutingDecision(
            action=action,
            current_tier=current_tier,
            target_tier=target,
            uncertainty=u_effective,
            threshold_low=thresholds.tau_low,
            threshold_high=thresholds.tau_high,
            u_verbalized=signal.U_verbalized,
            u_consistency=signal.U_consistency,
            reason=reason,
            vc_mode=vc_mode_used,
            verbalized_parsed=(
                signal.verbalized.parse_success if signal.verbalized else False
            ),
            budget_remaining_ratio=self.budget.remaining_ratio,
            # NEW: CA²R diagnostics
            ca2r_budget_factor=thresholds.budget_factor,
            ca2r_cache_factor=(
                thresholds.cache_factor_low if action == RoutingAction.STAY
                else thresholds.cache_factor_high
            ),
            ca2r_h_target=thresholds.h_target_observed,
            ca2r_target_tier=thresholds.target_tier_used,
            ca2r_cache_factor_low=thresholds.cache_factor_low,
            ca2r_cache_factor_high=thresholds.cache_factor_high,
            ca2r_h_current=thresholds.h_current_observed,
            ca2r_current_tier=thresholds.current_tier_used,
        )
    
    def _next_tier_above(self, current_tier: str) -> Optional[str]:
        ladder = self.config.tier_ladder
        try:
            idx = ladder.index(current_tier)
        except ValueError:
            logger.warning(f"current_tier {current_tier} not in ladder {ladder}")
            return None
        if idx + 1 >= len(ladder):
            return None
        return ladder[idx + 1]
    
    def _aggregate_branch_samples(
        self,
        primary: CompletionResponse,
        signal: UncertaintySignal,
        tier: str,
        messages: list[dict],
        task_type: TaskType,
        task_id: Optional[str],
    ) -> CompletionResponse:
        if signal.consistency is None or len(signal.consistency.samples) < 2:
            return primary
        
        cons = signal.consistency
        
        if task_type in (TaskType.NUMERIC, TaskType.MULTIPLE_CHOICE):
            from collections import Counter
            votes = Counter()
            for ans in cons.extracted_answers:
                if ans is not None:
                    votes[ans] += 1
            if votes:
                winning = votes.most_common(1)[0][0]
                for i, ans in enumerate(cons.extracted_answers):
                    if ans == winning:
                        return self._wrap_sample_as_response(cons.samples[i], primary)
            return primary
        
        n = len(cons.samples)
        if n < 2:
            return primary
        
        from chr_cp.utils.text_sim import rouge_l_f1
        from chr_cp.utils.ast_diff import code_similarity
        import re as _re

        # For CODE task: prefer samples that contain a code block
        if task_type == TaskType.CODE:
            code_block_pattern = _re.compile(r"```(?:python)?\s*", _re.IGNORECASE)
            samples_with_code = [
                i for i, s in enumerate(cons.samples)
                if code_block_pattern.search(s) or "def " in s
            ]
            primary_has_code = (
                code_block_pattern.search(primary.content)
                or "def " in primary.content
            )

            if not samples_with_code and primary_has_code:
                # All branch samples lost the code; fall back to primary
                return primary

            if not samples_with_code:
                # No code anywhere; broken step, return primary anyway
                return primary

            # Pick the most consistent sample among those with code
            if len(samples_with_code) == 1:
                return self._wrap_sample_as_response(
                    cons.samples[samples_with_code[0]], primary
                )

            sim_func = lambda a, b: code_similarity(a, b, language="python")
            scores = {i: 0.0 for i in samples_with_code}
            for i in samples_with_code:
                for j in samples_with_code:
                    if i != j:
                        scores[i] += sim_func(cons.samples[i], cons.samples[j])
            best_idx = max(scores, key=lambda k: scores[k])
            return self._wrap_sample_as_response(cons.samples[best_idx], primary)

        # OPEN_TEXT: original ROUGE-L logic
        sim_func = rouge_l_f1
        scores = [0.0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    scores[i] += sim_func(cons.samples[i], cons.samples[j])

        best_idx = max(range(n), key=lambda k: scores[k])
        return self._wrap_sample_as_response(cons.samples[best_idx], primary)
    
    def _wrap_sample_as_response(
        self,
        sample_text: str,
        template: CompletionResponse,
    ) -> CompletionResponse:
        from dataclasses import replace
        return replace(template, content=sample_text)