"""End-to-end CHR-CP orchestrator: glues L1 + L2 + L3 together with CA²R+CADS+CTOR."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time

from loguru import logger

from chr_cp.clients import ClientPool
from chr_cp.clients.base_client import CompletionResponse
from chr_cp.confidence import VC2Estimator, TaskType
from chr_cp.prompts import StablePrefixBuilder
from chr_cp.prompts.role_templates import get_role
from chr_cp.routing.l1_coarse import L1Router, L1Config, AgentPoolConfig, TaskCategory
from chr_cp.routing.l2_step import L2Router, L2Config
from chr_cp.routing.l3_cache import L3CacheManager, L3Config
from chr_cp.routing.budget import BudgetTracker
from chr_cp.routing.decisions import StepResult, RoutingAction


@dataclass
class CHRCPResult:
    final_answer: str
    final_tier: str

    step_results: list[StepResult] = field(default_factory=list)
    l1_config: Optional[AgentPoolConfig] = None

    total_cost_usd: float = 0.0
    total_latency_seconds: float = 0.0
    total_calls: int = 0

    # L3 metrics
    distillations_triggered: int = 0

    # Action counts
    num_stay: int = 0
    num_branch: int = 0
    num_escalate: int = 0

    # CA²R diagnostics
    avg_cache_factor: float = 1.0

    # CTOR audit trail
    ctor_handoffs: list[dict] = field(default_factory=list)

    # Difficulty-aware routing audit fields
    routing_mode: str = "cascade"
    difficulty_probe: Optional[any] = None  # type: ignore
    start_tier: str = "?"
    escalation_jumps: list[dict] = field(default_factory=list)
    branch_disabled_reason: str = ""
    tau_low_used: float = 0.0
    t1_crosscheck: str = "skipped"

    def summary_line(self) -> str:
        return (
            f"[{self.l1_config.category.value if self.l1_config else '?'}] "
            f"calls={self.total_calls} cost=${self.total_cost_usd:.6f} "
            f"latency={self.total_latency_seconds:.1f}s "
            f"actions=(S{self.num_stay}/B{self.num_branch}/E{self.num_escalate}) "
            f"L3=(D{self.distillations_triggered}/CTOR{len(self.ctor_handoffs)}) "
            f"cache_factor={self.avg_cache_factor:.3f}"
        )


@dataclass
class OrchestratorConfig:
    l1_config: L1Config = field(default_factory=L1Config)
    l2_config: L2Config = field(default_factory=L2Config)
    l3_config: L3Config = field(default_factory=L3Config)
    
    per_task_budget_usd: float = 0.50
    max_tokens: int = 2048
    
    # Difficulty-aware routing
    enable_difficulty_routing: bool = True

    # CA²R parameters
    cache_sensitivity: float = 0.3   # β
    cache_window_size: int = 10


class CHRCPOrchestrator:
    """End-to-end CHR-CP runner with CA²R + CADS + UD-SCW."""
    
    def __init__(
        self,
        pool: ClientPool,
        config: Optional[OrchestratorConfig] = None,
        cost_tracker=None,
    ):
        self.pool = pool
        self.config = config or OrchestratorConfig()
        self.cost_tracker = cost_tracker
        
        self.l1 = L1Router(self.config.l1_config)
        self.vc2 = VC2Estimator(
            pool=pool,
            alpha=self.config.l2_config.alpha,
            k_samples=getattr(self.config.l2_config, 'k_samples', 3),
            cost_tracker=cost_tracker,
        )
        from chr_cp.routing.cache_history import SharedCacheHistory
        self.shared_cache_history = SharedCacheHistory(window_size=50)
        logger.info("CHRCPOrchestrator initialized (CA²R + CADS + CTOR)")
    
    def run(
        self,
        task: str,
        task_id: Optional[str] = None,
        benchmark: Optional[str] = None,
    ) -> CHRCPResult:
        t_start = time.time()
        result = CHRCPResult(final_answer="", final_tier="?")
        
        # === L1 ===
        l1_config = self.l1.classify_and_configure(task)
        result.l1_config = l1_config
        logger.info(f"[{task_id}] L1: {l1_config}")

        # === L1b: Difficulty probe (single T1 call, cost ~$0.00004) ===
        diff_probe = None
        start_tier_override = None
        if getattr(self.config, "enable_difficulty_routing", True):
            try:
                from chr_cp.routing.difficulty_probe import (
                    DIFFICULTY_PROBE_PROMPT, parse_probe_response, DifficultyProbeResult,
                )
                probe_prompt = DIFFICULTY_PROBE_PROMPT.format(problem=task[:3000])
                t0_probe = time.time()
                probe_resp = self.pool.invoke(
                    tier="T1",
                    messages=[{"role": "user", "content": probe_prompt}],
                    max_tokens=512,
                    temperature=0.1,
                )
                diff_probe = parse_probe_response(probe_resp.content)
                diff_probe.probe_cost_usd = probe_resp.cost_usd
                result.total_cost_usd += diff_probe.probe_cost_usd
                result.total_calls += 1
                logger.info(
                    f"[{task_id}] Difficulty probe: score={diff_probe.difficulty_score}, "
                    f"self={diff_probe.t1_self_assessment}, depth={diff_probe.reasoning_depth}, "
                    f"expert={diff_probe.needs_expert_knowledge}, "
                    f"direct={diff_probe.is_direct_candidate}"
                )
            except Exception as e:
                logger.warning(f"[{task_id}] Difficulty probe failed: {e}")

            # Direct dispatch: skip cascade, go straight to T4
            if diff_probe and diff_probe.is_direct_candidate:
                logger.info(f"[{task_id}] Direct dispatch to T4 (cannot_solve + deep)")
                try:
                    # Format instruction matching SingleTierBaseline per benchmark
                    fmt = {
                        "gpqa": "Respond with the letter (A-D) inside \\boxed{}, e.g., \\boxed{B}.",
                        "mmlu": "Respond with the letter (A-E) inside \\boxed{}, e.g., \\boxed{C}.",
                        "aime": "AIME answers are integers 0-999. Provide your answer in \\boxed{N}.",
                        "math": "End with \\boxed{ANSWER}.",
                        "humaneval": "Provide a complete Python implementation in ```python ... ``` block.",
                    }.get(benchmark or "", "")
                    messages = [
                        {"role": "system", "content": f"You are a careful problem solver. {fmt}"},
                        {"role": "user", "content": task},
                    ]
                    t4_resp = self.pool.invoke(tier="T4", messages=messages, max_tokens=16384)
                    result.total_cost_usd += t4_resp.cost_usd
                    result.total_calls += 1
                    result.total_latency_seconds = time.time() - t_start
                    result.final_answer = t4_resp.content
                    result.final_tier = "T4"
                    result.routing_mode = "direct"
                    result.difficulty_probe = diff_probe
                    result.start_tier = "T4"
                    return result
                except Exception as e:
                    logger.error(f"[{task_id}] Direct T4 failed: {e}, falling back to cascade")

            # Override start tier based on probe
            if diff_probe:
                start_tier_override = diff_probe.start_tier

        # === Setup per-task L2/L3/Budget with shared state ===
        budget = BudgetTracker(
            total_budget_usd=self.config.per_task_budget_usd,
            cache_sensitivity=self.config.cache_sensitivity,
            cache_window_size=self.config.cache_window_size,
            shared_history=self.shared_cache_history,  # NEW
        )
        l3 = L3CacheManager(
            pool=self.pool,
            config=self.config.l3_config,
            cost_tracker=self.cost_tracker,
            budget_tracker=budget,  # CA²R: shared budget tracker
        )
        
        # History provider for L2 → L3 M2 distillation
        prior_history: list[dict] = []
        
        def get_history_text() -> str:
            """Build a plain text representation of all prior agent outputs."""
            if not prior_history:
                return ""
            return "\n\n".join(
                f"[{m['role'].upper()}]: {m['content']}" for m in prior_history
            )
        
        l2 = L2Router(
            pool=self.pool,
            vc2_estimator=self.vc2,
            budget=budget,
            config=self.config.l2_config,
            cost_tracker=self.cost_tracker,
            l3_manager=l3,                       # NEW: pass L3 manager
            history_provider=get_history_text,   # NEW: history accessor for M2
        )
        
        # === Determine task type ===
        task_type = (
            TaskType.from_benchmark(benchmark)
            if benchmark
            else self._infer_task_type_from_l1(l1_config.category)
        )
        
        # === Build stable prefix (L3-M1) ===
        prefix_builder = StablePrefixBuilder(task=task)
        
        last_response: Optional[CompletionResponse] = None
        last_tier: str = "?"
        cache_factor_sum = 0.0
        decision_count = 0

        # Set audit fields
        result.difficulty_probe = diff_probe
        if start_tier_override:
            result.start_tier = start_tier_override
        # Pass probe to L2 for BRANCH gating, τ_low adaptation, cross-check
        l2.diff_probe = diff_probe

        for agent_idx, (role_name, tier) in enumerate(l1_config.agents):
            # Override first agent tier with difficulty probe result
            effective_tier = tier
            if agent_idx == 0 and start_tier_override:
                effective_tier = start_tier_override
                logger.info(f"[{task_id}] Start tier overridden: {tier} → {effective_tier}")

            role = get_role(role_name, ctor_mode=self.config.l3_config.ctor_mode)
            
            step_payload = self._build_step_payload(
                role_name=role_name,
                agent_idx=agent_idx,
                last_response=last_response,
                topology=l1_config.topology,
            )
            
            # Build messages with stable prefix
            # Rolling shared context: each agent only sees the immediately
            # preceding agent's output, not the full accumulated history.
            # This keeps SHARED-CONTEXT compact without affecting prefix cache.
            messages = prefix_builder.build_messages(
                role_prompt=role.build(),
                step_payload=step_payload,
                prior_history=prior_history[-1:] if l1_config.topology == "chain" else None,
                replace_shared=(agent_idx > 0 and l1_config.topology == "chain"),
            )
           # Tier inheritance with role-aware capability diversity:
            # - solver (idx=0): start from agent_pool tier (L1's task-aware default)
            # - verifier: always use ONE tier above the previous step's final_tier
            #             (ensures different model from solver → U signal works)
            # - escalator and beyond: inherit from previous step
            #             (preserves any ESCALATE that happened upstream)
            if agent_idx == 0:
                # Solver: use agent_pool tier
                inherited_tier = tier
            elif role_name == "verifier":
                # Verifier: force one tier above last to ensure capability diversity
                base_tier = last_tier if last_tier != "?" else tier
                ladder = self.config.l2_config.tier_ladder
                try:
                    base_idx = ladder.index(base_tier)
                    if base_idx + 1 < len(ladder):
                        inherited_tier = ladder[base_idx + 1]
                    else:
                        inherited_tier = base_tier  # already at top
                except ValueError:
                    inherited_tier = base_tier
            else:
                # escalator and others: pure inherit
                inherited_tier = last_tier if last_tier != "?" else tier
            # === L2 execution (with CA²R + L3-coupled ESCALATE) ===
            try:
                step_result = l2.execute_step(
                    messages=messages,
                    current_tier=inherited_tier,
                    task_type=task_type,
                    task_id=task_id,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as e:
                logger.error(f"[{task_id}] Agent {agent_idx} failed: {e}")
                break
            
            result.step_results.append(step_result)
            result.total_cost_usd += step_result.total_cost_usd
            result.total_latency_seconds += step_result.total_latency_seconds
            result.total_calls += len(step_result.all_responses)
            
            # Tally actions and CA²R diagnostics
            for d in step_result.decisions:
                if d.action == RoutingAction.STAY:
                    result.num_stay += 1
                elif d.action == RoutingAction.BRANCH:
                    result.num_branch += 1
                elif d.action == RoutingAction.ESCALATE:
                    result.num_escalate += 1
                    if d.target_tier and l3.is_cross_vendor(d.current_tier, d.target_tier):
                        result.distillations_triggered += 1
                    # Track skip-level jumps
                    from_tier_num = int(d.current_tier[1])
                    to_tier_num = int(d.target_tier[1]) if d.target_tier else from_tier_num + 1
                    result.escalation_jumps.append({
                        "from": d.current_tier, "to": d.target_tier,
                        "skipped": to_tier_num - from_tier_num > 1,
                        "U": d.uncertainty,
                    })
                    # CTOR audit
                    ctor_info = {
                        "step": agent_idx,
                        "source_tier": d.current_tier,
                        "target_tier": d.target_tier,
                        "ctor_mode": self.config.l3_config.ctor_mode,
                        "U": d.uncertainty,
                    }
                    result.ctor_handoffs.append(ctor_info)

                cache_factor_sum += d.ca2r_cache_factor
                decision_count += 1
            
            # === Early exit: skip remaining agents when verifier is very confident ===
            if (
                role_name == "verifier"
                and step_result.final_uncertainty is not None
                and step_result.final_uncertainty.U < self.config.l2_config.tau_low
                and step_result.final_response is not None
            ):
                remaining = len(l1_config.agents) - (agent_idx + 1)
                logger.info(
                    f"[{task_id}] Verifier U={step_result.final_uncertainty.U:.3f} < "
                    f"τ_low={self.config.l2_config.tau_low:.3f}, "
                    f"skipping {remaining} remaining agent(s)"
                )
                last_response = step_result.final_response
                last_tier = step_result.final_tier
                prior_history.append({
                    "role": "assistant",
                    "content": last_response.content,
                })
                break

            # === Update history ===
            last_response = step_result.final_response
            last_tier = step_result.final_tier
            prior_history.append({
                "role": "assistant",
                "content": last_response.content,
            })
        
        # === Final answer ===
        if last_response is not None:
            final_content = last_response.content

            # Last-resort fallback: if final answer doesn't contain expected format
            # for CODE tasks, fall back to an earlier agent's output that does
            if task_type == TaskType.CODE:
                if "```" not in final_content and "def " not in final_content:
                    for step in reversed(result.step_results[:-1]):
                        if (step.final_response
                            and ("```" in step.final_response.content
                                 or "def " in step.final_response.content)):
                            logger.warning(
                                f"[{task_id}] Final response missing code block; "
                                f"falling back to {step.final_tier} output"
                            )
                            final_content = step.final_response.content
                            last_tier = step.final_tier
                            break

            result.final_answer = final_content
            result.final_tier = last_tier
        
        result.total_latency_seconds = time.time() - t_start
        if decision_count > 0:
            result.avg_cache_factor = cache_factor_sum / decision_count
        
        logger.info(f"[{task_id}] CHR-CP done: {result.summary_line()}")
        return result
    
    # -------- Helpers --------
    
    def _build_step_payload(
        self,
        role_name: str,
        agent_idx: int,
        last_response: Optional[CompletionResponse],
        topology: str,
    ) -> str:
        if agent_idx == 0:
            return "Solve the task above. Show reasoning and give the final answer."
        
        if role_name == "verifier" and last_response:
            return (
                f"A previous agent provided this candidate solution:\n\n"
                f"--- BEGIN CANDIDATE ---\n{last_response.content}\n--- END CANDIDATE ---\n\n"
                f"Verify it. If correct, confirm. If wrong, identify the error "
                f"and give the corrected answer."
            )
        
        if role_name == "aggregator":
            return (
                f"Prior agents have produced candidate answers (last shown):\n\n"
                f"--- LAST CANDIDATE ---\n"
                f"{last_response.content if last_response else '(none)'}\n"
                f"--- END ---\n\n"
                f"Produce the FINAL synthesized answer. Output only the answer "
                f"in the format the task requires."
            )
        
        return (
            f"Continue the analysis. Previous step output:\n\n"
            f"{last_response.content if last_response else '(none)'}"
        )
    
    def _infer_task_type_from_l1(self, category: TaskCategory) -> TaskType:
        mapping = {
            TaskCategory.MATH: TaskType.NUMERIC,
            TaskCategory.CODE: TaskType.CODE,
            TaskCategory.KNOWLEDGE_QA: TaskType.MULTIPLE_CHOICE,
            TaskCategory.LOGICAL_REASONING: TaskType.OPEN_TEXT,
            TaskCategory.OPEN_TEXT: TaskType.OPEN_TEXT,
        }
        return mapping.get(category, TaskType.OPEN_TEXT)