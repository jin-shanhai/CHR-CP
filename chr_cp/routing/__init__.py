"""CHR-CP routing modules: L1 (coarse), L2 (step-level), L3 (cache-preserved switch)."""

from chr_cp.routing.decisions import (
    RoutingAction,
    RoutingDecision,
    StepResult,
)
from chr_cp.routing.budget import BudgetTracker, AdaptiveThresholds
from chr_cp.routing.l1_coarse import (
    L1Router,
    L1Config,
    TaskCategory,
    AgentPoolConfig,
)
from chr_cp.routing.l2_step import L2Router, L2Config
from chr_cp.routing.l3_cache import L3CacheManager, L3Config, HandoffResult
from chr_cp.routing.orchestrator import CHRCPOrchestrator, OrchestratorConfig, CHRCPResult

__all__ = [
    # Decisions
    "RoutingAction",
    "RoutingDecision",
    "StepResult",
    # Budget
    "BudgetTracker",
    "AdaptiveThresholds",
    # L1
    "L1Router",
    "L1Config",
    "TaskCategory",
    "AgentPoolConfig",
    # L2
    "L2Router",
    "L2Config",
    # L3
    "L3CacheManager",
    "L3Config",
    "HandoffResult",
    # Orchestrator
    "CHRCPOrchestrator",
    "OrchestratorConfig",
    "CHRCPResult",
]