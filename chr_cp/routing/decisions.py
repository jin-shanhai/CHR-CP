"""Routing decision data structures.

Three actions are defined (a key novelty over prior 2-action routers):
- STAY: keep current tier, accept the response
- BRANCH: same-tier multi-sampling with voting (cheap)
- ESCALATE: upgrade to a stronger tier (expensive)

The 3-action design is what enables CHR-CP to outperform pure 2-action
baselines on cost-accuracy Pareto.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any

from chr_cp.confidence.vc2 import UncertaintySignal
from chr_cp.clients.base_client import CompletionResponse


class RoutingAction(str, Enum):
    """The three L2 routing actions."""
    STAY = "STAY"            # Accept current response, no extra cost
    BRANCH = "BRANCH"        # Same-tier resampling with voting
    ESCALATE = "ESCALATE"    # Upgrade to higher tier


@dataclass
class RoutingDecision:
    """One L2 routing decision with full trace.
    
    This is recorded for every step to produce the per-step routing log
    used in the paper's case study (Section 4.6) and ablation analysis.
    """
    
    action: RoutingAction
    current_tier: str
    target_tier: Optional[str] = None  # only meaningful for ESCALATE
    
    # The signal that drove the decision
    uncertainty: float = 0.0
    threshold_low: float = 0.0
    threshold_high: float = 0.0
    
    # Sub-signals for analysis
    u_verbalized: float = 0.0
    u_consistency: float = 0.0
    
    # Why this decision was made (human-readable)
    reason: str = ""
    
    # Whether VC² used full or lite mode
    vc_mode: str = "full"
    
    # Whether verbalized parse succeeded (affects fusion)
    verbalized_parsed: bool = True
    
    # Budget state at decision time
    budget_remaining_ratio: float = 1.0
    
    # === NEW: CA²R diagnostic fields ===
    ca2r_budget_factor: float = 1.0       # (2 − r_budget)
    ca2r_cache_factor: float = 1.0        # (1 − β · h_target)
    ca2r_h_target: float = 0.0            # observed cache hit rate of target tier
    ca2r_target_tier: Optional[str] = None  # which tier was used as target_tier in adjustment
    
    ca2r_cache_factor_low: float = 1.0   # NEW
    ca2r_cache_factor_high: float = 1.0  # NEW
    ca2r_h_current: float = 0.0          # NEW
    ca2r_current_tier: Optional[str] = None  # NEW
    def __repr__(self) -> str:
        return (
            f"RoutingDecision({self.action.value}, "
            f"tier={self.current_tier}"
            f"{' → ' + self.target_tier if self.target_tier else ''}, "
            f"U={self.uncertainty:.3f}, "
            f"reason={self.reason!r})"
        )

@dataclass
class StepResult:
    """Result of a single L2 step (after all routing actions resolved).
    
    Contains the FINAL accepted response for this step, plus full trace
    of decisions and any branch/escalate sub-calls.
    """
    
    # Final accepted response
    final_response: CompletionResponse
    final_tier: str
    
    # Decision history for this step (1 STAY = 1 entry, BRANCH = 2 entries, etc.)
    decisions: list[RoutingDecision] = field(default_factory=list)
    
    # All API calls made for this step (for cost accounting)
    all_responses: list[CompletionResponse] = field(default_factory=list)
    
    # Aggregated metrics
    total_cost_usd: float = 0.0
    total_latency_seconds: float = 0.0
    
    # Final uncertainty signal (after all actions)
    final_uncertainty: Optional[UncertaintySignal] = None
    
    # Number of actions taken
    @property
    def num_branches(self) -> int:
        return sum(1 for d in self.decisions if d.action == RoutingAction.BRANCH)
    
    @property
    def num_escalates(self) -> int:
        return sum(1 for d in self.decisions if d.action == RoutingAction.ESCALATE)
    
    def summary(self) -> str:
        """One-line summary for logging."""
        path = " → ".join(d.action.value for d in self.decisions)
        return (
            f"[{self.final_tier}] {path} | "
            f"${self.total_cost_usd:.6f} | "
            f"{self.total_latency_seconds:.2f}s"
        )