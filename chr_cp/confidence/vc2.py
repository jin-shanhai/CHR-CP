"""VC²: Verbalized + Consistency Confidence — the core uncertainty signal of CHR-CP.

This is the central novelty of the paper:
- Most prior work assumes logprobs are available (white-box setting)
- In API setting, thinking-mode models disable logprobs
- VC² fuses two API-accessible signals to fill this gap

Signal A: Verbalized Confidence (model self-reports its certainty)
Signal B: Multi-sample Consistency (sampling-based uncertainty)

Fusion: U = α * U_consistency + (1 - α) * U_verbalized,  α ∈ [0, 1]
Default α = 0.6 (consistency weighted higher; verifiable in ablation)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from chr_cp.clients import ClientPool, Tier
from chr_cp.clients.base_client import CompletionResponse
from chr_cp.confidence.verbalized import (
    VerbalizedConfidenceParser,
    VerbalizedConfidence,
)
from chr_cp.confidence.consistency import (
    ConsistencyEstimator,
    ConsistencyResult,
    TaskType,
)


@dataclass
class UncertaintySignal:
    """Full output of VC² estimation, includes all sub-signals for analysis."""
    
    # Combined uncertainty in [0, 1] (higher = more uncertain)
    U: float
    
    # Sub-signals
    U_verbalized: float
    U_consistency: float
    U_logprob: Optional[float] = None  # only populated if T1 + thinking off
    
    # Raw verbalized confidence (for paper's per-sample analysis)
    verbalized: Optional[VerbalizedConfidence] = None
    
    # Raw consistency result
    consistency: Optional[ConsistencyResult] = None
    
    # Fusion weights actually used
    alpha: float = 0.6  # weight for consistency
    beta: float = 0.0   # weight for logprob (0 unless used)
    
    # Diagnostic
    fusion_method: str = "vc2"
    
    def __repr__(self) -> str:
        return (
            f"UncertaintySignal(U={self.U:.3f}, "
            f"verbalized={self.U_verbalized:.3f}, "
            f"consistency={self.U_consistency:.3f})"
        )


class VC2Estimator:
    """Combined Verbalized + Consistency estimator.
    
    The estimator can operate in two modes:
    
    1. **Lite mode** (cheap, fast):
       - Use verbalized confidence ONLY from the primary response
       - Skip consistency sampling
       - Used when L2 already has high-confidence STAY signal
    
    2. **Full mode** (expensive, accurate):
       - Verbalized confidence from primary
       - Plus K-sample consistency
       - Used when L2 is at the BRANCH/ESCALATE boundary
    
    The two-mode design lets L2 escalate uncertainty estimation cost only
    when needed (a meta-routing of the routing signal).
    
    Usage:
        estimator = VC2Estimator(pool=pool)
        signal = estimator.estimate(
            primary_response=response,
            messages=messages_for_consistency,
            tier=Tier.T2,
            task_type=TaskType.NUMERIC,
            mode="full",
        )
        if signal.U > 0.65:
            # ESCALATE
            ...
    """
    
    def __init__(
        self,
        pool: ClientPool,
        alpha: float = 0.6,
        k_samples: int = 3,                       # NEW
        consistency_estimator: Optional[ConsistencyEstimator] = None,
        vc_parser: Optional[VerbalizedConfidenceParser] = None,
        cost_tracker=None,
    ):
        """
        Args:
            pool: Client pool
            alpha: Weight for U_consistency in U = α*U_cons + (1-α)*U_verb
            k_samples: Number of consistency samples (3, 5, 7). Used when
                    consistency_estimator is not explicitly provided.
            consistency_estimator: Optional pre-built estimator (overrides k_samples)
            vc_parser: Optional confidence tag parser
            cost_tracker: Optional CostTracker
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        
        self.pool = pool
        self.alpha = alpha
        self.k_samples = k_samples
        self.vc_parser = vc_parser or VerbalizedConfidenceParser()
        
        # Build a fresh ConsistencyEstimator with the desired K
        # (unless user provided one explicitly)
        self.consistency_estimator = consistency_estimator or ConsistencyEstimator(
            pool=pool,
            k_samples=k_samples,                  # NEW: pass K down
            vc_parser=self.vc_parser,
            cost_tracker=cost_tracker,
        )
        self.cost_tracker = cost_tracker
    
    def estimate(
        self,
        primary_response: CompletionResponse,
        messages: list[dict],
        tier: str | Tier,
        task_type: TaskType,
        mode: str = "full",
        max_tokens: int = 2048,
        task_id: Optional[str] = None,
    ) -> UncertaintySignal:
        """Compute VC² uncertainty signal.

        Args:
            primary_response: The main agent's output (we extract verbalized from this)
            messages: Original prompt for consistency sampling
            tier: Tier to use for consistency samples
            task_type: Task type for similarity metric
            mode: "lite" (verbalized only) or "full" (verbalized + consistency)
            max_tokens: Per-sample completion limit for consistency samples
            task_id: For cost tracking

        Returns:
            UncertaintySignal
        """
        # === Signal A: Verbalized confidence ===
        verbalized = self.vc_parser.parse(primary_response.content)
        u_verbalized = verbalized.uncertainty

        if mode == "lite":
            return UncertaintySignal(
                U=u_verbalized,
                U_verbalized=u_verbalized,
                U_consistency=0.5,
                verbalized=verbalized,
                consistency=None,
                alpha=0.0,
                fusion_method="vc2_lite",
            )

        # Plan A: Skip consistency on T4 when model is very confident.
        # T3 removed from Plan A: T3 is always reached via escalation (the problem
        # is already proven "hard"), so T3's Uv=0 cannot be fully trusted.
        # T4 is extremely rare/expensive; still skip when confident.
        tier_str = tier.value if isinstance(tier, Tier) else tier
        if tier_str == "T4" and u_verbalized < 0.1:
            return UncertaintySignal(
                U=u_verbalized,
                U_verbalized=u_verbalized,
                U_consistency=0.5,
                verbalized=verbalized,
                consistency=None,
                alpha=0.0,
                fusion_method="vc2_lite_t3_skip",
            )

        # === Signal B: Consistency (full mode only) ===
        # Extract primary answer as anchor for verify-based consistency
        import re
        anchor = None
        m = re.search(r'\\boxed\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', primary_response.content)
        if m:
            anchor = m.group(1).strip()

        consistency_result = self.consistency_estimator.estimate(
            messages=messages,
            tier=tier,
            task_type=task_type,
            max_tokens=max_tokens,
            task_id=task_id,
            anchor_answer=anchor,
        )
        u_consistency = consistency_result.uncertainty
        
        # === Fusion ===
        u_combined = self.alpha * u_consistency + (1 - self.alpha) * u_verbalized
        u_combined = max(0.0, min(1.0, u_combined))  # clip to [0, 1]
        
        return UncertaintySignal(
            U=u_combined,
            U_verbalized=u_verbalized,
            U_consistency=u_consistency,
            verbalized=verbalized,
            consistency=consistency_result,
            alpha=self.alpha,
            fusion_method="vc2_full",
        )
    
    def estimate_lite(
        self,
        primary_response: CompletionResponse,
    ) -> UncertaintySignal:
        """Shortcut for lite-mode estimation (verbalized only)."""
        verbalized = self.vc_parser.parse(primary_response.content)
        return UncertaintySignal(
            U=verbalized.uncertainty,
            U_verbalized=verbalized.uncertainty,
            U_consistency=0.5,
            verbalized=verbalized,
            consistency=None,
            alpha=0.0,
            fusion_method="vc2_lite",
        )