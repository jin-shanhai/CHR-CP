"""Budget + Cache State tracking with CA²R closed-loop adaptive thresholds.

CA²R (Cache-Aware Adaptive Routing) formula:
    τ_adj = τ_base · (2 − r_budget) · (1 − β · h_target)

where:
- r_budget    = remaining_budget / total_budget       (forward signal)
- h_target    = target_tier rolling cache hit rate    (reverse signal)
- β           = cache_sensitivity ∈ [0, 1]            (default 0.3)

This implements the closed-loop coupling between L2 routing decisions
and L3 cache state — the core novelty claim of CHR-CP.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


@dataclass
class AdaptiveThresholds:
    """Container for the (cache- and budget-adjusted) τ_low and τ_high.
    
    Includes diagnostic factors so that experiments can attribute the
    adjustment to budget vs. cache contributions separately.
    """
    tau_low: float
    tau_high: float
    
    # Decomposed factors (for paper figures and ablation analysis)
    budget_factor: float = 1.0       # (2 − r_budget)
    cache_factor: float = 1.0        # (1 − β · h_target)
    target_tier_used: Optional[str] = None
    h_target_observed: float = 0.0
    
      # NEW: bidirectional cache factors
    cache_factor_low: float = 1.0
    cache_factor_high: float = 1.0
    h_current_observed: float = 0.0
    current_tier_used: Optional[str] = None
    def __post_init__(self):
        if not 0.0 <= self.tau_low <= self.tau_high <= 1.0:
            raise ValueError(
                f"Invalid thresholds: τ_low={self.tau_low} must be in "
                f"[0, τ_high={self.tau_high}] ⊆ [0, 1]"
            )


class BudgetTracker:
    """Closed-loop budget + cache-state tracker (CA²R).
    
    Maintains:
    - Cumulative spend vs. total budget (forward feedback to L2)
    - Per-tier sliding window of cache hit rates (reverse feedback to L2)
    
    Usage:
        tracker = BudgetTracker(total_budget_usd=0.50, cache_sensitivity=0.3)
        
        # In routing loop:
        thresholds = tracker.adjust_thresholds(
            tau_low=0.3, tau_high=0.65,
            target_tier="T3",   # NEW: tier we'd ESCALATE to if needed
        )
        
        # After each API call:
        tracker.add_cost(response.cost_usd)
        tracker.record_cache_event(
            tier=response.tier_name,
            cache_hit_tokens=response.cache_hit_tokens or 0,
            total_input_tokens=response.prompt_tokens,
        )
    """
    
    def __init__(
        self,
        total_budget_usd: float,
        warmup_floor: float = 0.05,
        cache_sensitivity: float = 0.3,    # β in CA²R
        cache_window_size: int = 10,        # W: rolling window length
        cache_floor: float = 0.5,           # min for (1 − β·h_target)
        shared_history: Optional["SharedCacheHistory"] = None,  # NEW
    ):
        """
        Args:
            total_budget_usd: Total budget for this task (or task batch).
            warmup_floor: Minimum budget_ratio used in adjustment (prevents
                          τ * 2.0 = 2.0 → all decisions become STAY).
            cache_sensitivity: β; how much the cache hit rate affects τ.
                               0 → no cache feedback (CA²R disabled).
                               1 → full cache discount (aggressive).
            cache_window_size: How many recent calls to consider for h_target.
            cache_floor: Lower bound on cache_factor; prevents τ → 0 even if
                         cache is fully hot.
        """
        if total_budget_usd <= 0:
            raise ValueError("total_budget_usd must be positive")
        if not 0.0 <= cache_sensitivity <= 1.0:
            raise ValueError(f"cache_sensitivity must be in [0, 1], got {cache_sensitivity}")
        if cache_window_size <= 0:
            raise ValueError("cache_window_size must be positive")
        
        self.total_budget = total_budget_usd
        self.spent = 0.0
        self.warmup_floor = max(0.0, min(1.0, warmup_floor))
        self.cache_sensitivity = cache_sensitivity
        self.cache_window_size = cache_window_size
        self.cache_floor = max(0.0, min(1.0, cache_floor))
        self._shared_history = shared_history
        # If no shared history, fall back to per-tracker storage
        self._tier_cache_history: dict[str, deque] = {} if shared_history is None else None
        # Per-tier sliding window of (cache_hit_tokens, total_input_tokens)
        self._tier_cache_history: dict[str, deque] = {}
    
    # -------- Cost tracking --------
    
    def add_cost(self, cost_usd: float) -> None:
        """Record cost incurred."""
        self.spent += max(0.0, cost_usd)
    
    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self.spent)
    
    @property
    def remaining_ratio(self) -> float:
        if self.total_budget <= 0:
            return 0.0
        return max(0.0, min(1.0, self.remaining / self.total_budget))
    
    # -------- Cache state tracking (NEW for CA²R) --------
    
    def record_cache_event(self, tier, cache_hit_tokens, total_input_tokens):
        if total_input_tokens <= 0:
            return
        if self._shared_history is not None:
            self._shared_history.record(tier, cache_hit_tokens, total_input_tokens)
            return
        # Fallback: per-tracker storage
        if tier not in self._tier_cache_history:
            self._tier_cache_history[tier] = deque(maxlen=self.cache_window_size)
        self._tier_cache_history[tier].append(
            (max(0, cache_hit_tokens), total_input_tokens)
        )
        
    def get_cache_hit_rate(self, tier):
        if self._shared_history is not None:
            return self._shared_history.get_hit_rate(tier)
        history = self._tier_cache_history.get(tier)
        if not history:
            return 0.0
        total_hits = sum(h for h, _ in history)
        total_inputs = sum(t for _, t in history)
        return total_hits / total_inputs if total_inputs > 0 else 0.0
    
    # -------- CA²R threshold adjustment --------
    
    def adjust_thresholds(
        self,
        tau_low: float,
        tau_high: float,
        current_tier: Optional[str] = None,    # NEW
        target_tier: Optional[str] = None,
    ) -> AdaptiveThresholds:
        """CA²R: bidirectional cache feedback.
        
        τ_low_adj  = τ_low  · (2 − r_budget) · (1 − β · h_current)
                        ↑ STAY likelihood ↑ when current tier's cache is hot
        
        τ_high_adj = τ_high · (2 − r_budget) · (1 − β · h_target)
                        ↑ ESCALATE likelihood ↑ when target tier's cache is hot
        """
        # Budget factor
        ratio = max(self.warmup_floor, self.remaining_ratio)
        budget_factor = 2.0 - ratio
        
        # Current-tier cache factor (affects τ_low → STAY direction)
        if current_tier is not None:
            h_current = self.get_cache_hit_rate(current_tier)
            cache_factor_low = 1.0 - self.cache_sensitivity * h_current
            cache_factor_low = max(self.cache_floor, cache_factor_low)
        else:
            h_current = 0.0
            cache_factor_low = 1.0
        
        # Target-tier cache factor (affects τ_high → ESCALATE direction)
        if target_tier is not None:
            h_target = self.get_cache_hit_rate(target_tier)
            cache_factor_high = 1.0 - self.cache_sensitivity * h_target
            cache_factor_high = max(self.cache_floor, cache_factor_high)
        else:
            h_target = 0.0
            cache_factor_high = 1.0
        
        new_low = min(1.0, tau_low * budget_factor * cache_factor_low)
        new_high = min(1.0, tau_high * budget_factor * cache_factor_high)
        
        if new_low > new_high:
            new_low = new_high
        
        return AdaptiveThresholds(
            tau_low=new_low,
            tau_high=new_high,
            budget_factor=budget_factor,
            cache_factor=cache_factor_high,    # report ESCALATE-side as primary
            target_tier_used=target_tier,
            h_target_observed=h_target,
            # NEW diagnostic fields
            cache_factor_low=cache_factor_low,
            cache_factor_high=cache_factor_high,
            h_current_observed=h_current,
            current_tier_used=current_tier,
        )
    # -------- Diagnostics --------
    
    def summary(self) -> dict:
        base = {
            "total_budget_usd": self.total_budget,
            "spent_usd": round(self.spent, 6),
            "remaining_usd": round(self.remaining, 6),
            "remaining_ratio": round(self.remaining_ratio, 4),
            "cache_sensitivity_beta": self.cache_sensitivity,
        }
        if self._shared_history is not None:
            base["tier_cache_hit_rates"] = self._shared_history.summary()
            base["history_mode"] = "shared"
        else:
            base["tier_cache_hit_rates"] = {
                tier: round(self.get_cache_hit_rate(tier), 4)
                for tier in self._tier_cache_history
            }
            base["history_mode"] = "per_task"
        return base