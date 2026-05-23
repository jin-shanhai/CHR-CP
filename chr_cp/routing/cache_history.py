"""Shared cache history for cross-task CA²R closed-loop feedback.

Lives at the Orchestrator level (not per-task) so that cache hit
patterns observed in earlier tasks influence routing decisions in later
tasks within the same experiment run.
"""

from __future__ import annotations
from collections import deque


class SharedCacheHistory:
    """Cross-task per-tier cache hit rate tracker.
    
    Designed to be created once at Orchestrator init and passed into
    each per-task BudgetTracker. All BudgetTrackers share the same
    underlying deques, so cache events from earlier tasks remain visible
    to later tasks.
    """
    
    def __init__(self, window_size: int = 50):
        """
        Args:
            window_size: Sliding window length per tier. 50 is reasonable
                         for benchmark runs of 100-1000 tasks.
        """
        self.window_size = window_size
        self._tier_history: dict[str, deque] = {}
    
    def record(
        self,
        tier: str,
        cache_hit_tokens: int,
        total_input_tokens: int,
    ) -> None:
        """Record a cache event."""
        if total_input_tokens <= 0:
            return
        if tier not in self._tier_history:
            self._tier_history[tier] = deque(maxlen=self.window_size)
        self._tier_history[tier].append(
            (max(0, cache_hit_tokens), total_input_tokens)
        )
    
    def get_hit_rate(self, tier: str) -> float:
        history = self._tier_history.get(tier)
        if not history:
            return 0.0
        total_hits = sum(h for h, _ in history)
        total_inputs = sum(t for _, t in history)
        return total_hits / total_inputs if total_inputs > 0 else 0.0
    
    def summary(self) -> dict:
        return {
            tier: round(self.get_hit_rate(tier), 4)
            for tier in self._tier_history
        }
    
    def total_observations(self, tier: str) -> int:
        return len(self._tier_history.get(tier, []))