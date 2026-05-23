"""Cost and cache hit tracking across the experiment lifetime.

Used to produce the cost-accuracy Pareto figures and cache hit rate
breakdown analysis in the paper.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Optional
from pathlib import Path
import json
import time

from chr_cp.clients.base_client import CompletionResponse


@dataclass
class CallRecord:
    """Single API call record."""
    timestamp: float
    tier: str
    provider: str
    model_id: str
    
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    cache_hit_tokens: Optional[int]
    cache_miss_tokens: Optional[int]
    
    cost_usd: float
    latency_seconds: float
    
    # Optional context
    task_id: Optional[str] = None
    benchmark: Optional[str] = None
    step_id: Optional[str] = None
    routing_action: Optional[str] = None  # "STAY" / "BRANCH" / "ESCALATE"


@dataclass
class TierStats:
    """Aggregated stats for a single tier."""
    tier: str
    call_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cache_hit_tokens: int = 0
    total_cache_miss_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_seconds: float = 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        total_input = self.total_cache_hit_tokens + self.total_cache_miss_tokens
        if total_input == 0:
            return 0.0
        return self.total_cache_hit_tokens / total_input
    
    @property
    def avg_latency(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency_seconds / self.call_count


class CostTracker:
    """Tracks all API calls during an experiment, computes summary stats.
    
    Usage:
        tracker = CostTracker()
        # ... in experiment loop ...
        response = pool.invoke(tier, messages)
        tracker.record(response, task_id="gsm8k_42", benchmark="gsm8k")
        # ... after experiment ...
        tracker.save("results/main/gsm8k_chrcp.json")
        print(tracker.summary())
    """
    
    def __init__(self):
        self.records: list[CallRecord] = []
        self._tier_stats: dict[str, TierStats] = defaultdict(
            lambda: TierStats(tier="unknown")
        )
        self._start_time = time.time()
    
    def record(
        self,
        response: CompletionResponse,
        task_id: Optional[str] = None,
        benchmark: Optional[str] = None,
        step_id: Optional[str] = None,
        routing_action: Optional[str] = None,
    ) -> None:
        """Record a single API response."""
        record = CallRecord(
            timestamp=time.time() - self._start_time,
            tier=response.tier_name,
            provider=response.provider,
            model_id=response.model_id,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            cache_hit_tokens=response.cache_hit_tokens,
            cache_miss_tokens=response.cache_miss_tokens,
            cost_usd=response.cost_usd,
            latency_seconds=response.latency_seconds,
            task_id=task_id,
            benchmark=benchmark,
            step_id=step_id,
            routing_action=routing_action,
        )
        self.records.append(record)
        
        # Update tier stats
        stats = self._tier_stats[response.tier_name]
        stats.tier = response.tier_name
        stats.call_count += 1
        stats.total_prompt_tokens += response.prompt_tokens
        stats.total_completion_tokens += response.completion_tokens
        if response.cache_hit_tokens is not None:
            stats.total_cache_hit_tokens += response.cache_hit_tokens
        if response.cache_miss_tokens is not None:
            stats.total_cache_miss_tokens += response.cache_miss_tokens
        else:
            # No cache info → treat all input as cache miss for cost accuracy
            stats.total_cache_miss_tokens += response.prompt_tokens
        stats.total_cost_usd += response.cost_usd
        stats.total_latency_seconds += response.latency_seconds
    
    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)
    
    @property
    def total_calls(self) -> int:
        return len(self.records)
    
    def get_tier_stats(self, tier: str) -> TierStats:
        return self._tier_stats[tier]
    
    def summary(self) -> dict:
        """Produce a summary dict for console display or JSON export."""
        return {
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost, 6),
            "total_runtime_seconds": time.time() - self._start_time,
            "by_tier": {
                tier: {
                    "calls": stats.call_count,
                    "prompt_tokens": stats.total_prompt_tokens,
                    "completion_tokens": stats.total_completion_tokens,
                    "cache_hit_tokens": stats.total_cache_hit_tokens,
                    "cache_hit_rate": round(stats.cache_hit_rate, 4),
                    "cost_usd": round(stats.total_cost_usd, 6),
                    "avg_latency": round(stats.avg_latency, 3),
                }
                for tier, stats in self._tier_stats.items()
            },
        }
    
    def save(self, path: str | Path) -> None:
        """Save full call history + summary to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "summary": self.summary(),
            "records": [asdict(r) for r in self.records],
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def reset(self) -> None:
        """Reset all tracking (useful between experiment phases)."""
        self.records.clear()
        self._tier_stats.clear()
        self._start_time = time.time()