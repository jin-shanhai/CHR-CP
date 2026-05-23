"""Base API client with unified interface across providers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger


@dataclass
class CompletionResponse:
    """Unified completion response across providers.
    
    All API providers normalize their output to this structure so that
    upper layers (L1/L2/L3 routers) don't need to handle vendor differences.
    """
    
    content: str  # The actual text response
    tier_name: str  # Which tier produced this (T1/T2/T3/T4)
    model_id: str  # Underlying model identifier
    provider: str  # "deepseek" / "qwen" / etc.
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Cache info (None if provider doesn't expose)
    cache_hit_tokens: Optional[int] = None
    cache_miss_tokens: Optional[int] = None
    
    # Cost (computed by client based on pricing config)
    cost_usd: float = 0.0
    
    # Logprobs (None if provider doesn't expose or thinking mode)
    logprobs: Optional[Any] = None
    
    # Reasoning content (for thinking-mode models like DeepSeek-V4 reasoner)
    reasoning_content: Optional[str] = None
    
    # Latency
    latency_seconds: float = 0.0
    
    # Raw response (for debugging)
    raw_response: Optional[Any] = field(default=None, repr=False)
    
    def __repr__(self) -> str:
        return (
            f"CompletionResponse(tier={self.tier_name}, "
            f"tokens={self.prompt_tokens}+{self.completion_tokens}, "
            f"cache_hit={self.cache_hit_tokens}, "
            f"cost=${self.cost_usd:.6f}, "
            f"latency={self.latency_seconds:.2f}s)"
        )


class BaseClient(ABC):
    """Abstract base class for all tier-specific clients."""
    
    def __init__(
        self,
        tier_name: str,
        config: dict,
        api_key: str,
        base_url: str,
    ):
        self.tier_name = tier_name
        self.config = config
        self.model_id = config["model_id"]
        self.provider = config["provider"]
        self.mode = config.get("mode", "non-thinking")
        self.pricing = config["pricing"]
        self.supports_logprobs = config.get("supports_logprobs", False)
        
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300.0,
        )
        
        logger.info(f"Initialized {tier_name} client: {self.model_id} ({self.mode})")
    
    @abstractmethod
    def _build_request_params(
        self,
        messages: list[dict],
        **kwargs,
    ) -> dict:
        """Build provider-specific request parameters."""
        pass
    
    @abstractmethod
    def _parse_response(
        self,
        raw_response: Any,
        latency: float,
    ) -> CompletionResponse:
        """Parse provider-specific response into unified CompletionResponse."""
        pass
    
    def _compute_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit_tokens: Optional[int] = None,
    ) -> float:
        """Compute USD cost based on tier pricing config."""
        cache_hit = cache_hit_tokens or 0
        cache_miss = prompt_tokens - cache_hit
        
        cost = (
            cache_miss * self.pricing["input_per_M"] / 1_000_000
            + completion_tokens * self.pricing["output_per_M"] / 1_000_000
        )
        
        if cache_hit > 0 and self.pricing.get("cache_hit_per_M"):
            cost += cache_hit * self.pricing["cache_hit_per_M"] / 1_000_000
        
        return cost
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def chat_completion(
        self,
        messages: list[dict],
        **kwargs,
    ) -> CompletionResponse:
        """Send a chat completion request and return unified response.
        
        Args:
            messages: List of message dicts in OpenAI format
            **kwargs: Override generation params (temperature, max_tokens, etc.)
        
        Returns:
            Unified CompletionResponse
        """
        params = self._build_request_params(messages, **kwargs)
        
        t0 = time.time()
        try:
            raw = self._client.chat.completions.create(**params)
        except Exception as e:
            logger.error(f"{self.tier_name} API call failed: {e}")
            raise
        
        latency = time.time() - t0
        return self._parse_response(raw, latency)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tier={self.tier_name}, model={self.model_id})"