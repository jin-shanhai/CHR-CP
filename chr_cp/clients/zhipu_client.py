"""Zhipu AI client (T1 in new tier ladder).

Zhipu's BigModel platform exposes an OpenAI-compatible API endpoint at
https://open.bigmodel.cn/api/paas/v4. We use the standard openai SDK
(via base_url override) rather than the zhipuai SDK, for consistency
with the rest of the project.
"""

from __future__ import annotations
from typing import Any
import os

from chr_cp.clients.base_client import BaseClient, CompletionResponse


class ZhipuClient(BaseClient):
    """Client for Zhipu AI GLM-* models.
    
    Handles T1 in new tier ladder:
    - T1: glm-4.7-flash (free tier, no per-token charge)
    """
    
    def __init__(self, tier_name: str, config: dict):
        api_key = os.getenv("ZHIPU_API_KEY")
        base_url = os.getenv(
            "ZHIPU_BASE_URL",
            "https://open.bigmodel.cn/api/paas/v4",
        )
        if not api_key:
            raise ValueError("ZHIPU_API_KEY not set in environment")
        
        super().__init__(
            tier_name=tier_name,
            config=config,
            api_key=api_key,
            base_url=base_url,
        )
    
    def _build_request_params(
        self,
        messages: list[dict],
        **kwargs,
    ) -> dict:
        """Build Zhipu request params (OpenAI-compatible)."""
        params = {
            "model": self.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False,
        }
        
        # Zhipu does not support logprobs in their OpenAI-compatible endpoint
        # (logprobs requested via SDK is silently ignored)
        
        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]
        
        return params
    
    def _parse_response(
        self,
        raw_response: Any,
        latency: float,
    ) -> CompletionResponse:
        """Parse Zhipu response (OpenAI-compatible format)."""
        choice = raw_response.choices[0]
        usage = raw_response.usage
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        # Zhipu does not currently expose prompt cache hit info in their
        # OpenAI-compatible endpoint. Set to None.
        cache_hit = None
        cache_miss = None
        
        cost = self._compute_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=cache_hit,
        )
        
        return CompletionResponse(
            content=choice.message.content or "",
            tier_name=self.tier_name,
            model_id=self.model_id,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_hit_tokens=cache_hit,
            cache_miss_tokens=cache_miss,
            cost_usd=cost,
            logprobs=None,
            reasoning_content=None,    # GLM-4.7-Flash is non-thinking
            latency_seconds=latency,
            raw_response=raw_response,
        )