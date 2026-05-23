"""Qwen API client (handles T3 tier via Alibaba Cloud DashScope)."""

from __future__ import annotations
from typing import Any
import os

from chr_cp.clients.base_client import BaseClient, CompletionResponse


class QwenClient(BaseClient):
    """Client for Qwen-Max via Alibaba Cloud DashScope.
    
    DashScope provides OpenAI-compatible interface but with quirks:
    - logprobs always returns null (regardless of request)
    - No cache_hit_tokens in usage response
    - Some response fields may differ slightly from OpenAI spec
    """
    
    def __init__(self, tier_name: str, config: dict):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set in environment")
        
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
        """Build DashScope-specific request parameters."""
        params = {
            "model": self.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False,
        }
        
        # Note: Qwen-Max doesn't support logprobs even if requested
        # (returns logprobs:null in response)
        
        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]
        
        return params
    
    def _parse_response(
        self,
        raw_response: Any,
        latency: float,
    ) -> CompletionResponse:
        """Parse DashScope response."""
        choice = raw_response.choices[0]
        usage = raw_response.usage
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        
        # DashScope doesn't expose cache hit; we estimate cost without it
        cost = self._compute_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=None,
        )
        
        return CompletionResponse(
            content=choice.message.content or "",
            tier_name=self.tier_name,
            model_id=self.model_id,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.total_tokens,
            cache_hit_tokens=None,  # Not exposed by DashScope
            cache_miss_tokens=None,
            cost_usd=cost,
            logprobs=None,  # Always null on Qwen-Max
            reasoning_content=None,  # qwen-max is non-thinking mode
            latency_seconds=latency,
            raw_response=raw_response,
        )