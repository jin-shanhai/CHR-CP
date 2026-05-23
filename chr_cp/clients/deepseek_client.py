"""DeepSeek API client (handles T1/T2/T4 tiers)."""

from __future__ import annotations
from typing import Any
import os

from chr_cp.clients.base_client import BaseClient, CompletionResponse


class DeepSeekClient(BaseClient):
    """Client for DeepSeek API.
    
    Handles three tiers:
    - T4: deepseek-v4-pro (thinking mode)
    - T2: deepseek-v4-flash (thinking mode, alias: deepseek-reasoner)
    - T1: deepseek-v4-flash (non-thinking mode, alias: deepseek-chat)
    
    Note: As of 2026-04-24, DeepSeek introduced v4-pro/v4-flash. The legacy
    aliases (deepseek-chat, deepseek-reasoner) still work and route to v4-flash
    until 2026-07-24 deprecation. After deprecation, update model_id in
    configs/models.yaml to "deepseek-v4-pro" or "deepseek-v4-flash".
    """
    
    def __init__(self, tier_name: str, config: dict):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set in environment")
        
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
        """Build DeepSeek-specific request parameters.
        
        For V4 series, thinking mode is enabled via two fields:
        - reasoning_effort: "high" or "max" (OpenAI-compatible top-level param)
        - extra_body={"thinking": {"type": "enabled"}} (DeepSeek-specific, must
        be passed via extra_body when using OpenAI SDK)
        
        Reference: https://api-docs.deepseek.com/guides/thinking_mode
        """
        default_max = 16384 if self.mode == "thinking" else 4096
        params = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", default_max),
            "stream": False,
        }
        
        # DeepSeek V4 thinking mode: pass via extra_body (OpenAI SDK convention)
        # Note: thinking mode does NOT support temperature/top_p/penalties
        if self.mode == "thinking":
            params["reasoning_effort"] = kwargs.get("reasoning_effort", "high")
            params["extra_body"] = {"thinking": {"type": "enabled"}}
            # Don't set temperature/top_p in thinking mode
        else:
            # Non-thinking mode: standard sampling params
            params["temperature"] = kwargs.get("temperature", 0.3)
            # top_p omitted — not all DeepSeek model aliases support it
            # Explicitly disable thinking
            params["extra_body"] = {"thinking": {"type": "disabled"}}
        
        # logprobs only supported in non-thinking mode (T1 historically; T2/T3 V4
        # do not currently expose logprobs)
        if self.supports_logprobs and self.mode == "non-thinking":
            if kwargs.get("logprobs", False):
                params["logprobs"] = True
                params["top_logprobs"] = kwargs.get("top_logprobs", 5)
        
        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]
        
        return params
    
    def _parse_response(
        self,
        raw_response: Any,  
        latency: float,
    ) -> CompletionResponse:
        """Parse DeepSeek response (compatible with OpenAI ChatCompletion format)."""
        choice = raw_response.choices[0]
        usage = raw_response.usage
        
        # DeepSeek-specific: cache hit tokens
        cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
        cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
        
        # Reasoning content (only thinking mode)
        reasoning = None
        if self.mode == "thinking":
            reasoning = getattr(choice.message, "reasoning_content", None)
        
        # Logprobs (only T1 non-thinking)
        logprobs = None
        if self.supports_logprobs and choice.logprobs:
            logprobs = choice.logprobs
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        
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
            total_tokens=usage.total_tokens,
            cache_hit_tokens=cache_hit,
            cache_miss_tokens=cache_miss,
            cost_usd=cost,
            logprobs=logprobs,
            reasoning_content=reasoning,
            latency_seconds=latency,
            raw_response=raw_response,
        )