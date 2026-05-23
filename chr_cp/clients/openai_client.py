"""OpenAI client (T4 in new tier ladder).

Uses the standard openai SDK against either the official endpoint
(api.openai.com) or a third-party proxy (e.g., shiyunapi.com) by setting
the OPENAI_BASE_URL environment variable. Both are OpenAI-compatible.
"""

from __future__ import annotations
from typing import Any
import os

from chr_cp.clients.base_client import BaseClient, CompletionResponse


class OpenAIClient(BaseClient):
    """Client for OpenAI / OpenAI-compatible proxy endpoints.
    
    Handles T4 in new tier ladder:
    - T4: gpt-5.5 (top tier, accessed via proxy or official endpoint)
    """
    
    def __init__(self, tier_name: str, config: dict):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
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
        """Build OpenAI request parameters.
        
        GPT-5.x reasoning models consume tokens for internal reasoning even
        in chat-completions mode. We default max_tokens to a higher value to
        avoid premature truncation. Callers can still override via kwargs.
        """
        # GPT-5.x reasoning models need more headroom for internal reasoning.
        # Default to 8192 unless explicitly overridden.
        default_max_tokens = 16384 if "gpt-5" in self.model_id.lower() else 4096
        
        params = {
            "model": self.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", default_max_tokens),
            "stream": False,
        }

        # top_p is not supported by some models (e.g. gpt-5.5);
        # only include it when explicitly requested by the caller.
        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]

        # GPT models support logprobs in non-reasoning mode
        if self.supports_logprobs and self.mode == "non-thinking":
            if kwargs.get("logprobs", False):
                params["logprobs"] = True
                params["top_logprobs"] = kwargs.get("top_logprobs", 5)

        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]

        # GPT-5.x models support reasoning_effort
        if "gpt-5" in self.model_id.lower() and "reasoning_effort" in kwargs:
            params["reasoning_effort"] = kwargs["reasoning_effort"]

        return params
    
    def _parse_response(
        self,
        raw_response: Any,
        latency: float,
    ) -> CompletionResponse:
        """Parse OpenAI response."""
        choice = raw_response.choices[0]
        usage = raw_response.usage
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        # OpenAI prompt cache: usage.prompt_tokens_details.cached_tokens
        cache_hit = None
        cache_miss = None
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", None)
            if cached is not None:
                cache_hit = cached
                cache_miss = max(0, prompt_tokens - cached)
        
        # Logprobs (only when explicitly requested)
        logprobs = None
        if self.supports_logprobs and getattr(choice, "logprobs", None):
            logprobs = choice.logprobs
        
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
            logprobs=logprobs,
            reasoning_content=None,
            latency_seconds=latency,
            raw_response=raw_response,
        )