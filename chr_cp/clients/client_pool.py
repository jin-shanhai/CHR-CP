"""Unified 4-tier client pool: dispatch requests to the right tier."""

from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional
import yaml

from loguru import logger

from chr_cp.clients.base_client import BaseClient, CompletionResponse
from chr_cp.clients.deepseek_client import DeepSeekClient
from chr_cp.clients.qwen_client import QwenClient
from chr_cp.clients.zhipu_client import ZhipuClient
from chr_cp.clients.openai_client import OpenAIClient

class Tier(str, Enum):
    """Tier identifiers (v2 ladder)."""
    T1 = "T1"  # weak: GLM-4.7-Flash (free, zhipu)
    T2 = "T2"  # mid: DeepSeek-V4-Flash (deepseek)
    T3 = "T3"  # strong: DeepSeek-V4-Pro (deepseek)
    T4 = "T4"  # top: GPT-5.5 (openai)


class ClientPool:
    """Unified 4-tier client pool.
    
    Provides a single interface to invoke any of the 4 tiers, with shared
    response format. Used as the foundation by L2/L3 routers.
    
    Example:
        pool = ClientPool.from_config("configs/models.yaml")
        response = pool.invoke(Tier.T2, messages=[...])
    """
    
    PROVIDER_TO_CLIENT = {
        "deepseek": DeepSeekClient,
        "qwen": QwenClient,
        "zhipu": ZhipuClient,
        "openai": OpenAIClient,
    }
    
    def __init__(self, configs: dict[str, dict]):
        """Initialize pool with tier configurations.
        
        Args:
            configs: Dict mapping tier names ("T1", "T2", ...) to config dicts.
        """
        self.configs = configs
        self._clients: dict[str, BaseClient] = {}
        
        for tier_name, tier_config in configs.items():
            provider = tier_config["provider"]
            client_cls = self.PROVIDER_TO_CLIENT.get(provider)
            if not client_cls:
                raise ValueError(f"Unknown provider: {provider} for tier {tier_name}")
            
            self._clients[tier_name] = client_cls(
                tier_name=tier_name,
                config=tier_config,
            )
        
        logger.info(f"ClientPool initialized with tiers: {list(self._clients.keys())}")
    
    @classmethod
    def from_config(cls, config_path: str | Path) -> "ClientPool":
        """Load pool from YAML config file."""
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
        
        return cls(configs=full_config["tiers"])
    
    def invoke(
        self,
        tier: str | Tier,
        messages: list[dict],
        **kwargs,
    ) -> CompletionResponse:
        """Invoke a specific tier with messages.

        Args:
            tier: Tier identifier ("T1"/"T2"/"T3"/"T4" or Tier enum)
            messages: OpenAI-format message list
            **kwargs: Generation params (temperature, max_tokens, logprobs, etc.)

        Returns:
            Unified CompletionResponse
        """
        tier_str = tier.value if isinstance(tier, Tier) else tier
        if tier_str not in self._clients:
            raise ValueError(
                f"Tier {tier_str} not configured. Available: {list(self._clients.keys())}"
            )

        logger.debug(
            f"[{tier_str}] REQUEST: max_tokens={kwargs.get('max_tokens')}, "
            f"temperature={kwargs.get('temperature')}, "
            f"n_messages={len(messages)}, "
            f"system_len={len(messages[0]['content']) if messages else 0}"
        )

        client = self._clients[tier_str]
        response = client.chat_completion(messages=messages, **kwargs)

        logger.debug(
            f"[{tier_str}] RESPONSE: {response.prompt_tokens}+{response.completion_tokens} tokens, "
            f"${response.cost_usd:.6f}, {response.latency_seconds:.2f}s, "
            f"content={response.content[:200]!r}"
        )
        
        return response
    
    def get_client(self, tier: str | Tier) -> BaseClient:
        """Get the underlying client for a tier (for advanced use)."""
        tier_str = tier.value if isinstance(tier, Tier) else tier
        return self._clients[tier_str]
    
    def list_tiers(self) -> list[str]:
        """List all configured tier names."""
        return list(self._clients.keys())
    
    def supports_logprobs(self, tier: str | Tier) -> bool:
        """Check whether a tier supports logprobs."""
        tier_str = tier.value if isinstance(tier, Tier) else tier
        return self._clients[tier_str].supports_logprobs