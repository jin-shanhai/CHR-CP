"""Prompt engineering utilities."""

from chr_cp.prompts.stable_prefix import StablePrefixBuilder, StablePrefix
from chr_cp.prompts.role_templates import (
    RoleTemplate,
    SOLVER,
    VERIFIER,
    AGGREGATOR,
    COMPRESSOR,
    ROLE_REGISTRY,
    get_role,
    CONFIDENCE_FOOTER,
)
from chr_cp.prompts.distillation import (
    DistilledContext,
    DISTILLATION_INSTRUCTION,
    build_distillation_messages,
    parse_distilled,
    estimate_tokens,
)

__all__ = [
    "StablePrefixBuilder",
    "StablePrefix",
    "RoleTemplate",
    "SOLVER",
    "VERIFIER",
    "AGGREGATOR",
    "COMPRESSOR",
    "ROLE_REGISTRY",
    "get_role",
    "CONFIDENCE_FOOTER",
    "DistilledContext",
    "DISTILLATION_INSTRUCTION",
    "build_distillation_messages",
    "parse_distilled",
    "estimate_tokens",
]