"""Backward-compatible re-exports from providers/_anthropic_reasoning.py.

All logic has moved to ``providers._anthropic_reasoning``.  This shim
keeps existing ``from jyagent.llm.reasoning import …`` working.
"""

from __future__ import annotations

from .providers._anthropic_reasoning import (  # noqa: F401
    AnthropicReasoningConfig,
    AnthropicThinkingAdaptiveConfig,
    AnthropicThinkingDisabledConfig,
    build_anthropic_request_reasoning,
    validate_anthropic_reasoning,
)

__all__ = [
    "AnthropicReasoningConfig",
    "AnthropicThinkingAdaptiveConfig",
    "AnthropicThinkingDisabledConfig",
    "build_anthropic_request_reasoning",
    "validate_anthropic_reasoning",
]
