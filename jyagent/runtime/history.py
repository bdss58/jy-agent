"""Message-history helpers — re-exports from the shared Anthropic helpers module.

This module preserves the public API for backward compatibility.
All logic lives in ``providers._anthropic_helpers``.
"""

from __future__ import annotations

from .providers._anthropic_helpers import (
    assistant_text,
    normalize_anthropic_tool_call_id,
    transform_messages_for_target,
)

__all__ = [
    "assistant_text",
    "normalize_anthropic_tool_call_id",
    "transform_messages_for_target",
]
