"""Message-history helpers — re-exports for backward compatibility.

All functions now live in the provider-neutral ``messages`` module.
"""

from __future__ import annotations

from .messages import (
    assistant_text,
    normalize_anthropic_tool_call_id,
    transform_messages_for_target,
)

__all__ = [
    "assistant_text",
    "normalize_anthropic_tool_call_id",
    "transform_messages_for_target",
]
