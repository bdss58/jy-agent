"""Message-history helpers — re-exports for backward compatibility.

``assistant_text`` is now provided by the provider-neutral ``messages`` module.
Anthropic-specific helpers remain in ``providers._anthropic_helpers``.
"""

from __future__ import annotations

from .messages import assistant_text
from .providers._anthropic_helpers import (
    normalize_anthropic_tool_call_id,
    transform_messages_for_target,
)

__all__ = [
    "assistant_text",
    "normalize_anthropic_tool_call_id",
    "transform_messages_for_target",
]
