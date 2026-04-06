from .core import RuntimeOwner, get_adapter, list_adapters, register_adapter
from .types import (
    AssistantMessage,
    AnthropicReasoningConfig,
    AnthropicThinkingAdaptiveConfig,
    AnthropicThinkingDisabledConfig,
    Context,
    Message,
    ModelSpec,
    ReasoningConfig,
    RuntimeOptions,
    StopReason,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    ToolChoice,
    ToolResultMessage,
    Usage,
    compute_total_tokens,
)

# Import provider modules for registration side effects.
# Each provider module calls register_adapter() at import time.
def _auto_register_providers():
    """Import provider modules for registration side effects, skip missing deps."""
    _provider_modules = ["anthropic", "openai"]  # Add "gemini" etc. here later
    for name in _provider_modules:
        try:
            __import__(f"{__name__}.providers.{name}", fromlist=[name])
        except ImportError:
            pass

_auto_register_providers()

__all__ = [
    "AssistantMessage",
    "AnthropicReasoningConfig",
    "AnthropicThinkingAdaptiveConfig",
    "AnthropicThinkingDisabledConfig",
    "Context",
    "Message",
    "ModelSpec",
    "ReasoningConfig",
    "RuntimeOptions",
    "RuntimeOwner",
    "StopReason",
    "StreamDoneEvent",
    "StreamErrorEvent",
    "StreamEvent",
    "ToolChoice",
    "ToolResultMessage",
    "Usage",
    "compute_total_tokens",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
