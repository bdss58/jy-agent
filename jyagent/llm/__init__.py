from .core import LLMOwner, get_adapter, list_adapters, register_adapter
from .streams import ErrorStream, make_error_assistant_message
from .types import (
    AssistantMessage,
    AnthropicReasoningConfig,
    AnthropicThinkingAdaptiveConfig,
    AnthropicThinkingDisabledConfig,
    Context,
    Message,
    ModelSpec,
    OpenAIReasoningConfig,
    ReasoningConfig,
    LLMOptions,
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
    _provider_modules = ["anthropic", "openai"]
    for name in _provider_modules:
        try:
            __import__(f"{__name__}.providers.{name}", fromlist=[name])
        except ImportError as exc:
            # Only skip if the top-level SDK package is missing.
            # Re-raise if it's an internal import error within our code.
            module_name = f"{__name__}.providers.{name}"
            if exc.name and not exc.name.startswith(module_name):
                pass  # SDK not installed — skip this provider
            else:
                raise  # Bug inside provider module — don't swallow

_auto_register_providers()

__all__ = [
    "AssistantMessage",
    "AnthropicReasoningConfig",
    "AnthropicThinkingAdaptiveConfig",
    "AnthropicThinkingDisabledConfig",
    "Context",
    "ErrorStream",
    "Message",
    "ModelSpec",
    "OpenAIReasoningConfig",
    "ReasoningConfig",
    "LLMOptions",
    "LLMOwner",
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
    "make_error_assistant_message",
    "register_adapter",
]
