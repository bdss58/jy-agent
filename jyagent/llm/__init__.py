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
_PROVIDER_DEPENDENCIES = {
    "anthropic": ("anthropic", True),
    "openai": ("openai", False),
}


def _auto_register_providers():
    """Import provider modules for registration side effects."""
    for name, (dependency, required) in _PROVIDER_DEPENDENCIES.items():
        try:
            __import__(f"{__name__}.providers.{name}", fromlist=[name])
        except ImportError as exc:
            # Only dependency imports may be handled here.  Internal import
            # errors in provider modules must stay visible.
            if exc.name == dependency:
                if required:
                    raise RuntimeError(
                        f"Required LLM provider dependency '{dependency}' is not installed "
                        f"for provider '{name}'."
                    ) from exc
                continue
            raise

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
