from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol, Required, TypedDict


StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]


class Usage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_tokens: int


def compute_total_tokens(usage: Usage) -> int:
    return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ThinkingBlock(TypedDict, total=False):
    type: Required[Literal["thinking"]]
    thinking: str
    signature: str
    redacted: bool
    id: str
    summary: list[str]
    encrypted_content: str
    status: str


class ToolCallBlock(TypedDict):
    type: Literal["tool_call"]
    id: str
    name: str
    arguments: dict[str, Any]


AssistantContentBlock = TextBlock | ThinkingBlock | ToolCallBlock


class UserMessage(TypedDict):
    role: Literal["user"]
    content: str


class AssistantMessage(TypedDict, total=False):
    role: Required[Literal["assistant"]]
    content: Required[list[AssistantContentBlock]]
    provider: str
    api: str
    model: str
    stop_reason: StopReason
    usage: Usage
    response_id: str
    id: str
    phase: Literal["commentary", "final_answer"]
    llm_warnings: list[str]
    error_message: str


class ToolResultMessage(TypedDict):
    role: Literal["tool_result"]
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool


Message = UserMessage | AssistantMessage | ToolResultMessage


class Context(TypedDict, total=False):
    system_prompt: str
    messages: list[Message]
    tools: list[dict[str, Any]]


# ─── Stream events ────────────────────────────────────────────────────────────

class StreamStartEvent(TypedDict):
    type: Literal["start"]


class TextStartEvent(TypedDict):
    type: Literal["text_start"]
    content_index: int


class TextDeltaEvent(TypedDict):
    type: Literal["text_delta"]
    text: str
    content_index: int


class TextEndEvent(TypedDict):
    type: Literal["text_end"]
    content_index: int


class ThinkingStartEvent(TypedDict):
    type: Literal["thinking_start"]
    content_index: int


class ThinkingDeltaEvent(TypedDict):
    type: Literal["thinking_delta"]
    text: str
    content_index: int


class ThinkingEndEvent(TypedDict):
    type: Literal["thinking_end"]
    content_index: int


class ToolCallStartEvent(TypedDict):
    type: Literal["tool_call_start"]
    content_index: int


class ToolCallDeltaEvent(TypedDict):
    type: Literal["tool_call_delta"]
    delta: str
    content_index: int


class ToolCallEndEvent(TypedDict):
    type: Literal["tool_call_end"]
    content_index: int


class StreamDoneEvent(TypedDict):
    type: Literal["done"]
    message: AssistantMessage


class StreamErrorEvent(TypedDict):
    type: Literal["error"]
    message: AssistantMessage


StreamEvent = (
    StreamStartEvent
    | TextStartEvent | TextDeltaEvent | TextEndEvent
    | ThinkingStartEvent | ThinkingDeltaEvent | ThinkingEndEvent
    | ToolCallStartEvent | ToolCallDeltaEvent | ToolCallEndEvent
    | StreamDoneEvent | StreamErrorEvent
)


# ─── Reasoning config ────────────────────────────────────────────────────────
# Anthropic-specific types are defined in providers/_anthropic_reasoning.py
# and re-exported here for backward compatibility.

from .providers._anthropic_reasoning import (
    AnthropicThinkingDisabledConfig,
    AnthropicThinkingAdaptiveConfig,
    AnthropicReasoningConfig,
)


class OpenAIReasoningConfig(TypedDict, total=False):
    effort: Literal["minimal", "none", "low", "medium", "high", "xhigh"]


# Union of all provider reasoning configs — extend as providers are added.
ReasoningConfig = AnthropicReasoningConfig | OpenAIReasoningConfig


# ─── Tool choice ──────────────────────────────────────────────────────────────

class ToolChoiceAuto(TypedDict):
    type: Literal["auto"]


class ToolChoiceAny(TypedDict):
    type: Literal["any"]


class ToolChoiceNone(TypedDict):
    type: Literal["none"]


class ToolChoiceTool(TypedDict):
    type: Literal["tool"]
    name: str


ToolChoice = ToolChoiceAuto | ToolChoiceAny | ToolChoiceNone | ToolChoiceTool


# ─── Model / options / stream ────────────────────────────────────────────────
#
# ``ModelSpec`` and ``LLMOptions`` were moved into
# ``jyagent.runtime.loop.llm_types`` as part of closing the
# runtime → llm dependency reversal.  The runtime owns the *shape* of these
# inputs because they
# are constructed by the engine and consumed by the LLM client; placing
# them on the consumer side rather than the producer side reverses the
# import direction.  We re-export here so existing
# ``from jyagent.llm.types import LLMOptions, ModelSpec`` continues to
# work — this is a pure reorganisation, not an API break.
from ..runtime.loop.llm_types import LLMOptions, ModelSpec  # noqa: F401


class LLMStream(Protocol):
    """Sync streaming interface.

    Contract:
    - ``__iter__`` always emits exactly one terminal event (``done`` or ``error``)
      as the last yielded value.
    - After a terminal event, ``get_final_message()`` returns the corresponding
      ``AssistantMessage`` and **never raises**.
    - Provider/network failures after ``stream()`` returns are represented as
      ``error`` events.  Only local validation errors may raise before stream
      creation.
    """

    def __iter__(self) -> Iterator[StreamEvent]:
        ...

    def get_final_message(self) -> AssistantMessage:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> LLMStream:
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...
