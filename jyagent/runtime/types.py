from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol, TypedDict


ProviderName = Literal["anthropic", "openai"]
StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]


class Usage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_tokens: int


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ThinkingBlock(TypedDict, total=False):
    type: Literal["thinking"]
    thinking: str
    signature: str
    redacted: bool
    id: str
    summary: list[str]
    encrypted_content: str


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
    role: Literal["assistant"]
    content: list[AssistantContentBlock]
    provider: str
    api: str
    model: str
    stop_reason: StopReason
    usage: Usage
    response_id: str
    id: str
    phase: Literal["commentary", "final_answer"]
    runtime_warnings: list[str]


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


class TextDeltaEvent(TypedDict):
    type: Literal["text_delta"]
    text: str


class ThinkingDeltaEvent(TypedDict):
    type: Literal["thinking_delta"]
    text: str


class ToolCallDeltaEvent(TypedDict):
    type: Literal["tool_call_delta"]
    delta: str


StreamEvent = TextDeltaEvent | ThinkingDeltaEvent | ToolCallDeltaEvent


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class RuntimeOptions:
    max_output_tokens: int | None = None
    timeout: float | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] | None = None
    tool_choice: Any = None


class RuntimeStream(Protocol):
    def __iter__(self) -> Iterator[StreamEvent]:
        ...

    def get_final_message(self) -> AssistantMessage:
        ...

    def close(self) -> None:
        ...
