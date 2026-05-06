"""Runtime validation for the normalized message / context / stream-event shapes.

Background
----------
The provider-neutral message types in ``jyagent.llm.types`` are
``TypedDict``s — they give static type checkers something to verify but
provide **zero** runtime guarantees.  ``LLMClient.complete`` /
``LLMClient.stream`` returns a plain ``dict`` at runtime and the engine
trusts the adapter to honor the documented shape.

When a provider SDK changes its response format, or a new adapter ships
with a typo, the loop silently consumes the wrong shape — there is no
``KeyError`` because ``message.get("usage", {})`` swallows it, no
``TypeError`` because dicts have no static contract, and the bug
manifests downstream as a token-count regression, a missing
``stop_reason`` branch, or a corrupted history.

Codex review (2026-05) flagged this as the highest-residual coupling in
the runtime: *"Replace normalized message dict conventions with typed
models plus validation at provider boundaries.  This will catch
adapter drift before it corrupts loop state."*

This module supplies the runtime half.  Each ``validate_*`` function:

  * accepts ``Any`` (caller's untyped output)
  * narrows to the corresponding ``TypedDict`` on success
  * raises :class:`MessageValidationError` with a precise pointer-style
    path on failure (``"messages[3].content[1].arguments"``) — never a
    bare ``KeyError`` or ``TypeError``.

Validators are pure functions, allocate nothing extra, and run in
microseconds.  They are designed to be called at provider boundaries
(after the adapter decodes the wire format, before the engine consumes
the result) — gated by ``LoopConfig.validate_provider_output`` so
production runs pay zero cost when the flag is off.

What gets validated
-------------------
Six shapes have engine consumers and therefore validators:

==============================  =================================
Shape                           Validator
==============================  =================================
``Context`` (request)           :func:`validate_context`
``Message`` (any role)          :func:`validate_message`
``AssistantMessage`` (response) :func:`validate_assistant_message`
``ToolResultMessage``           :func:`validate_tool_result_message`
``UserMessage``                 :func:`validate_user_message`
``StreamEvent``                 :func:`validate_stream_event`
==============================  =================================

Content blocks (TextBlock / ThinkingBlock / ToolCallBlock) are
validated transitively when validating an ``AssistantMessage``;
they are not exposed as standalone validators because no boundary
hands a content block to the engine in isolation.

Forward compatibility
---------------------
The validators are **strict on required fields** and **lenient on
optional fields** — unknown keys are tolerated (a newer provider may
return a field the engine doesn't read yet).  If a key is required by
the TypedDict definition, it must be present and well-typed; if it is
optional or marked ``total=False``, missing is allowed but presence
forces a type check.  This matches the ``LoopCheckpoint.from_json``
forward-compat philosophy already in the codebase.
"""

from __future__ import annotations

from typing import Any, cast

from .types import (
    AssistantMessage,
    Context,
    Message,
    StreamEvent,
    ToolResultMessage,
    UserMessage,
)


__all__ = [
    "MessageValidationError",
    "validate_assistant_message",
    "validate_context",
    "validate_message",
    "validate_stream_event",
    "validate_tool_result_message",
    "validate_user_message",
]


class MessageValidationError(TypeError):
    """Raised when an adapter / caller hands the engine a malformed shape.

    Inherits from ``TypeError`` (not ``ValueError``) because the failure
    is **shape-level**, not a semantic value error.  Engine code that
    catches ``TypeError`` from API SDK paths (e.g., the existing retry
    layer in ``llm_runner``) will surface validator failures the same
    way as adapter-level malformed-response errors.

    The ``path`` argument records the dotted/indexed location of the
    first offending field, e.g. ``"context.messages[3].content[1].id"``
    — useful when the caller dumps a 100-message context and the engine
    rejects it with one error.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _check_dict(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise MessageValidationError(path, f"expected dict, got {type(value).__name__}")
    return value


def _check_str(value: Any, path: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise MessageValidationError(path, f"expected str, got {type(value).__name__}")
    if not allow_empty and not value:
        raise MessageValidationError(path, "must be non-empty")
    return value


def _check_list(value: Any, path: str) -> list:
    if not isinstance(value, list):
        raise MessageValidationError(path, f"expected list, got {type(value).__name__}")
    return value


def _check_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise MessageValidationError(path, f"expected bool, got {type(value).__name__}")
    return value


def _check_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MessageValidationError(path, f"expected int, got {type(value).__name__}")
    return value


def _check_required(d: dict, key: str, path: str) -> Any:
    if key not in d:
        raise MessageValidationError(f"{path}.{key}", "required key missing")
    return d[key]


# ─── Content blocks (used transitively by validate_assistant_message) ────────


_VALID_BLOCK_TYPES = {"text", "thinking", "tool_call"}


def _validate_text_block(block: dict, path: str) -> None:
    text = _check_required(block, "text", path)
    _check_str(text, f"{path}.text")


def _validate_thinking_block(block: dict, path: str) -> None:
    # ThinkingBlock has every field optional except `type` (Required[Literal])
    # — provider variants populate different subsets (Anthropic uses
    # `thinking` + `signature`; OpenAI summary mode populates `summary`).
    if "thinking" in block:
        _check_str(block["thinking"], f"{path}.thinking")
    if "signature" in block:
        _check_str(block["signature"], f"{path}.signature")
    if "redacted" in block:
        _check_bool(block["redacted"], f"{path}.redacted")
    if "encrypted_content" in block:
        _check_str(block["encrypted_content"], f"{path}.encrypted_content")
    if "summary" in block:
        summary = _check_list(block["summary"], f"{path}.summary")
        for i, item in enumerate(summary):
            _check_str(item, f"{path}.summary[{i}]")


def _validate_tool_call_block(block: dict, path: str) -> None:
    tc_id = _check_required(block, "id", path)
    _check_str(tc_id, f"{path}.id", allow_empty=False)
    name = _check_required(block, "name", path)
    _check_str(name, f"{path}.name", allow_empty=False)
    arguments = _check_required(block, "arguments", path)
    _check_dict(arguments, f"{path}.arguments")


_BLOCK_VALIDATORS = {
    "text": _validate_text_block,
    "thinking": _validate_thinking_block,
    "tool_call": _validate_tool_call_block,
}


def _validate_content_block(block: Any, path: str) -> None:
    block_d = _check_dict(block, path)
    block_type = _check_required(block_d, "type", path)
    _check_str(block_type, f"{path}.type")
    validator = _BLOCK_VALIDATORS.get(block_type)
    if validator is None:
        # Forward-compat: unknown block types are tolerated. The engine's
        # consumers only special-case known types and pass-through the rest.
        return
    validator(block_d, path)


# ─── Message-level validators ────────────────────────────────────────────────


def validate_user_message(value: Any, path: str = "message") -> UserMessage:
    """Validate a ``UserMessage`` shape and narrow the type."""
    d = _check_dict(value, path)
    role = _check_required(d, "role", path)
    if role != "user":
        raise MessageValidationError(f"{path}.role", f"expected 'user', got {role!r}")
    content = _check_required(d, "content", path)
    _check_str(content, f"{path}.content")
    return cast(UserMessage, d)


def validate_tool_result_message(value: Any, path: str = "message") -> ToolResultMessage:
    """Validate a ``ToolResultMessage`` shape and narrow the type."""
    d = _check_dict(value, path)
    role = _check_required(d, "role", path)
    if role != "tool_result":
        raise MessageValidationError(f"{path}.role", f"expected 'tool_result', got {role!r}")
    tc_id = _check_required(d, "tool_call_id", path)
    _check_str(tc_id, f"{path}.tool_call_id", allow_empty=False)
    name = _check_required(d, "tool_name", path)
    _check_str(name, f"{path}.tool_name")
    # `content` may legally be a string or a list of content parts (some
    # providers return structured tool_result payloads).  Accept either.
    content = _check_required(d, "content", path)
    if not isinstance(content, (str, list)):
        raise MessageValidationError(
            f"{path}.content",
            f"expected str or list, got {type(content).__name__}",
        )
    is_error = _check_required(d, "is_error", path)
    _check_bool(is_error, f"{path}.is_error")
    return cast(ToolResultMessage, d)


_VALID_STOP_REASONS = {"stop", "length", "tool_use", "error", "aborted"}


def validate_assistant_message(value: Any, path: str = "message") -> AssistantMessage:
    """Validate an ``AssistantMessage`` shape and narrow the type.

    Required fields: ``role == "assistant"``, ``content`` (list).
    Optional: ``stop_reason`` (must be one of the documented values),
    ``usage`` (dict if present), plus provider-specific extras passed
    through unchecked.
    """
    d = _check_dict(value, path)
    role = _check_required(d, "role", path)
    if role != "assistant":
        raise MessageValidationError(f"{path}.role", f"expected 'assistant', got {role!r}")
    content = _check_required(d, "content", path)
    blocks = _check_list(content, f"{path}.content")
    for i, block in enumerate(blocks):
        _validate_content_block(block, f"{path}.content[{i}]")
    if "stop_reason" in d:
        sr = d["stop_reason"]
        _check_str(sr, f"{path}.stop_reason")
        if sr not in _VALID_STOP_REASONS:
            raise MessageValidationError(
                f"{path}.stop_reason",
                f"unknown stop_reason {sr!r}; expected one of {sorted(_VALID_STOP_REASONS)}",
            )
    if "usage" in d:
        usage = _check_dict(d["usage"], f"{path}.usage")
        for usage_key in ("input_tokens", "output_tokens",
                          "cache_creation_input_tokens", "cache_read_input_tokens",
                          "total_tokens"):
            if usage_key in usage:
                _check_int(usage[usage_key], f"{path}.usage.{usage_key}")
    return cast(AssistantMessage, d)


def validate_message(value: Any, path: str = "message") -> Message:
    """Validate any message (user / assistant / tool_result) by its role."""
    d = _check_dict(value, path)
    role = _check_required(d, "role", path)
    if role == "user":
        return validate_user_message(d, path)
    if role == "assistant":
        return validate_assistant_message(d, path)
    if role == "tool_result":
        return validate_tool_result_message(d, path)
    raise MessageValidationError(
        f"{path}.role",
        f"unknown role {role!r}; expected 'user' | 'assistant' | 'tool_result'",
    )


def validate_context(value: Any, path: str = "context") -> Context:
    """Validate a request ``Context`` (system_prompt + messages [+ tools])."""
    d = _check_dict(value, path)
    if "system_prompt" in d:
        _check_str(d["system_prompt"], f"{path}.system_prompt")
    messages = _check_required(d, "messages", path)
    msg_list = _check_list(messages, f"{path}.messages")
    for i, msg in enumerate(msg_list):
        validate_message(msg, f"{path}.messages[{i}]")
    if "tools" in d:
        tools = _check_list(d["tools"], f"{path}.tools")
        for i, tool in enumerate(tools):
            _check_dict(tool, f"{path}.tools[{i}]")
    return cast(Context, d)


# ─── Stream events ───────────────────────────────────────────────────────────


_STREAM_EVENT_TYPES = {
    "start", "text_start", "text_delta", "text_end",
    "thinking_start", "thinking_delta", "thinking_end",
    "tool_call_start", "tool_call_delta", "tool_call_end",
    "done", "error",
}


def validate_stream_event(value: Any, path: str = "event") -> StreamEvent:
    """Validate a stream event yielded by ``LLMClient.stream``.

    All variants share ``type``; terminal events (``done`` / ``error``)
    additionally require ``message`` (an ``AssistantMessage``).  Delta
    events with text payloads require ``text``; tool-call deltas
    require ``delta``.  Unknown event types are rejected — unlike
    content blocks, the engine's stream loop dispatches on type
    explicitly and an unknown type would silently drop content.
    """
    d = _check_dict(value, path)
    ev_type = _check_required(d, "type", path)
    _check_str(ev_type, f"{path}.type")
    if ev_type not in _STREAM_EVENT_TYPES:
        raise MessageValidationError(
            f"{path}.type",
            f"unknown stream event type {ev_type!r}; expected one of {sorted(_STREAM_EVENT_TYPES)}",
        )
    if ev_type in {"done", "error"}:
        msg = _check_required(d, "message", path)
        validate_assistant_message(msg, f"{path}.message")
    elif ev_type in {"text_delta", "thinking_delta"}:
        if "text" in d:
            _check_str(d["text"], f"{path}.text")
    elif ev_type == "tool_call_delta":
        if "delta" in d:
            _check_str(d["delta"], f"{path}.delta")
    # All other events have no required payload beyond `type`.
    return cast(StreamEvent, d)
