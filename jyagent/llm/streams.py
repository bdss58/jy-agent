"""Shared stream utilities — reusable by all provider adapters.

No provider-specific imports.
"""

from __future__ import annotations

from typing import Any

from .types import AssistantMessage, ModelSpec, LLMStream


def make_error_assistant_message(
    model_spec: ModelSpec,
    error: BaseException,
    *,
    api: str | None = None,
    partial_content: list[dict] | None = None,
) -> AssistantMessage:
    """Build a normalized error AssistantMessage from any exception."""
    error_text = f"[{type(error).__name__}] {error}"
    msg: AssistantMessage = {
        "role": "assistant",
        "content": partial_content or [{"type": "text", "text": error_text}],
        "provider": model_spec.provider,
        "model": model_spec.model,
        "stop_reason": "error",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "error_message": error_text,
    }
    if api:
        msg["api"] = api
    return msg


class BaseStream:
    """Shared stream infrastructure for provider adapters."""

    def __init__(self, stream_cm: Any, model_spec: ModelSpec) -> None:
        self._stream_cm = stream_cm
        self._stream: Any = None
        self._model_spec = model_spec
        self._final_message: AssistantMessage | None = None
        self._closed = False
        self._consumed = False

    def get_final_message(self) -> AssistantMessage:
        if self._final_message is not None:
            return self._final_message
        for _ in self:
            pass
        assert self._final_message is not None
        return self._final_message

    def close(self) -> None:
        if self._closed:
            return
        if self._stream is not None:
            try:
                self._stream_cm.__exit__(None, None, None)
            except Exception:
                pass
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class ErrorStream(LLMStream):
    """A ``LLMStream`` that immediately yields a terminal error event.

    Reusable by any provider adapter when stream creation itself fails.
    """

    def __init__(self, model_spec: ModelSpec, error: BaseException) -> None:
        self._message = make_error_assistant_message(model_spec, error)

    def __iter__(self):
        yield {"type": "start"}
        yield {"type": "error", "message": self._message}

    def get_final_message(self) -> AssistantMessage:
        return self._message

    def close(self) -> None:
        pass

    def __enter__(self) -> ErrorStream:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


__all__ = ["BaseStream", "ErrorStream", "make_error_assistant_message"]
