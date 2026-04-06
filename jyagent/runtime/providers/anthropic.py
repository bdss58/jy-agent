from __future__ import annotations

import os
from typing import Any

import anthropic
import httpx

from ..core import register_adapter
from ..types import AssistantMessage, Context, ModelSpec, RuntimeOptions, RuntimeStream
from ._anthropic_helpers import (
    assistant_from_response,
    build_request_kwargs,
    make_error_assistant_message,
)


# ─── _ErrorStream ─────────────────────────────────────────────────────────────

class _ErrorStream:
    """A ``RuntimeStream`` that immediately yields a terminal error event."""

    def __init__(self, model_spec: ModelSpec, error: BaseException) -> None:
        self._message = make_error_assistant_message(model_spec, error)

    def __iter__(self):
        yield {"type": "start"}
        yield {"type": "error", "message": self._message}

    def get_final_message(self) -> AssistantMessage:
        return self._message

    def close(self) -> None:
        pass

    def __enter__(self) -> _ErrorStream:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


# ─── _AnthropicStream ────────────────────────────────────────────────────────

class _AnthropicStream(RuntimeStream):
    def __init__(self, stream_cm: Any, model_spec: ModelSpec):
        self._stream_cm = stream_cm
        self._stream: Any = None
        self._model_spec = model_spec
        self._final_message: AssistantMessage | None = None
        self._closed = False

    def __iter__(self):
        # Enter the SDK context manager at iteration start so failures
        # are captured as error events rather than raised from __init__.
        if self._stream is None:
            try:
                self._stream = self._stream_cm.__enter__()
            except Exception as err:
                self._final_message = make_error_assistant_message(self._model_spec, err)
                yield {"type": "start"}
                yield {"type": "error", "message": self._final_message}
                return

        yield {"type": "start"}

        block_types: dict[int, str] = {}

        try:
            for event in self._stream:
                etype = getattr(event, "type", None)

                if etype == "content_block_start":
                    idx = event.index
                    block = event.content_block
                    btype = getattr(block, "type", "")
                    block_types[idx] = btype
                    if btype == "text":
                        yield {"type": "text_start", "content_index": idx}
                    elif btype in ("thinking", "redacted_thinking"):
                        yield {"type": "thinking_start", "content_index": idx}
                    elif btype == "tool_use":
                        yield {"type": "tool_call_start", "content_index": idx}

                elif etype == "content_block_delta":
                    idx = event.index
                    delta = event.delta
                    if hasattr(delta, "text"):
                        yield {"type": "text_delta", "text": delta.text, "content_index": idx}
                    elif getattr(delta, "type", None) == "thinking_delta":
                        yield {"type": "thinking_delta", "text": delta.thinking, "content_index": idx}
                    elif getattr(delta, "type", None) == "input_json_delta":
                        yield {"type": "tool_call_delta", "delta": delta.partial_json, "content_index": idx}

                elif etype == "content_block_stop":
                    idx = event.index
                    btype = block_types.get(idx, "")
                    if btype == "text":
                        yield {"type": "text_end", "content_index": idx}
                    elif btype in ("thinking", "redacted_thinking"):
                        yield {"type": "thinking_end", "content_index": idx}
                    elif btype == "tool_use":
                        yield {"type": "tool_call_end", "content_index": idx}

        except Exception as err:
            self._final_message = make_error_assistant_message(self._model_spec, err)
            yield {"type": "error", "message": self._final_message}
            return

        # Successful completion — resolve final message from the SDK.
        try:
            raw_final = self._stream.get_final_message()
            self._final_message = assistant_from_response(self._model_spec, raw_final)
        except Exception as err:
            self._final_message = make_error_assistant_message(self._model_spec, err)
            yield {"type": "error", "message": self._final_message}
            return

        yield {"type": "done", "message": self._final_message}

    def get_final_message(self) -> AssistantMessage:
        if self._final_message is not None:
            return self._final_message
        # Drive iteration to completion — terminal event always sets _final_message.
        for _ in self:
            pass
        assert self._final_message is not None
        return self._final_message

    def close(self) -> None:
        if self._closed:
            return
        if self._stream is not None:
            self._stream_cm.__exit__(None, None, None)
        self._closed = True

    def __enter__(self) -> _AnthropicStream:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

class AnthropicAdapter:
    provider = "anthropic"
    api_name = "anthropic-messages"

    def __init__(self) -> None:
        self._cached_client: anthropic.Anthropic | None = None
        self._cached_base_url: str | None = None
        self._cached_auth_token: str | None = None

    def _client(self) -> anthropic.Anthropic:
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
        if (
            self._cached_client is not None
            and self._cached_base_url == base_url
            and self._cached_auth_token == auth_token
        ):
            return self._cached_client
        kwargs: dict[str, Any] = {"http_client": httpx.Client(verify=False)}
        if base_url:
            kwargs["base_url"] = base_url
        if auth_token:
            kwargs["api_key"] = auth_token
        self._cached_client = anthropic.Anthropic(**kwargs)
        self._cached_base_url = base_url
        self._cached_auth_token = auth_token
        return self._cached_client

    def stream(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> RuntimeStream:
        options = options or RuntimeOptions()
        kwargs = build_request_kwargs(model_spec, context, options)
        timeout = options.timeout
        try:
            client = self._client()
            stream_cm = client.messages.stream(**kwargs, timeout=timeout)
        except Exception as err:
            return _ErrorStream(model_spec, err)
        return _AnthropicStream(stream_cm, model_spec)

    def complete(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> AssistantMessage:
        options = options or RuntimeOptions()
        kwargs = build_request_kwargs(model_spec, context, options)
        timeout = options.timeout
        try:
            client = self._client()
            response = client.messages.create(**kwargs, timeout=timeout)
        except Exception:
            raise
        return assistant_from_response(model_spec, response)


register_adapter(AnthropicAdapter())
