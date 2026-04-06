from __future__ import annotations

import logging
import os
from typing import Any

import anthropic
import httpx

from ...observability import LLMCallLogger, new_call_id, summarize_runtime_context
from ..core import register_adapter
from ..types import AssistantMessage, Context, ModelSpec, RuntimeOptions, RuntimeStream
from ._anthropic_helpers import (
    assistant_from_response,
    build_request_kwargs,
    make_error_assistant_message,
)


logger = logging.getLogger(__name__)


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
    def __init__(self, stream_cm: Any, model_spec: ModelSpec, call_logger: LLMCallLogger | None = None):
        self._stream_cm = stream_cm
        self._call_logger = call_logger
        self._stream: Any = None
        self._model_spec = model_spec
        self._final_message: AssistantMessage | None = None
        self._event_count = 0
        self._text_delta_chars = 0
        self._thinking_delta_chars = 0
        self._tool_call_delta_chars = 0
        self._closed = False

    def __iter__(self):
        # Enter the SDK context manager at iteration start so failures
        # are captured as error events rather than raised from __init__.
        if self._stream is None:
            try:
                self._stream = self._stream_cm.__enter__()
            except Exception as err:
                if self._call_logger is not None:
                    self._call_logger.failed(err, stage="stream_enter")
                self._final_message = make_error_assistant_message(self._model_spec, err)
                yield {"type": "start"}
                yield {"type": "error", "message": self._final_message}
                return

        yield {"type": "start"}

        block_types: dict[int, str] = {}

        try:
            for event in self._stream:
                self._event_count += 1
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
                        self._text_delta_chars += len(getattr(delta, "text", "") or "")
                        yield {"type": "text_delta", "text": delta.text, "content_index": idx}
                    elif getattr(delta, "type", None) == "thinking_delta":
                        self._thinking_delta_chars += len(getattr(delta, "thinking", "") or "")
                        yield {"type": "thinking_delta", "text": delta.thinking, "content_index": idx}
                    elif getattr(delta, "type", None) == "input_json_delta":
                        self._tool_call_delta_chars += len(getattr(delta, "partial_json", "") or "")
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
            if self._call_logger is not None:
                self._call_logger.failed(err, stage="stream_iter", stream_state=self.log_snapshot())
            self._final_message = make_error_assistant_message(self._model_spec, err)
            yield {"type": "error", "message": self._final_message}
            return

        # Successful completion — resolve final message from the SDK.
        try:
            raw_final = self._stream.get_final_message()
            self._final_message = assistant_from_response(self._model_spec, raw_final)
        except Exception as err:
            if self._call_logger is not None:
                self._call_logger.failed(err, stage="final_message", stream_state=self.log_snapshot())
            self._final_message = make_error_assistant_message(self._model_spec, err)
            yield {"type": "error", "message": self._final_message}
            return

        if self._call_logger is not None:
            self._call_logger.succeeded(self._final_message, stream_state=self.log_snapshot())
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
        self._stream_cm.__exit__(None, None, None)
        self._closed = True

    def __enter__(self) -> _AnthropicStream:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def log_snapshot(self) -> dict[str, Any]:
        return {
            "event_count": self._event_count,
            "text_delta_chars": self._text_delta_chars,
            "thinking_delta_chars": self._thinking_delta_chars,
            "tool_call_delta_chars": self._tool_call_delta_chars,
        }


# ─── Adapter ──────────────────────────────────────────────────────────────────

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
        call_logger = LLMCallLogger(
            logger,
            call_id=new_call_id(),
            provider=self.provider,
            api=self.api_name,
            model=model_spec.model,
            metadata=options.metadata or {},
            request_summary=summarize_runtime_context(context, options),
            request_payload=kwargs,
        )
        call_logger.started()
        try:
            client = self._client()
            stream_cm = client.messages.stream(**kwargs, timeout=timeout)
        except Exception as err:
            call_logger.failed(err, stage="stream_open")
            return _ErrorStream(model_spec, err)
        return _AnthropicStream(stream_cm, model_spec, call_logger=call_logger)

    def complete(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> AssistantMessage:
        options = options or RuntimeOptions()
        kwargs = build_request_kwargs(model_spec, context, options)
        timeout = options.timeout
        call_logger = LLMCallLogger(
            logger,
            call_id=new_call_id(),
            provider=self.provider,
            api=self.api_name,
            model=model_spec.model,
            metadata=options.metadata or {},
            request_summary=summarize_runtime_context(context, options),
            request_payload=kwargs,
        )
        call_logger.started()
        try:
            client = self._client()
            response = client.messages.create(**kwargs, timeout=timeout)
        except Exception as err:
            call_logger.failed(err, stage="complete")
            raise
        message = assistant_from_response(model_spec, response)
        call_logger.succeeded(message)
        return message


register_adapter(AnthropicAdapter())
