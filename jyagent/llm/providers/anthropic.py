from __future__ import annotations

import os
from typing import Any

import anthropic
import httpx

from ..core import register_adapter
from ..streams import BaseStream, ErrorStream, make_error_assistant_message
from ..types import AssistantMessage, Context, ModelSpec, LLMOptions, LLMStream
from ._anthropic_helpers import (
    assistant_from_response,
    build_request_kwargs,
)


# ─── _AnthropicStream ────────────────────────────────────────────────────────

class _AnthropicStream(BaseStream):
    def __init__(self, stream_cm: Any, model_spec: ModelSpec):
        super().__init__(stream_cm, model_spec)

    def __iter__(self):
        if self._consumed:
            raise RuntimeError("Stream already consumed")
        self._consumed = True

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
            self.close()
            return

        # Successful completion — resolve final message from the SDK.
        try:
            raw_final = self._stream.get_final_message()
            self._final_message = assistant_from_response(self._model_spec, raw_final)
        except Exception as err:
            self._final_message = make_error_assistant_message(self._model_spec, err)
            yield {"type": "error", "message": self._final_message}
            self.close()
            return

        yield {"type": "done", "message": self._final_message}
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
        kwargs: dict[str, Any] = {
            "http_client": httpx.Client(
                verify=os.environ.get("SSL_VERIFY", "1").lower() not in ("0", "false", "no"),
            ),
        }
        if base_url:
            kwargs["base_url"] = base_url
        if auth_token:
            kwargs["api_key"] = auth_token
        self._cached_client = anthropic.Anthropic(**kwargs)
        self._cached_base_url = base_url
        self._cached_auth_token = auth_token
        return self._cached_client

    def stream(self, model_spec: ModelSpec, context: Context, options: LLMOptions | None = None) -> LLMStream:
        options = options or LLMOptions()
        kwargs = build_request_kwargs(model_spec, context, options)
        timeout = options.timeout
        try:
            client = self._client()
            stream_cm = client.messages.stream(**kwargs, timeout=timeout)
        except Exception as err:
            return ErrorStream(model_spec, err)
        return _AnthropicStream(stream_cm, model_spec)

    def complete(self, model_spec: ModelSpec, context: Context, options: LLMOptions | None = None) -> AssistantMessage:
        options = options or LLMOptions()
        kwargs = build_request_kwargs(model_spec, context, options)
        timeout = options.timeout
        client = self._client()
        response = client.messages.create(**kwargs, timeout=timeout)
        return assistant_from_response(model_spec, response)


register_adapter(AnthropicAdapter())
