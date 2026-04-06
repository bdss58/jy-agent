"""OpenAI Chat Completions provider adapter.

Implements the full RuntimeAdapter protocol using the OpenAI Python SDK
with the Chat Completions API (NOT the Responses API).
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import openai as _openai_sdk  # noqa: F401 — presence check
except ImportError:
    raise  # re-raise so _auto_register_providers catches it

import httpx

from ..core import register_adapter
from ..streams import ErrorStream, make_error_assistant_message
from ..types import (
    AssistantMessage,
    Context,
    ModelSpec,
    RuntimeOptions,
    RuntimeStream,
    Usage,
    compute_total_tokens,
)
from ._openai_helpers import (
    assistant_from_response,
    build_request_kwargs,
    map_stop_reason,
    usage_from_response,
)

# Register "openai" as a known provider in the config layer too.
from ...config import register_provider as _register_config_provider


# ─── _OpenAIStream ──────────────────────────────────────────────────────────

class _OpenAIStream(RuntimeStream):
    """Wraps an OpenAI streaming ChatCompletion response.

    Tracks content indices: text gets index 0, each tool_call gets subsequent
    indices.  Accumulates tool call name/id/arguments across deltas to build
    the final AssistantMessage on stream completion.
    """

    def __init__(self, stream: Any, model_spec: ModelSpec) -> None:
        self._stream = stream
        self._model_spec = model_spec
        self._final_message: AssistantMessage | None = None
        self._closed = False

    def __iter__(self):
        yield {"type": "start"}

        # Accumulated state for building the final message
        text_parts: list[str] = []
        # tool_calls keyed by their index in the delta.tool_calls array
        # Each entry: {"id": str, "name": str, "arguments": str}
        tool_calls_acc: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        response_id: str = ""
        usage_data: Any = None

        # Track content_index: text always gets 0 (if present),
        # tool calls get 1, 2, 3... (offset by 1 if text is present)
        text_started = False
        # Map from OpenAI tool_call index -> our content_index
        tool_call_content_indices: dict[int, int] = {}
        next_content_index = 0
        tool_call_started: set[int] = set()
        tool_call_ended: set[int] = set()

        try:
            for chunk in self._stream:
                # Capture response-level metadata
                if not response_id and hasattr(chunk, "id") and chunk.id:
                    response_id = chunk.id

                # Capture usage if present (some APIs include it on the last chunk)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_data = chunk.usage

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                # Capture finish_reason
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta is None:
                    continue

                # Text content
                if delta.content is not None:
                    if not text_started:
                        text_started = True
                        next_content_index = 1
                        yield {"type": "text_start", "content_index": 0}
                    text_parts.append(delta.content)
                    yield {"type": "text_delta", "text": delta.content, "content_index": 0}

                # Tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        tc_index = tc_delta.index

                        # Assign a content_index for this tool call if we haven't yet
                        if tc_index not in tool_call_content_indices:
                            # If text was emitted, text has index 0, tool calls start at 1
                            # If no text, tool calls start at 0
                            base = 1 if text_started else 0
                            tool_call_content_indices[tc_index] = base + tc_index
                            if base + tc_index >= next_content_index:
                                next_content_index = base + tc_index + 1

                        content_idx = tool_call_content_indices[tc_index]

                        # Initialize accumulator for this tool call
                        if tc_index not in tool_calls_acc:
                            tool_calls_acc[tc_index] = {"id": "", "name": "", "arguments": ""}

                        acc = tool_calls_acc[tc_index]

                        # Accumulate id/name from first delta
                        if tc_delta.id:
                            acc["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            acc["name"] = tc_delta.function.name

                        # Emit tool_call_start when we first see this tool call
                        if tc_index not in tool_call_started:
                            tool_call_started.add(tc_index)
                            # End text block if it was open
                            if text_started and 0 not in tool_call_ended:
                                # We only end text once, before first tool call
                                pass  # text end is emitted after the loop
                            yield {"type": "tool_call_start", "content_index": content_idx}

                        # Accumulate arguments
                        if tc_delta.function and tc_delta.function.arguments:
                            acc["arguments"] += tc_delta.function.arguments
                            yield {
                                "type": "tool_call_delta",
                                "delta": tc_delta.function.arguments,
                                "content_index": content_idx,
                            }

        except Exception as err:
            self._final_message = make_error_assistant_message(
                self._model_spec, err, api="openai-chat",
            )
            yield {"type": "error", "message": self._final_message}
            return

        # End text block if it was started
        if text_started:
            yield {"type": "text_end", "content_index": 0}

        # End all tool call blocks
        for tc_index in sorted(tool_call_started):
            content_idx = tool_call_content_indices[tc_index]
            yield {"type": "tool_call_end", "content_index": content_idx}

        # Build the final AssistantMessage from accumulated data
        content: list[dict[str, Any]] = []
        if text_parts:
            content.append({"type": "text", "text": "".join(text_parts)})
        for tc_index in sorted(tool_calls_acc):
            acc = tool_calls_acc[tc_index]
            try:
                parsed_args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except (json.JSONDecodeError, TypeError):
                parsed_args = {}
            content.append({
                "type": "tool_call",
                "id": acc["id"],
                "name": acc["name"],
                "arguments": parsed_args,
            })

        usage = usage_from_response(usage_data)

        self._final_message = {
            "role": "assistant",
            "content": content,
            "provider": self._model_spec.provider,
            "api": "openai-chat",
            "model": self._model_spec.model,
            "stop_reason": map_stop_reason(finish_reason),
            "usage": usage,
            "response_id": response_id,
            "id": response_id,
        }

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
            try:
                self._stream.close()
            except Exception:
                pass
        self._closed = True

    def __enter__(self) -> _OpenAIStream:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ─── OpenAIAdapter ──────────────────────────────────────────────────────────

class OpenAIAdapter:
    provider = "openai"
    api_name = "openai-chat"

    def __init__(self) -> None:
        self._cached_client: _openai_sdk.OpenAI | None = None
        self._cached_base_url: str | None = None
        self._cached_api_key: str | None = None

    def _client(self) -> _openai_sdk.OpenAI:
        base_url = os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if (
            self._cached_client is not None
            and self._cached_base_url == base_url
            and self._cached_api_key == api_key
        ):
            return self._cached_client
        kwargs: dict[str, Any] = {"http_client": httpx.Client(verify=False)}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._cached_client = _openai_sdk.OpenAI(**kwargs)
        self._cached_base_url = base_url
        self._cached_api_key = api_key
        return self._cached_client

    def stream(
        self,
        model_spec: ModelSpec,
        context: Context,
        options: RuntimeOptions | None = None,
    ) -> RuntimeStream:
        options = options or RuntimeOptions()
        kwargs = build_request_kwargs(model_spec, context, options)
        timeout = options.timeout
        try:
            client = self._client()
            # Request stream_options to get usage in the final chunk
            stream = client.chat.completions.create(
                **kwargs,
                stream=True,
                stream_options={"include_usage": True},
                timeout=timeout,
            )
        except Exception as err:
            return ErrorStream(model_spec, err)
        return _OpenAIStream(stream, model_spec)

    def complete(
        self,
        model_spec: ModelSpec,
        context: Context,
        options: RuntimeOptions | None = None,
    ) -> AssistantMessage:
        options = options or RuntimeOptions()
        kwargs = build_request_kwargs(model_spec, context, options)
        timeout = options.timeout
        try:
            client = self._client()
            response = client.chat.completions.create(**kwargs, timeout=timeout)
        except Exception:
            raise
        return assistant_from_response(model_spec, response)


register_adapter(OpenAIAdapter())
_register_config_provider("openai")
