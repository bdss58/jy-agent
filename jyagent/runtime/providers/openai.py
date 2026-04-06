"""OpenAI Responses API provider adapter.

Implements the full RuntimeAdapter protocol using the OpenAI Python SDK
with the Responses API (``client.responses.create()`` / ``.stream()``).
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
    """Wraps an OpenAI Responses API streaming response.

    The stream context manager (``ResponseStreamManager``) yields
    ``ResponseStreamEvent`` objects.  We map these to our normalized
    ``StreamEvent`` types while accumulating state for the final
    ``AssistantMessage``.
    """

    def __init__(self, stream_cm: Any, model_spec: ModelSpec) -> None:
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
                self._final_message = make_error_assistant_message(
                    self._model_spec, err, api="openai-responses",
                )
                yield {"type": "start"}
                yield {"type": "error", "message": self._final_message}
                return

        yield {"type": "start"}

        # Maps output_index → content_index in our normalized event stream
        output_index_to_content_index: dict[int, int] = {}
        next_content_index = 0
        # Track output item types for end events
        output_index_types: dict[int, str] = {}

        # Accumulated state for building the final message from events
        text_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, str]] = {}  # output_index → {call_id, name, arguments}
        thinking_parts: dict[int, list[str]] = {}  # output_index → summary text parts
        final_response: Any = None

        try:
            for event in self._stream:
                etype = getattr(event, "type", None)

                # ── Message output item added ──
                if etype == "response.output_item.added":
                    item = getattr(event, "item", None)
                    item_type = getattr(item, "type", None)
                    output_index = getattr(event, "output_index", 0)

                    ci = next_content_index
                    output_index_to_content_index[output_index] = ci
                    next_content_index += 1
                    output_index_types[output_index] = item_type or ""

                    if item_type == "function_call":
                        call_id = getattr(item, "call_id", "")
                        name = getattr(item, "name", "")
                        tool_calls_acc[output_index] = {
                            "call_id": call_id,
                            "name": name,
                            "arguments": "",
                        }
                        yield {"type": "tool_call_start", "content_index": ci}

                    elif item_type == "reasoning":
                        thinking_parts[output_index] = []
                        yield {"type": "thinking_start", "content_index": ci}

                    # For "message" type, we wait for content_part.added

                # ── Content part added (within a message output item) ──
                elif etype == "response.content_part.added":
                    output_index = getattr(event, "output_index", 0)
                    part = getattr(event, "part", None)
                    part_type = getattr(part, "type", None)

                    if part_type in ("output_text", "text"):
                        # Assign content index if the message output wasn't yet tracked
                        if output_index not in output_index_to_content_index:
                            ci = next_content_index
                            output_index_to_content_index[output_index] = ci
                            next_content_index += 1
                        else:
                            ci = output_index_to_content_index[output_index]
                        output_index_types[output_index] = "message"
                        yield {"type": "text_start", "content_index": ci}

                # ── Text delta ──
                elif etype == "response.output_text.delta":
                    output_index = getattr(event, "output_index", 0)
                    delta = getattr(event, "delta", "")
                    ci = output_index_to_content_index.get(output_index, 0)
                    text_parts.append(delta)
                    yield {"type": "text_delta", "text": delta, "content_index": ci}

                # ── Content part done ──
                elif etype == "response.content_part.done":
                    output_index = getattr(event, "output_index", 0)
                    part = getattr(event, "part", None)
                    part_type = getattr(part, "type", None)
                    if part_type in ("output_text", "text"):
                        ci = output_index_to_content_index.get(output_index, 0)
                        yield {"type": "text_end", "content_index": ci}

                # ── Function call arguments delta ──
                elif etype == "response.function_call_arguments.delta":
                    output_index = getattr(event, "output_index", 0)
                    delta = getattr(event, "delta", "")
                    ci = output_index_to_content_index.get(output_index, 0)
                    if output_index in tool_calls_acc:
                        tool_calls_acc[output_index]["arguments"] += delta
                    yield {"type": "tool_call_delta", "delta": delta, "content_index": ci}

                # ── Function call done ──
                elif etype == "response.output_item.done":
                    item = getattr(event, "item", None)
                    item_type = getattr(item, "type", None)
                    output_index = getattr(event, "output_index", 0)
                    ci = output_index_to_content_index.get(output_index, 0)

                    if item_type == "function_call":
                        # Update accumulated tool call with final data
                        if output_index in tool_calls_acc:
                            tool_calls_acc[output_index]["call_id"] = getattr(item, "call_id", tool_calls_acc[output_index]["call_id"])
                            tool_calls_acc[output_index]["name"] = getattr(item, "name", tool_calls_acc[output_index]["name"])
                            final_args = getattr(item, "arguments", "")
                            if final_args:
                                tool_calls_acc[output_index]["arguments"] = final_args
                        yield {"type": "tool_call_end", "content_index": ci}

                    elif item_type == "reasoning":
                        yield {"type": "thinking_end", "content_index": ci}

                # ── Reasoning summary text delta ──
                elif etype == "response.reasoning_summary_text.delta":
                    output_index = getattr(event, "output_index", 0)
                    delta = getattr(event, "delta", "")
                    ci = output_index_to_content_index.get(output_index, 0)
                    if output_index in thinking_parts:
                        thinking_parts[output_index].append(delta)
                    yield {"type": "thinking_delta", "text": delta, "content_index": ci}

                # ── Response completed ──
                elif etype == "response.completed":
                    final_response = getattr(event, "response", None)

                # ── Response failed / error ──
                elif etype in ("response.failed", "response.error"):
                    error_resp = getattr(event, "response", None)
                    error_msg = ""
                    if error_resp:
                        err_obj = getattr(error_resp, "error", None)
                        if err_obj:
                            error_msg = getattr(err_obj, "message", str(err_obj))
                    self._final_message = make_error_assistant_message(
                        self._model_spec,
                        RuntimeError(error_msg or "OpenAI Responses API stream failed"),
                        api="openai-responses",
                    )
                    yield {"type": "error", "message": self._final_message}
                    return

        except Exception as err:
            self._final_message = make_error_assistant_message(
                self._model_spec, err, api="openai-responses",
            )
            yield {"type": "error", "message": self._final_message}
            return

        # ── Build final AssistantMessage ──
        if final_response is not None:
            try:
                self._final_message = assistant_from_response(self._model_spec, final_response)
            except Exception as err:
                self._final_message = make_error_assistant_message(
                    self._model_spec, err, api="openai-responses",
                )
                yield {"type": "error", "message": self._final_message}
                return
        else:
            # Fallback: build from accumulated data
            content: list[dict[str, Any]] = []
            if text_parts:
                content.append({"type": "text", "text": "".join(text_parts)})
            for oi in sorted(thinking_parts):
                summary_text = "".join(thinking_parts[oi])
                if summary_text:
                    content.append({
                        "type": "thinking",
                        "thinking": summary_text,
                        "summary": [summary_text],
                    })
            for oi in sorted(tool_calls_acc):
                acc = tool_calls_acc[oi]
                try:
                    parsed_args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                except (json.JSONDecodeError, TypeError):
                    parsed_args = {}
                content.append({
                    "type": "tool_call",
                    "id": acc["call_id"],
                    "name": acc["name"],
                    "arguments": parsed_args,
                })

            has_tool_calls = bool(tool_calls_acc)
            self._final_message = {
                "role": "assistant",
                "content": content,
                "provider": self._model_spec.provider,
                "api": "openai-responses",
                "model": self._model_spec.model,
                "stop_reason": "tool_use" if has_tool_calls else "stop",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "response_id": "",
                "id": "",
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
                self._stream_cm.__exit__(None, None, None)
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
    api_name = "openai-responses"

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
            # client.responses.stream() returns a context manager;
            # no need to pass stream=True — the .stream() method handles it.
            stream_cm = client.responses.stream(**kwargs, timeout=timeout)
        except Exception as err:
            return ErrorStream(model_spec, err)
        return _OpenAIStream(stream_cm, model_spec)

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
            response = client.responses.create(**kwargs, timeout=timeout)
        except Exception:
            raise
        return assistant_from_response(model_spec, response)


register_adapter(OpenAIAdapter())
_register_config_provider("openai")
