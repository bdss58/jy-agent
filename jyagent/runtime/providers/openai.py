from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import Any

import httpx
from openai import OpenAI

from ...observability import LLMCallLogger, log_event, new_call_id, scrub_string, summarize_runtime_context
from ..core import register_adapter
from ..history import transform_messages_for_target
from ..reasoning import validate_openai_reasoning
from ..types import AssistantMessage, Context, Message, ModelSpec, RuntimeOptions, RuntimeStream


logger = logging.getLogger(__name__)


def _json_error_details(error: BaseException) -> dict[str, Any]:
    if not isinstance(error, json.JSONDecodeError):
        return {}

    raw_doc = getattr(error, "doc", "") or ""
    details: dict[str, Any] = {
        "json_error_line": error.lineno,
        "json_error_column": error.colno,
        "json_error_position": error.pos,
        "stream_payload_kind": _stream_payload_kind(raw_doc),
        "stream_payload_summary": _stream_payload_summary(_stream_payload_kind(raw_doc)),
    }
    snippet = _json_error_snippet(error)
    if snippet:
        details["json_error_snippet"] = scrub_string(snippet)
    return details


def _json_error_snippet(error: json.JSONDecodeError, radius: int = 120) -> str:
    doc = getattr(error, "doc", "") or ""
    if not doc:
        return ""

    pos = min(max(getattr(error, "pos", 0), 0), len(doc))
    start = max(0, pos - radius)
    end = min(len(doc), pos + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(doc) else ""
    return f"{prefix}{doc[start:end]}{suffix}"


def _stream_payload_kind(doc: str) -> str:
    text = (doc or "").lower()
    if not text:
        return "unknown"
    if '"event"' in text and "keepalive" in text:
        return "keepalive"
    if '"type":"keepalive"' in text or '"type": "keepalive"' in text:
        return "keepalive"
    if "function_call" in text or '"arguments"' in text:
        return "tool_call_delta"
    if "output_text" in text or "text.delta" in text:
        return "text_delta"
    if "reasoning" in text:
        return "reasoning"
    return "unknown"


def _stream_payload_summary(kind: str) -> str:
    summaries = {
        "keepalive": "Malformed keepalive stream payload.",
        "tool_call_delta": "Malformed tool-call delta stream payload.",
        "text_delta": "Malformed text-delta stream payload.",
        "reasoning": "Malformed reasoning stream payload.",
        "unknown": "Malformed unclassified stream payload.",
    }
    return summaries.get(kind, summaries["unknown"])


def _attach_stream_error_diagnostics(
    error: BaseException,
    *,
    response_id: str = "",
    **extra: Any,
) -> BaseException:
    if not isinstance(error, json.JSONDecodeError):
        return error

    diagnostics = getattr(error, "stream_diagnostics", None)
    if not isinstance(diagnostics, dict):
        diagnostics = _json_error_details(error)
    if response_id:
        diagnostics["response_id"] = response_id
    for key, value in extra.items():
        if value not in (None, ""):
            diagnostics[key] = value
    setattr(error, "stream_diagnostics", diagnostics)
    return error


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def _parse_tool_call_arguments(arguments: Any) -> dict[str, Any] | None:
    if not isinstance(arguments, str) or not arguments.strip():
        return None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _snapshot_discard_reason(response: Any) -> str | None:
    output = getattr(response, "output", None) or []
    if not output:
        return "snapshot contains no output items"

    saw_message = False
    saw_function_call = False
    saw_reasoning = False
    saw_unknown = False

    for item in output:
        item_type = getattr(item, "type", "")
        if item_type == "message":
            saw_message = True
            for part in getattr(item, "content", None) or []:
                part_type = getattr(part, "type", "")
                if part_type == "output_text" and str(getattr(part, "text", "") or "").strip():
                    return None
                if part_type == "refusal" and str(getattr(part, "refusal", "") or "").strip():
                    return None
        elif item_type == "function_call":
            saw_function_call = True
            if _parse_tool_call_arguments(getattr(item, "arguments", None)) is not None:
                return None
        elif item_type == "reasoning":
            saw_reasoning = True
        else:
            saw_unknown = True

    if saw_function_call:
        return "snapshot function_call arguments were missing, non-object, or invalid JSON"
    if saw_message:
        return "snapshot message items contained no assistant-visible text"
    if saw_reasoning:
        return "snapshot contained reasoning-only output"
    if saw_unknown:
        return "snapshot contained unsupported output items only"
    return "snapshot contained no usable assistant output"


def _message_item(message_id: str, texts: list[str], phase: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text} for text in texts],
    }
    if phase:
        item["phase"] = phase
    return item


def _assistant_items(message: dict[str, Any], sequence_prefix: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    text_chunks: list[str] = []
    message_id = message.get("id") or f"asst_{sequence_prefix}"
    text_index = 0

    def flush_text() -> None:
        nonlocal text_chunks, text_index
        if not text_chunks:
            return
        current_id = message_id if text_index == 0 else f"{message_id}_{text_index}"
        items.append(_message_item(current_id, text_chunks, message.get("phase")))
        text_chunks = []
        text_index += 1

    for block in message.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            text_chunks.append(block.get("text", ""))
            continue
        flush_text()
        if block_type == "thinking":
            summary = [{"type": "summary_text", "text": text} for text in block.get("summary", [])]
            reasoning_item: dict[str, Any] = {
                "id": block.get("id") or f"rs_{sequence_prefix}_{len(items)}",
                "type": "reasoning",
                "summary": summary,
            }
            if block.get("thinking", "").strip():
                reasoning_item["content"] = [{"type": "reasoning_text", "text": block["thinking"]}]
            if block.get("encrypted_content"):
                reasoning_item["encrypted_content"] = block["encrypted_content"]
            items.append(reasoning_item)
        elif block_type == "tool_call":
            items.append({
                "type": "function_call",
                "call_id": block["id"],
                "name": block["name"],
                "arguments": json.dumps(block.get("arguments", {}), ensure_ascii=False),
            })
    flush_text()
    return items


def _convert_messages(model_spec: ModelSpec, messages: list[Message]) -> list[dict[str, Any]]:
    transformed = transform_messages_for_target(messages, model_spec)
    out: list[dict[str, Any]] = []
    for index, message in enumerate(transformed):
        role = message.get("role")
        if role == "user":
            out.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": message.get("content", "")}],
            })
        elif role == "assistant":
            out.extend(_assistant_items(message, str(index)))
        elif role == "tool_result":
            content = message["content"]
            if message.get("is_error") and not content.startswith("Error:"):
                content = f"Error: {content}"
            out.append({
                "type": "function_call_output",
                "call_id": message["tool_call_id"],
                "output": content,
            })
    return out


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            "strict": False,
        }
        for tool in tools
    ]


def _stop_reason(response: Any, content: list[dict[str, Any]]) -> str:
    if getattr(response, "error", None) is not None:
        return "error"
    if any(block.get("type") == "tool_call" for block in content):
        return "tool_use"
    incomplete = getattr(response, "incomplete_details", None)
    incomplete_reason = getattr(incomplete, "reason", None)
    if incomplete_reason == "max_output_tokens":
        return "length"
    if incomplete_reason == "content_filter":
        return "error"
    return "stop"


def _assistant_from_response(model_spec: ModelSpec, response: Any) -> AssistantMessage:
    content: list[dict[str, Any]] = []
    message_id = ""
    phase = None
    for item in getattr(response, "output", []):
        if item.type == "message":
            if not message_id:
                message_id = getattr(item, "id", "")
            if phase is None:
                phase = getattr(item, "phase", None)
            for part in item.content:
                if part.type == "output_text":
                    content.append({"type": "text", "text": part.text})
                elif part.type == "refusal":
                    content.append({"type": "text", "text": part.refusal})
        elif item.type == "reasoning":
            thinking = ""
            if getattr(item, "content", None):
                thinking = "".join(part.text for part in item.content if getattr(part, "type", "") == "reasoning_text")
            content.append({
                "type": "thinking",
                "thinking": thinking,
                "id": item.id,
                "summary": [part.text for part in getattr(item, "summary", [])],
                "encrypted_content": getattr(item, "encrypted_content", None) or "",
                "redacted": bool(getattr(item, "encrypted_content", None) and not thinking),
            })
        elif item.type == "function_call":
            try:
                arguments = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            content.append({
                "type": "tool_call",
                "id": item.call_id,
                "name": item.name,
                "arguments": arguments,
            })
    message: AssistantMessage = {
        "role": "assistant",
        "content": content,
        "provider": model_spec.provider,
        "api": "openai-responses",
        "model": model_spec.model,
        "stop_reason": _stop_reason(response, content),
        "usage": _response_usage(response),
        "response_id": getattr(response, "id", ""),
        "id": message_id or getattr(response, "id", ""),
    }
    if phase:
        message["phase"] = phase
    return message


class _OpenAIStream(RuntimeStream):
    _MISSING_COMPLETED_EVENT = "Didn't receive a `response.completed` event."

    def __init__(
        self,
        manager: Any,
        model_spec: ModelSpec,
        responses_api: Any,
        timeout: float | None = None,
        call_logger: LLMCallLogger | None = None,
    ):
        self._manager = manager
        self._call_logger = call_logger
        try:
            self._stream = manager.__enter__()
        except Exception as err:
            if self._call_logger is not None:
                self._call_logger.failed(err, stage="stream_enter")
            raise
        self._model_spec = model_spec
        self._responses_api = responses_api
        self._timeout = timeout
        self._final_message: AssistantMessage | None = None
        self._response_id = ""
        self._incomplete_details = None
        self._event_count = 0
        self._text_delta_chars = 0
        self._tool_call_delta_chars = 0
        self._recovery_method = ""
        self._stream_error: BaseException | None = None
        self._stream_error_stage = ""
        self._closed = False

    def __iter__(self):
        try:
            for event in self._stream:
                self._event_count += 1
                event_type = getattr(event, "type", "")
                response = getattr(event, "response", None)
                if response is not None and not self._response_id:
                    self._response_id = getattr(response, "id", "") or self._response_id
                if event_type == "response.incomplete":
                    self._incomplete_details = getattr(response, "incomplete_details", None)
                    if self._incomplete_details is None:
                        self._incomplete_details = getattr(event, "incomplete_details", None)
                if event_type == "response.output_text.delta":
                    self._text_delta_chars += len(getattr(event, "delta", "") or "")
                    yield {"type": "text_delta", "text": event.delta}
                elif event_type == "response.function_call_arguments.delta":
                    self._tool_call_delta_chars += len(getattr(event, "delta", "") or "")
                    yield {"type": "tool_call_delta", "delta": event.delta}
        except Exception as err:
            if isinstance(err, json.JSONDecodeError):
                self._stream_error = _attach_stream_error_diagnostics(
                    err,
                    response_id=self._response_id,
                    stage="stream_iter",
                )
                self._stream_error_stage = "stream_iter"
                return
            if self._call_logger is not None:
                self._call_logger.failed(
                    err,
                    stage="stream_iter",
                    response_id=self._response_id,
                    stream_state=self.log_snapshot(),
                    **_json_error_details(err),
                )
            raise

    def get_final_message(self) -> AssistantMessage:
        if self._final_message is None:
            try:
                if self._stream_error is not None:
                    raise self._stream_error
                response = self._stream.get_final_response()
                self._final_message = _assistant_from_response(self._model_spec, response)
            except (RuntimeError, json.JSONDecodeError) as err:
                recovered = self._recover_response(err)
                if recovered is None:
                    _attach_stream_error_diagnostics(
                        err,
                        response_id=self._response_id,
                        stage=self._stream_error_stage or "final_response",
                    )
                    if self._call_logger is not None:
                        self._call_logger.failed(
                            err,
                            stage=self._stream_error_stage or "final_response",
                            response_id=self._response_id,
                            stream_state=self.log_snapshot(),
                            **_json_error_details(err),
                        )
                    raise
                response, warning = recovered
                self._response_id = getattr(response, "id", "") or self._response_id
                if self._recovery_method == "stream_snapshot":
                    discard_reason = _snapshot_discard_reason(response)
                    if discard_reason is not None:
                        _attach_stream_error_diagnostics(
                            err,
                            response_id=self._response_id,
                            stage=self._stream_error_stage or "final_response",
                            recovery_method=self._recovery_method,
                            discard_reason=discard_reason,
                        )
                        self._log_discarded_recovery(discard_reason)
                        if self._call_logger is not None:
                            self._call_logger.failed(
                                err,
                                stage=self._stream_error_stage or "final_response",
                                response_id=self._response_id,
                                stream_state=self.log_snapshot(),
                                recovery_method=self._recovery_method,
                                discard_reason=discard_reason,
                                **_json_error_details(err),
                            )
                        raise
                self._final_message = _assistant_from_response(self._model_spec, response)
                self._final_message["runtime_warnings"] = [warning]
            except Exception as err:
                if self._call_logger is not None:
                    self._call_logger.failed(
                        err,
                        stage="final_response",
                        response_id=self._response_id,
                        stream_state=self.log_snapshot(),
                    )
                raise
            if self._call_logger is not None:
                self._call_logger.succeeded(self._final_message, stream_state=self.log_snapshot())
        return self._final_message

    def close(self) -> None:
        if self._closed:
            return
        self._manager.__exit__(None, None, None)
        self._closed = True

    def _log_discarded_recovery(self, reason: str) -> None:
        log_event(
            logger,
            logging.WARNING,
            "llm.request.recovery_discarded",
            call_id=self._call_logger.call_id if self._call_logger is not None else "",
            provider=self._model_spec.provider,
            api="openai-responses",
            model=self._model_spec.model,
            response_id=self._response_id,
            recovery_method=self._recovery_method or None,
            reason=reason,
            stream_state=self.log_snapshot(),
        )

    def log_snapshot(self) -> dict[str, Any]:
        incomplete_reason = None
        if self._incomplete_details is not None:
            incomplete_reason = getattr(self._incomplete_details, "reason", None) or str(self._incomplete_details)
        return {
            "response_id": self._response_id,
            "event_count": self._event_count,
            "text_delta_chars": self._text_delta_chars,
            "tool_call_delta_chars": self._tool_call_delta_chars,
            "incomplete_reason": incomplete_reason,
            "recovery_method": self._recovery_method or None,
        }

    def _recover_response(self, error: BaseException) -> tuple[Any, str] | None:
        if isinstance(error, json.JSONDecodeError):
            retrieved = self._retrieve_response()
            if retrieved is not None:
                self._recovery_method = "responses.retrieve"
                return (
                    retrieved,
                    "Recovered OpenAI stream after malformed SSE JSON via responses.retrieve().",
                )

            snapshot = self._snapshot_response()
            if snapshot is not None:
                self._recovery_method = "stream_snapshot"
                return (
                    snapshot,
                    "Recovered OpenAI stream after malformed SSE JSON from partial stream snapshot.",
                )

            return None

        if self._MISSING_COMPLETED_EVENT not in str(error):
            return None

        retrieved = self._retrieve_response()
        if retrieved is not None:
            self._recovery_method = "responses.retrieve"
            return (
                retrieved,
                "Recovered OpenAI stream after missing terminal event via responses.retrieve().",
            )

        snapshot = self._snapshot_response()
        if snapshot is not None:
            self._recovery_method = "stream_snapshot"
            return (
                snapshot,
                "Recovered OpenAI stream after missing terminal event from partial stream snapshot.",
            )

        return None

    def _retrieve_response(self) -> Any | None:
        if not self._response_id:
            return None
        retrieve = getattr(self._responses_api, "retrieve", None)
        if not callable(retrieve):
            return None
        try:
            return retrieve(response_id=self._response_id, timeout=self._timeout)
        except Exception:
            return None

    def _snapshot_response(self) -> Any | None:
        state = getattr(self._stream, "_state", None)
        if state is None:
            return None
        snapshot = getattr(state, "_ResponseStreamState__current_snapshot", None)
        if snapshot is None:
            return None

        output = getattr(snapshot, "output", None) or []
        response_id = getattr(snapshot, "id", "") or self._response_id
        if not output:
            return None

        return SimpleNamespace(
            id=response_id,
            output=output,
            usage=getattr(snapshot, "usage", None),
            error=getattr(snapshot, "error", None),
            incomplete_details=(
                getattr(snapshot, "incomplete_details", None)
                or self._incomplete_details
            ),
        )


class OpenAIAdapter:
    provider = "openai"
    api_name = "openai-responses"

    def _client(self) -> OpenAI:
        kwargs: dict[str, Any] = {"http_client": httpx.Client(verify=False)}
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        organization = os.environ.get("OPENAI_ORGANIZATION")
        project = os.environ.get("OPENAI_PROJECT")
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization
        if project:
            kwargs["project"] = project
        return OpenAI(**kwargs)

    def _request_kwargs(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> dict[str, Any]:
        options = options or RuntimeOptions()
        kwargs: dict[str, Any] = {
            "model": model_spec.model,
            "input": _convert_messages(model_spec, context.get("messages", [])),
            "tools": _convert_tools(context.get("tools")),
            "instructions": context.get("system_prompt"),
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": options.max_output_tokens,
            "parallel_tool_calls": True,
        }
        if options.reasoning is not None:
            kwargs["reasoning"] = validate_openai_reasoning(options.reasoning)
        kwargs = {k: v for k, v in kwargs.items() if v not in (None, [], {})}
        if options.tool_choice is not None:
            kwargs["tool_choice"] = options.tool_choice
        return kwargs

    def stream(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> RuntimeStream:
        options = options or RuntimeOptions()
        kwargs = self._request_kwargs(model_spec, context, options)
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
            manager = client.responses.stream(**kwargs, timeout=timeout)
        except Exception as err:
            call_logger.failed(err, stage="stream_open")
            raise
        return _OpenAIStream(
            manager,
            model_spec,
            client.responses,
            timeout=timeout,
            call_logger=call_logger,
        )

    def complete(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> AssistantMessage:
        options = options or RuntimeOptions()
        kwargs = self._request_kwargs(model_spec, context, options)
        kwargs["stream"] = False
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
            response = client.responses.create(**kwargs, timeout=timeout)
        except Exception as err:
            call_logger.failed(err, stage="complete_create")
            raise
        try:
            message = _assistant_from_response(model_spec, response)
        except Exception as err:
            call_logger.failed(
                err,
                stage="complete_response",
                response_id=getattr(response, "id", ""),
            )
            raise
        call_logger.succeeded(message)
        return message


register_adapter(OpenAIAdapter())
