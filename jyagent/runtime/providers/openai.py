from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import httpx
from openai import OpenAI

from ..core import register_adapter
from ..history import transform_messages_for_target
from ..reasoning import validate_openai_reasoning
from ..types import AssistantMessage, Context, Message, ModelSpec, RuntimeOptions, RuntimeStream


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
    if incomplete and getattr(incomplete, "reason", None) == "max_output_tokens":
        return "length"
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

    def __init__(self, manager: Any, model_spec: ModelSpec, responses_api: Any, timeout: float | None = None):
        self._manager = manager
        self._stream = manager.__enter__()
        self._model_spec = model_spec
        self._responses_api = responses_api
        self._timeout = timeout
        self._final_message: AssistantMessage | None = None
        self._response_id = ""
        self._incomplete_details = None
        self._closed = False

    def __iter__(self):
        for event in self._stream:
            event_type = getattr(event, "type", "")
            response = getattr(event, "response", None)
            if response is not None and not self._response_id:
                self._response_id = getattr(response, "id", "") or self._response_id
            if event_type == "response.incomplete":
                self._incomplete_details = getattr(response, "incomplete_details", None)
                if self._incomplete_details is None:
                    self._incomplete_details = getattr(event, "incomplete_details", None)
            if event_type == "response.output_text.delta":
                yield {"type": "text_delta", "text": event.delta}
            elif event_type == "response.function_call_arguments.delta":
                yield {"type": "tool_call_delta", "delta": event.delta}

    def get_final_message(self) -> AssistantMessage:
        if self._final_message is None:
            try:
                response = self._stream.get_final_response()
                self._final_message = _assistant_from_response(self._model_spec, response)
            except RuntimeError as err:
                recovered = self._recover_response(err)
                if recovered is None:
                    raise
                response, warning = recovered
                self._final_message = _assistant_from_response(self._model_spec, response)
                self._final_message["runtime_warnings"] = [warning]
        return self._final_message

    def close(self) -> None:
        if self._closed:
            return
        self._manager.__exit__(None, None, None)
        self._closed = True

    def _recover_response(self, error: RuntimeError) -> tuple[Any, str] | None:
        if self._MISSING_COMPLETED_EVENT not in str(error):
            return None

        retrieved = self._retrieve_response()
        if retrieved is not None:
            return (
                retrieved,
                "Recovered OpenAI stream after missing terminal event via responses.retrieve().",
            )

        snapshot = self._snapshot_response()
        if snapshot is not None:
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
        client = self._client()
        kwargs = self._request_kwargs(model_spec, context, options)
        timeout = None if options is None else options.timeout
        return _OpenAIStream(client.responses.stream(**kwargs, timeout=timeout), model_spec, client.responses, timeout=timeout)

    def complete(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> AssistantMessage:
        stream = self.stream(model_spec, context, options)
        try:
            for _event in stream:
                pass
            return stream.get_final_message()
        finally:
            stream.close()


register_adapter(OpenAIAdapter())
