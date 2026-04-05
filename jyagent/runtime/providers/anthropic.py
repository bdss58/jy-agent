from __future__ import annotations

import logging
import os
from typing import Any

import anthropic
import httpx

from ...observability import LLMCallLogger, new_call_id, summarize_runtime_context
from ..core import register_adapter
from ..history import transform_messages_for_target
from ..reasoning import validate_anthropic_thinking
from ..types import AssistantMessage, Context, Message, ModelSpec, RuntimeOptions, RuntimeStream, Usage


logger = logging.getLogger(__name__)


def _usage_from_response(usage: Any) -> Usage:
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def _map_stop_reason(reason: str | None) -> str:
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "tool_use"
    if reason in {"end_turn", "stop_sequence", "pause_turn", None}:
        return "stop"
    if reason in {"refusal", "sensitive"}:
        return "error"
    return "error"


def _thinking_block(block: Any) -> dict[str, Any]:
    if block.type == "redacted_thinking":
        return {
            "type": "thinking",
            "thinking": "",
            "signature": getattr(block, "data", ""),
            "redacted": True,
        }
    return {
        "type": "thinking",
        "thinking": getattr(block, "thinking", ""),
        "signature": getattr(block, "signature", ""),
    }


def _assistant_from_response(model_spec: ModelSpec, response: Any) -> AssistantMessage:
    content = []
    message_id = getattr(response, "id", "")
    for block in response.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type in {"thinking", "redacted_thinking"}:
            content.append(_thinking_block(block))
        elif block.type == "tool_use":
            content.append({
                "type": "tool_call",
                "id": block.id,
                "name": block.name,
                "arguments": block.input or {},
            })
    return {
        "role": "assistant",
        "content": content,
        "provider": model_spec.provider,
        "api": "anthropic-messages",
        "model": model_spec.model,
        "stop_reason": _map_stop_reason(getattr(response, "stop_reason", None)),
        "usage": _usage_from_response(getattr(response, "usage", None)),
        "response_id": message_id,
        "id": message_id,
    }


def _convert_assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for block in message.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "thinking":
            if block.get("redacted") and block.get("signature"):
                blocks.append({"type": "redacted_thinking", "data": block["signature"]})
            elif block.get("signature"):
                blocks.append({
                    "type": "thinking",
                    "thinking": block.get("thinking", ""),
                    "signature": block["signature"],
                })
            elif block.get("thinking", "").strip():
                blocks.append({"type": "text", "text": block["thinking"]})
        elif block_type == "tool_call":
            blocks.append({
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": block.get("arguments", {}),
            })
    return blocks


def _convert_messages(model_spec: ModelSpec, messages: list[Message]) -> list[dict[str, Any]]:
    transformed = transform_messages_for_target(messages, model_spec)
    out: list[dict[str, Any]] = []
    idx = 0
    while idx < len(transformed):
        message = transformed[idx]
        role = message.get("role")
        if role == "user":
            out.append({"role": "user", "content": message.get("content", "")})
            idx += 1
            continue
        if role == "assistant":
            out.append({"role": "assistant", "content": _convert_assistant_blocks(message)})
            idx += 1
            continue
        tool_results = []
        while idx < len(transformed) and transformed[idx].get("role") == "tool_result":
            tool_result = transformed[idx]
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_result["tool_call_id"],
                "content": tool_result["content"],
                "is_error": tool_result["is_error"],
            })
            idx += 1
        out.append({"role": "user", "content": tool_results})
    return out


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema", {"type": "object", "properties": {}}),
        }
        for tool in tools
    ]


class _AnthropicStream(RuntimeStream):
    def __init__(self, stream_cm: Any, model_spec: ModelSpec, call_logger: LLMCallLogger | None = None):
        self._stream_cm = stream_cm
        self._call_logger = call_logger
        try:
            self._stream = stream_cm.__enter__()
        except Exception as err:
            if self._call_logger is not None:
                self._call_logger.failed(err, stage="stream_enter")
            raise
        self._model_spec = model_spec
        self._final_message: AssistantMessage | None = None
        self._event_count = 0
        self._text_delta_chars = 0
        self._thinking_delta_chars = 0
        self._tool_call_delta_chars = 0
        self._closed = False

    def __iter__(self):
        try:
            for event in self._stream:
                self._event_count += 1
                if getattr(event, "type", None) != "content_block_delta":
                    continue
                delta = event.delta
                if hasattr(delta, "text"):
                    self._text_delta_chars += len(getattr(delta, "text", "") or "")
                    yield {"type": "text_delta", "text": delta.text}
                elif getattr(delta, "type", None) == "thinking_delta":
                    self._thinking_delta_chars += len(getattr(delta, "thinking", "") or "")
                    yield {"type": "thinking_delta", "text": delta.thinking}
                elif getattr(delta, "type", None) == "input_json_delta":
                    self._tool_call_delta_chars += len(getattr(delta, "partial_json", "") or "")
                    yield {"type": "tool_call_delta", "delta": delta.partial_json}
        except Exception as err:
            if self._call_logger is not None:
                self._call_logger.failed(err, stage="stream_iter", stream_state=self.log_snapshot())
            raise

    def get_final_message(self) -> AssistantMessage:
        if self._final_message is None:
            try:
                self._final_message = _assistant_from_response(self._model_spec, self._stream.get_final_message())
            except Exception as err:
                if self._call_logger is not None:
                    self._call_logger.failed(err, stage="final_message", stream_state=self.log_snapshot())
                raise
            if self._call_logger is not None:
                self._call_logger.succeeded(self._final_message, stream_state=self.log_snapshot())
        return self._final_message

    def close(self) -> None:
        if self._closed:
            return
        self._stream_cm.__exit__(None, None, None)
        self._closed = True

    def log_snapshot(self) -> dict[str, Any]:
        return {
            "event_count": self._event_count,
            "text_delta_chars": self._text_delta_chars,
            "thinking_delta_chars": self._thinking_delta_chars,
            "tool_call_delta_chars": self._tool_call_delta_chars,
        }


class AnthropicAdapter:
    provider = "anthropic"
    api_name = "anthropic-messages"

    def _client(self) -> anthropic.Anthropic:
        kwargs: dict[str, Any] = {"http_client": httpx.Client(verify=False)}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
        if base_url:
            kwargs["base_url"] = base_url
        if auth_token:
            kwargs["api_key"] = auth_token
        return anthropic.Anthropic(**kwargs)

    def _request_kwargs(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> dict[str, Any]:
        options = options or RuntimeOptions()
        kwargs: dict[str, Any] = {
            "model": model_spec.model,
            "max_tokens": options.max_output_tokens,
            "messages": _convert_messages(model_spec, context.get("messages", [])),
        }
        if options.reasoning is not None:
            kwargs["thinking"] = validate_anthropic_thinking(
                options.reasoning,
                max_output_tokens=options.max_output_tokens,
            )
        if context.get("system_prompt"):
            kwargs["system"] = context["system_prompt"]
        tools = _convert_tools(context.get("tools"))
        if tools:
            kwargs["tools"] = tools
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
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
            stream_cm = client.messages.stream(**kwargs, timeout=timeout)
        except Exception as err:
            call_logger.failed(err, stage="stream_open")
            raise
        return _AnthropicStream(stream_cm, model_spec, call_logger=call_logger)

    def complete(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> AssistantMessage:
        stream = self.stream(model_spec, context, options)
        try:
            for _event in stream:
                pass
            return stream.get_final_message()
        finally:
            stream.close()


register_adapter(AnthropicAdapter())
