from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = os.path.join("data", "logs", "jyagent.jsonl")
DEFAULT_LOG_LLM_FAILURE_PAYLOADS = True
DEFAULT_LOG_MAX_TEXT_CHARS = 4000

_SENSITIVE_FIELD_MARKER = "[REDACTED]"
_SENSITIVE_KEYWORDS = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "token",
    "secret",
    "password",
    "passwd",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_log_path(path: str | None) -> Path:
    chosen = (path or os.environ.get("AGENT_LOG_FILE") or DEFAULT_LOG_FILE).strip() or DEFAULT_LOG_FILE
    file_path = Path(chosen)
    if not file_path.is_absolute():
        file_path = _project_root() / file_path
    return file_path


def get_log_settings() -> dict[str, Any]:
    return {
        "level": (os.environ.get("AGENT_LOG_LEVEL") or DEFAULT_LOG_LEVEL).strip().upper() or DEFAULT_LOG_LEVEL,
        "log_file": _resolve_log_path(os.environ.get("AGENT_LOG_FILE")),
        "llm_failure_payloads": _parse_bool(
            os.environ.get("AGENT_LOG_LLM_FAILURE_PAYLOADS"),
            DEFAULT_LOG_LLM_FAILURE_PAYLOADS,
        ),
        "max_text_chars": _parse_int(
            os.environ.get("AGENT_LOG_MAX_TEXT_CHARS"),
            DEFAULT_LOG_MAX_TEXT_CHARS,
        ),
    }


def _iso_timestamp(created: float) -> str:
    return datetime.fromtimestamp(created, tz=timezone.utc).isoformat()


def _is_sensitive_key(key: Any) -> bool:
    text = str(key).strip().lower()
    return any(token in text for token in _SENSITIVE_KEYWORDS)


def _secret_values() -> list[str]:
    values = set()
    for key, value in os.environ.items():
        if not value or len(value) < 8:
            continue
        if _is_sensitive_key(key):
            values.add(value)
    return sorted(values, key=len, reverse=True)


def truncate_text(value: str, max_chars: int | None = None) -> str:
    if max_chars is None:
        max_chars = get_log_settings()["max_text_chars"]
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}... [truncated {omitted} chars]"


def scrub_string(value: str, *, max_text_chars: int | None = None, secret_values: list[str] | None = None) -> str:
    text = value
    for secret in secret_values or _secret_values():
        text = text.replace(secret, _SENSITIVE_FIELD_MARKER)
    return truncate_text(text, max_chars=max_text_chars)


def sanitize_for_logging(
    value: Any,
    *,
    max_text_chars: int | None = None,
    secret_values: list[str] | None = None,
) -> Any:
    if max_text_chars is None:
        max_text_chars = get_log_settings()["max_text_chars"]
    if secret_values is None:
        secret_values = _secret_values()

    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                sanitized[str(key)] = _SENSITIVE_FIELD_MARKER
            else:
                sanitized[str(key)] = sanitize_for_logging(
                    item,
                    max_text_chars=max_text_chars,
                    secret_values=secret_values,
                )
        return sanitized

    if isinstance(value, list):
        return [
            sanitize_for_logging(item, max_text_chars=max_text_chars, secret_values=secret_values)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            sanitize_for_logging(item, max_text_chars=max_text_chars, secret_values=secret_values)
            for item in value
        ]

    if isinstance(value, str):
        return scrub_string(value, max_text_chars=max_text_chars, secret_values=secret_values)

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, Path):
        return scrub_string(str(value), max_text_chars=max_text_chars, secret_values=secret_values)

    if hasattr(value, "__dict__"):
        return sanitize_for_logging(
            vars(value),
            max_text_chars=max_text_chars,
            secret_values=secret_values,
        )

    return scrub_string(repr(value), max_text_chars=max_text_chars, secret_values=secret_values)


def summarize_runtime_context(context: dict[str, Any], options: Any = None) -> dict[str, Any]:
    messages = context.get("messages", []) or []
    role_counts: dict[str, int] = {}
    for message in messages:
        role = str(message.get("role", "unknown"))
        role_counts[role] = role_counts.get(role, 0) + 1

    tools = context.get("tools", []) or []
    summary: dict[str, Any] = {
        "message_count": len(messages),
        "message_roles": role_counts,
        "tool_count": len(tools),
        "tool_names": [tool.get("name", "") for tool in tools],
    }
    if options is not None:
        summary["max_output_tokens"] = getattr(options, "max_output_tokens", None)
        summary["timeout"] = getattr(options, "timeout", None)
        reasoning = getattr(options, "reasoning", None)
        if reasoning is not None:
            summary["reasoning"] = sanitize_for_logging(reasoning, max_text_chars=200)
    return summary


def summarize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content", []) or []
    tool_names = [block.get("name", "") for block in content if isinstance(block, dict) and block.get("type") == "tool_call"]
    output_text_chars = sum(
        len(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return {
        "response_id": message.get("response_id") or message.get("id", ""),
        "stop_reason": message.get("stop_reason"),
        "usage": sanitize_for_logging(message.get("usage", {}), max_text_chars=200),
        "tool_call_names": tool_names,
        "output_text_chars": output_text_chars,
        "runtime_warnings": list(message.get("runtime_warnings", []) or []),
        "phase": message.get("phase"),
    }


def new_call_id() -> str:
    return uuid4().hex


def format_traceback(error: BaseException) -> str:
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


def log_event(logger: logging.Logger, level: int, event: str, **payload: Any) -> None:
    logger.log(level, event, extra={"event": event, "payload": payload})


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "payload", {})
        event = getattr(record, "event", record.getMessage())
        data = {
            "timestamp": _iso_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "event": event,
            "payload": payload,
        }
        return json.dumps(data, ensure_ascii=False, default=str)


class HumanReadableFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", record.getMessage())
        payload = getattr(record, "payload", {}) or {}
        detail = (
            payload.get("error_message")
            or payload.get("warning")
            or payload.get("message")
            or payload.get("status")
            or payload.get("call_id")
            or ""
        )
        base = f"{record.levelname} {record.name} {event}"
        if detail:
            return f"{base}: {detail}"
        return base


def setup_logging(
    *,
    level: str | None = None,
    log_file: str | os.PathLike[str] | None = None,
    stderr_level: int = logging.WARNING,
) -> logging.Logger:
    settings = get_log_settings()
    resolved_level = (level or settings["level"]).strip().upper()
    resolved_path = _resolve_log_path(str(log_file) if log_file is not None else str(settings["log_file"]))

    logger = logging.getLogger("jyagent")
    logger.setLevel(getattr(logging, resolved_level, logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_jyagent_managed", False):
            logger.removeHandler(handler)
            handler.close()

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonLinesFormatter())
    file_handler._jyagent_managed = True  # type: ignore[attr-defined]

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(stderr_level)
    stream_handler.setFormatter(HumanReadableFormatter())
    stream_handler._jyagent_managed = True  # type: ignore[attr-defined]

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger._jyagent_logging_config = {  # type: ignore[attr-defined]
        "level": resolved_level,
        "log_file": str(resolved_path),
        "stderr_level": stderr_level,
    }
    return logger


class LLMCallLogger:
    def __init__(
        self,
        logger: logging.Logger,
        *,
        call_id: str,
        provider: str,
        api: str,
        model: str,
        metadata: dict[str, Any] | None = None,
        request_summary: dict[str, Any] | None = None,
        request_payload: dict[str, Any] | None = None,
    ):
        self._logger = logger
        self._call_id = call_id
        self._provider = provider
        self._api = api
        self._model = model
        self._metadata = metadata or {}
        self._request_summary = request_summary or {}
        self._request_payload = request_payload or {}
        self._finished = False
        settings = get_log_settings()
        self._log_failure_payloads = settings["llm_failure_payloads"]
        self._max_text_chars = settings["max_text_chars"]
        self._secret_values = _secret_values()

    @property
    def call_id(self) -> str:
        return self._call_id

    def started(self) -> None:
        log_event(
            self._logger,
            logging.INFO,
            "llm.request.started",
            call_id=self._call_id,
            provider=self._provider,
            api=self._api,
            model=self._model,
            metadata=sanitize_for_logging(self._metadata, max_text_chars=200, secret_values=self._secret_values),
            **self._request_summary,
        )

    def succeeded(self, message: dict[str, Any], **extra: Any) -> None:
        if self._finished:
            return
        payload = {
            "call_id": self._call_id,
            "provider": self._provider,
            "api": self._api,
            "model": self._model,
            "metadata": sanitize_for_logging(self._metadata, max_text_chars=200, secret_values=self._secret_values),
            **summarize_assistant_message(message),
        }
        if extra:
            payload.update(sanitize_for_logging(extra, max_text_chars=200, secret_values=self._secret_values))
        log_event(self._logger, logging.INFO, "llm.request.succeeded", **payload)
        self._finished = True

    def failed(self, error: BaseException, **extra: Any) -> None:
        if self._finished:
            return
        payload: dict[str, Any] = {
            "call_id": self._call_id,
            "provider": self._provider,
            "api": self._api,
            "model": self._model,
            "metadata": sanitize_for_logging(self._metadata, max_text_chars=200, secret_values=self._secret_values),
            "error_type": type(error).__name__,
            "error_message": scrub_string(
                str(error),
                max_text_chars=self._max_text_chars,
                secret_values=self._secret_values,
            ),
            "traceback": scrub_string(
                format_traceback(error),
                max_text_chars=self._max_text_chars,
                secret_values=self._secret_values,
            ),
        }
        if self._log_failure_payloads and self._request_payload:
            payload["request"] = sanitize_for_logging(
                self._request_payload,
                max_text_chars=self._max_text_chars,
                secret_values=self._secret_values,
            )
        if extra:
            payload.update(
                sanitize_for_logging(
                    extra,
                    max_text_chars=self._max_text_chars,
                    secret_values=self._secret_values,
                )
            )
        log_event(self._logger, logging.ERROR, "llm.request.failed", **payload)
        self._finished = True
