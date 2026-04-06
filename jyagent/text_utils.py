from __future__ import annotations

import os
from typing import Any


DEFAULT_MAX_TEXT_CHARS = 4000

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
    max_chars = DEFAULT_MAX_TEXT_CHARS if max_chars is None else max_chars
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}... [truncated {omitted} chars]"


def scrub_string(
    value: str,
    *,
    max_text_chars: int | None = None,
    secret_values: list[str] | None = None,
) -> str:
    text = value
    for secret in secret_values or _secret_values():
        text = text.replace(secret, _SENSITIVE_FIELD_MARKER)
    return truncate_text(text, max_chars=max_text_chars)
