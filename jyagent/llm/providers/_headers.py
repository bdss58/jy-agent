"""Provider request header helpers."""

from __future__ import annotations

from typing import Any

from ..types import LLMOptions


STEPCODE_SESSION_HEADER = "x-stepcode-session-id"


def request_headers_from_options(options: LLMOptions) -> dict[str, str]:
    """Build per-request HTTP headers from provider-neutral options."""
    metadata: dict[str, Any] = options.metadata or {}
    session_id = metadata.get("session_id")
    if isinstance(session_id, str):
        session_id = session_id.strip()
        if session_id:
            return {STEPCODE_SESSION_HEADER: session_id}
    return {}
