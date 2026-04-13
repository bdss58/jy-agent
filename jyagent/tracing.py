"""Lightweight JSONL trace logger for the jy-agent loop engine.

Each run() call produces a RunTrace that accumulates SpanEvents (LLM calls,
tool calls, compaction, cost checks, etc.) and can be flushed to a single
.jsonl file under data/traces/.

Controlled by env var AGENT_TRACE_ENABLED.  When disabled (the default),
every RunTrace method is a no-op with zero overhead.

Usage:
    from jyagent.tracing import get_tracer, RunTrace

    trace = get_tracer()            # None when disabled
    if trace:
        trace.start("anthropic", "claude-sonnet-4-20250514")
        with trace.span(step=1, event_type="llm_call") as s:
            ...  # do work
            s.tokens_in = 500
            s.tokens_out = 120
        trace.finish(status="ok", total_steps=5, total_cost_usd=0.012)
        trace.flush()
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRACE_ENABLED: bool = os.environ.get("AGENT_TRACE_ENABLED", "").lower() in (
    "1",
    "true",
    "yes",
)

TRACES_DIR = Path(__file__).resolve().parent.parent / "data" / "traces"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SpanEvent:
    """A single trace span — one LLM call, tool invocation, or internal event."""

    step: int = 0
    event_type: str = ""  # llm_call | tool_call | compaction | cost_check | dedup_break | verification
    tool_name: Optional[str] = None
    tool_args_summary: Optional[str] = None  # first 200 chars of JSON args
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None

    # -- internal bookkeeping (excluded from serialization) --
    _start_ns: int = field(default=0, repr=False, compare=False)

    def _begin(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def _end(self) -> None:
        if self._start_ns:
            self.duration_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000

    def to_dict(self) -> dict:
        """Serialize, dropping internal fields and None values."""
        d = asdict(self)
        d.pop("_start_ns", None)
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# RunTrace
# ---------------------------------------------------------------------------


class RunTrace:
    """Accumulates trace events for a single agent run.

    When tracing is disabled, callers should hold ``None`` instead of this
    class (see ``get_tracer()``).  All public methods are safe to call; they
    do real work only when the instance exists.
    """

    def __init__(self) -> None:
        self.trace_id: str = uuid.uuid4().hex
        self.start_time: str = ""
        self.end_time: str = ""
        self.provider: str = ""
        self.model: str = ""
        self.status: str = ""
        self.total_steps: int = 0
        self.total_cost_usd: float = 0.0
        self.spans: list[SpanEvent] = []

    # -- lifecycle -----------------------------------------------------------

    def start(self, provider: str, model: str) -> None:
        """Mark the beginning of a run."""
        self.provider = provider
        self.model = model
        self.start_time = _now_iso()

    def finish(
        self,
        status: str = "ok",
        total_steps: int = 0,
        total_cost_usd: float = 0.0,
    ) -> None:
        """Mark the end of a run."""
        self.end_time = _now_iso()
        self.status = status
        self.total_steps = total_steps
        self.total_cost_usd = total_cost_usd

    # -- span recording ------------------------------------------------------

    def span(
        self,
        step: int,
        event_type: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
    ) -> _SpanContext:
        """Return a context manager that times a span and appends it.

        Example::

            with trace.span(step=1, event_type="tool_call", tool_name="run_shell") as s:
                result = run_shell(cmd)
                s.tokens_in = ...
        """
        ev = SpanEvent(step=step, event_type=event_type, tool_name=tool_name)
        if tool_args is not None:
            try:
                raw = json.dumps(tool_args, ensure_ascii=False)
            except (TypeError, ValueError):
                raw = str(tool_args)
            ev.tool_args_summary = raw[:200]
        return _SpanContext(ev, self.spans)

    def add_span(
        self,
        step: int,
        event_type: str,
        duration_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
    ) -> None:
        """Imperatively add a completed span (no context manager needed)."""
        ev = SpanEvent(
            step=step,
            event_type=event_type,
            duration_ms=duration_ms,
            success=success,
            error=error,
            tool_name=tool_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        if tool_args is not None:
            try:
                raw = json.dumps(tool_args, ensure_ascii=False)
            except (TypeError, ValueError):
                raw = str(tool_args)
            ev.tool_args_summary = raw[:200]
        self.spans.append(ev)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "total_steps": self.total_steps,
            "total_cost_usd": self.total_cost_usd,
            "spans": [s.to_dict() for s in self.spans],
        }

    def flush(self) -> Optional[Path]:
        """Write the trace to a JSONL file and return its path.

        Returns ``None`` if the trace has no start_time (i.e. ``start()``
        was never called).
        """
        if not self.start_time:
            return None

        TRACES_DIR.mkdir(parents=True, exist_ok=True)

        # Build filename: YYYY-MM-DD_HHMMSS_{trace_id[:8]}.jsonl
        try:
            dt = datetime.fromisoformat(self.start_time)
        except (ValueError, TypeError):
            dt = datetime.now(timezone.utc)
        stamp = dt.strftime("%Y-%m-%d_%H%M%S")
        short_id = self.trace_id[:8]
        path = TRACES_DIR / f"{stamp}_{short_id}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
            f.write("\n")

        return path


# ---------------------------------------------------------------------------
# Span context manager
# ---------------------------------------------------------------------------


class _SpanContext:
    """Thin context manager returned by ``RunTrace.span()``."""

    __slots__ = ("event", "_target")

    def __init__(self, event: SpanEvent, target: list[SpanEvent]) -> None:
        self.event = event
        self._target = target

    def __enter__(self) -> SpanEvent:
        self.event._begin()
        return self.event

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.event._end()
        if exc_type is not None:
            self.event.success = False
            self.event.error = f"{exc_type.__name__}: {exc_val}"
        self._target.append(self.event)
        return None  # do not suppress exceptions


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_tracer() -> Optional[RunTrace]:
    """Return a fresh ``RunTrace`` if tracing is enabled, else ``None``.

    Callers should guard with ``if trace:`` before calling methods.
    This keeps the hot path at zero cost when tracing is off.
    """
    if not TRACE_ENABLED:
        return None
    return RunTrace()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
