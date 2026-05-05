"""LoopCallbacks — observer hooks the agent loop fires during run().

Extracted from the monolithic loop_engine.py during the runtime-package
refactor (phase 2). All fields are Optional[Callable]; None means the
loop runs silently (sub-agent / non-streaming mode).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class LoopCallbacks:
    # All Optional[Callable].  None = silent (sub-agent mode).
    on_text_delta: Callable[[str], None] | None = None
    on_thinking_start: Callable[[], None] | None = None
    on_thinking_stop: Callable[[], None] | None = None
    on_tool_start: Callable[[str, dict], None] | None = None
    # Gate callback fired AFTER ``on_tool_start`` and BEFORE the tool actually
    # runs.  Return ``"deny"`` to skip execution (the engine will synthesize a
    # ``Denied by user`` ToolResult); return ``"allow"`` / ``None`` to proceed.
    # Used by interactive UIs to implement an approval gate (Claude-Code-style
    # ``--ask`` flag).  Sub-agent and silent runs leave this as None.
    on_tool_pre_execute: Callable[[str, dict], str | None] | None = None
    on_tool_end: Callable[..., None] | None = None  # (name, content, is_error, duration_ms | None)
    on_retry: Callable[[int, Exception], None] | None = None  # (attempt, error)
    on_compaction: Callable[[int, int], None] | None = None  # (before_len, after_len)
    on_usage: Callable[[dict], None] | None = None  # raw Usage dict from response
    on_step_progress: Callable[[int, int], None] | None = None  # (step, max_steps)
    on_assistant_message: Callable[[dict], dict] | None = None  # transform before append
    on_warning: Callable[[str], None] | None = None  # runtime warnings
    on_truncation: Callable[[], None] | None = None  # response truncated, retrying
    on_tool_batch: Callable[[int], None] | None = None  # number of tools in batch
    # Fires once before a stream retry (transient error OR truncation recovery).
    # `reason` is "transient_error" or "truncation".  `partial_text` is whatever
    # the current attempt had emitted before the restart — UIs can use this to
    # mark, clear, or strikethrough the duplicated output that will follow.
    on_stream_retry: Callable[[str, str], None] | None = None  # (reason, partial_text)
    # Fires when the engine injects a mid-loop reflection prompt.  `reason`
    # is "every_n" or "after_subagent".  Callback is purely observational —
    # does not affect the loop.
    on_reflection: Callable[[str], None] | None = None
    # Fires when a PhasePolicy assigns a phase to the current step.  `phase`
    # is a short string ("plan" | "act" | "verify" | "finalize" | custom).
    # Observational only.
    on_phase_enter: Callable[[str], None] | None = None
    # Fires after a checkpoint is persisted.  Args: (path, step).  Use
    # for logging / progress display.  Step < 0 indicates a terminal
    # checkpoint (e.g. final.json).
    on_checkpoint: Callable[[str, int], None] | None = None


__all__ = ["LoopCallbacks"]
