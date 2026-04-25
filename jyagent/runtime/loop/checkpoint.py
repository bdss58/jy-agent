# jyagent/checkpoint.py — Checkpointed replay for the agent loop.
#
# Serialises enough state per-step that a crashed or cancelled run can be
# resumed without re-executing tool calls from the beginning.  Inspired by
# LangGraph's persistence layer and swe-agent's trajectory recording: the
# core insight is that the message list and a few counters are a sufficient
# statistic for the loop at a step boundary.
#
# Policy (all opt-in, off by default):
#   * AgentLoop writes ``step_<N>.json`` every ``checkpoint_every_n_steps``
#     steps to ``checkpoint_dir/<run_id>/``.
#   * Terminal exits (completed / max_steps / error / interrupted /
#     dedup_break / cost_limit) always write ``final.json`` when enabled.
#   * ``LoopCheckpoint.load(path)`` restores a checkpoint, returning a
#     dict the caller threads back into a fresh ``AgentLoop.run(...)``.
#
# v1 design choice: resume means "re-run from the checkpointed messages".
# The inner counters (tool_calls_count, stuck_state) reset at run()
# entry.  This is "recovery" semantics, which is sufficient for crash
# resilience and debugging.  True mid-rollout continuation (preserving
# the stuck detector and every counter) is a follow-up.

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class LoopCheckpoint:
    """A single checkpoint of loop state at a step boundary."""

    run_id: str
    step: int                           # 0-based step completed
    saved_at: str                       # ISO-8601 UTC
    messages: list = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_calls_count: int = 0
    todos: list = field(default_factory=list)
    # Optional metadata so the checkpoint is self-describing when found
    # loose on disk.
    provider: Optional[str] = None
    model: Optional[str] = None
    status: Optional[str] = None        # "in_progress" | "completed" | "error" | ...
    error: Optional[str] = None

    # ── Serialization ────────────────────────────────────────────────────

    def to_json(self) -> str:
        """JSON-encode.  Non-serializable objects in `messages` are handled
        via `default=str` as a last-resort fallback."""
        return json.dumps(asdict(self), ensure_ascii=False, default=str, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "LoopCheckpoint":
        obj = json.loads(text)
        return cls(**obj)

    # ── File I/O ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Write to *path* with crash-durable atomic semantics.

        Sequence (matters for the durability claim):
          1. Write content to a temp file in the same directory.
          2. ``flush()`` + ``fsync(fd)`` on the temp file — guarantees the
             bytes are on the storage medium, not just in the kernel page
             cache.
          3. ``os.replace(tmp, path)`` — atomic on POSIX.
          4. ``fsync()`` on the parent directory — guarantees the rename
             itself is durable (without this, the file may revert to its
             pre-rename name after a crash on ext4/xfs).

        Without steps 2 and 4 the rename in step 3 only gives "atomic
        visibility within the running kernel", not crash durability —
        the prior implementation would happily lose the checkpoint on
        power-loss. The parent-dir fsync is silently a no-op on
        platforms where ``os.open(dir)`` is not permitted (notably
        Windows); we suppress that one error and keep the temp/file
        fsync which is what matters most.
        """
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
            fh.flush()
            os.fsync(fh.fileno())
        # Atomic rename so a partial write never shadows a good checkpoint.
        os.replace(tmp, path)
        # Durability of the rename itself — best effort; not all platforms
        # allow opening a directory as a file descriptor.
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    @classmethod
    def load(cls, path: str) -> "LoopCheckpoint":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())


# ── Helpers for the loop engine ──────────────────────────────────────────────


def new_run_id() -> str:
    """Generate a fresh run id.  UUID4 without dashes — compact, file-safe."""
    return uuid.uuid4().hex


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_path(dir_: str, run_id: str, step: int | str) -> str:
    """Canonical path for a given run / step.  ``step`` may be an int or
    the literal string ``"final"`` for the terminal checkpoint."""
    safe_run = run_id.replace(os.sep, "_")
    filename = f"step_{step:04d}.json" if isinstance(step, int) else f"{step}.json"
    return os.path.join(dir_, safe_run, filename)


__all__ = [
    "LoopCheckpoint",
    "checkpoint_path",
    "iso_utc_now",
    "new_run_id",
]
