# Append-only per-session event log.
#
# Bug-#2 fix: separate the durable conversation history (this log) from the
# in-memory "view" (ConversationMemory.messages) that the LLM sees.  Compaction
# rewrites the view but only ever APPENDS to this log, so pre-compaction
# context stays recoverable on disk.
#
# Wire shape (one JSONL line per event):
#   {"seq":<int>, "ts":"<iso>", "kind":"message", "message": {...full msg...}}
#   {"seq":<int>, "ts":"<iso>", "kind":"compaction",
#    "drop_count": <int>, "replacement_messages":[...], "summary":"...",
#    "before_tokens":<int>, "after_tokens":<int>}
#
# Why this schema:
#   - "kind":"message" + "message":<full dict> preserves every field a future
#     role/tool-result might add. Don't flatten role to kind.
#   - Compaction MUST embed replacement_messages (the synthetic summary +
#     file-reinjection messages) because those are not reconstructible from
#     the filesystem at replay time.
#   - drop_count is unambiguous; ranges are not.
#
# Crash safety:
#   - Append handle stays open with line buffering.  emit_many() writes all
#     events, flushes, and fsync's ONCE at the boundary.  This batches
#     fsync at turn granularity (cheap) instead of per-event (expensive).

import json
import os
import threading
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

ASIA_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def _now_iso() -> str:
    return datetime.now(ASIA_SHANGHAI_TZ).isoformat(timespec="seconds")


def event_log_path(session_id: str, base_dir: str) -> str:
    """Compute the on-disk path for a session's event log."""
    return os.path.join(base_dir, "events", f"{session_id}.jsonl")


class EventLog:
    """Append-only per-session event log backed by a JSONL file.

    Not thread-safe across processes.  Within a single process the writes are
    serialized by an internal lock so accidental concurrent emits don't
    interleave bytes.
    """

    def __init__(self, session_id: str, path: str):
        self.session_id = session_id
        self.path = path
        self._lock = threading.Lock()
        self._fh = None  # opened lazily on first emit
        # Count existing events on disk so seq numbers continue monotonically
        # if we attach to a pre-existing log (e.g. /continue).
        self._next_seq = self._count_existing_events()

    # ─── lifecycle ────────────────────────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._fh is not None:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Line-buffered append handle.  We still flush+fsync explicitly at
        # checkpoint boundaries; line buffering just avoids partial-line
        # glimpses if someone tails the file.
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                except OSError:
                    pass
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None

    def __len__(self) -> int:
        return self._next_seq

    def _count_existing_events(self) -> int:
        if not os.path.isfile(self.path):
            return 0
        try:
            n = 0
            with open(self.path, "r", encoding="utf-8") as f:
                for _ in f:
                    n += 1
            return n
        except OSError:
            return 0

    # ─── write path ───────────────────────────────────────────────────────

    def emit(self, event: dict) -> int:
        """Append a single event.  Returns the seq of the emitted event."""
        return self.emit_many([event])

    def emit_many(self, events: list[dict]) -> int:
        """Atomically append a batch of events with ONE flush+fsync.

        Returns the seq of the LAST emitted event.  Raises on I/O failure —
        callers must NOT proceed to write the snapshot if this fails (that
        would put snapshot ahead of log, violating the source-of-truth
        invariant).
        """
        if not events:
            return self._next_seq - 1
        with self._lock:
            self._ensure_open()
            assert self._fh is not None
            last_seq = -1
            now = _now_iso()
            for evt in events:
                seq = self._next_seq
                self._next_seq += 1
                # Defensive copy + stamp seq/ts.  Never trust callers to set them.
                envelope: dict[str, Any] = {"seq": seq, "ts": now, **evt}
                # Force seq/ts to our values even if caller pre-set them.
                envelope["seq"] = seq
                envelope["ts"] = now
                self._fh.write(json.dumps(envelope, ensure_ascii=False) + "\n")
                last_seq = seq
            self._fh.flush()
            os.fsync(self._fh.fileno())
            return last_seq

    # ─── read path ────────────────────────────────────────────────────────

    def get_events(self, start: int = 0, end: Optional[int] = None) -> list[dict]:
        """Return events in [start, end) by seq.  end=None means open-ended.

        Reads the file fresh each call (no caching) — this is meant for
        diagnostics and replay, not hot-path access.
        """
        if not os.path.isfile(self.path):
            return []
        out: list[dict] = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seq = evt.get("seq", -1)
                    if seq < start:
                        continue
                    if end is not None and seq >= end:
                        break
                    out.append(evt)
        except OSError:
            return out
        return out


def open_event_log(session_id: str, base_dir: str) -> EventLog:
    """Open (or create-on-first-emit) an event log for a session."""
    return EventLog(session_id, event_log_path(session_id, base_dir))
