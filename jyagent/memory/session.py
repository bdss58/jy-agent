# Session persistence — event-log-only.
#
# The per-session append-only event log (``data/sessions/events/<sid>.jsonl``)
# is the ONE source of truth. There are no snapshots, no archives. A tiny
# ``data/sessions/latest.txt`` pointer records which session ``/continue``
# resumes by default.
#
# History: an earlier design wrote ``latest.json`` + timestamped archives
# alongside the log. The dual-writer created a "log first, then snapshot"
# invariant that had to be defended with ``last_event_seq`` bookkeeping and
# a log-ahead-of-snapshot recovery branch. Simplification (2026-05): the log
# is enough; list/resume walk the log directly.
#
# Event kinds (grown from the original message + compaction pair):
#   {"kind": "session_start", "metadata": {...}}    # first event of every new log
#   {"kind": "session_meta",  "metadata": {...}}    # emitted when metadata changes
#   {"kind": "message",       "message": {...}}     # one per view message
#   {"kind": "compaction",    "drop_count": N,
#                             "replacement_messages": [...],
#                             "summary": "...",
#                             "before_tokens": N, "after_tokens": N}
#   {"kind": "session_end",   "reason": "new"}      # emitted when user runs /new
#
# All event records also carry auto-stamped ``seq`` and ``ts`` from event_log.py.
#
# Concurrency: single-CLI assumption. ``EventLog`` is process-local; two
# concurrent jyagent processes writing to the same session will race on seq
# numbers and on latest.txt. Not a supported configuration.

import json
import os
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from .. import config
from .conversation import ConversationMemory, _new_session_id
from .event_log import EventLog, event_log_path, open_event_log


ASIA_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")

# Plain-text pointer: contents are the session_id (no trailing whitespace
# beyond a single newline). Atomic write via temp + rename + fsync.
_LATEST_POINTER_BASENAME = "latest.txt"

# Sentinel for the per-log metadata cache (distinguishes "never queried"
# from "queried, found no metadata").
_META_CACHE_UNSET = object()


# ─── Public API ──────────────────────────────────────────────────────────────

def ensure_session_dir() -> None:
    """Create the sessions dir (and the events/ subdir) if missing."""
    os.makedirs(os.path.join(config.SESSIONS_DIR, "events"), exist_ok=True)


def checkpoint_session(
    conversation: ConversationMemory,
    metadata: Optional[dict] = None,
) -> str:
    """Per-turn durability: flush pending message events to the log.

    Side effects:
      1. Lazy-creates the log on first call (emits a ``session_start`` event
         with ``metadata``).
      2. Emits a ``session_meta`` event if ``metadata`` differs from the
         most recent session_start/session_meta.
      3. Flushes any pending ``message`` events (the log handler does the
         fsync).
      4. Atomically updates ``latest.txt`` to point at this session.

    Returns the session_id on success, or ``""`` if the conversation is empty.
    Raises on log I/O failure — the caller decides whether to swallow.
    """
    ensure_session_dir()

    if not conversation.messages:
        return ""

    log = _ensure_log_attached(conversation, metadata or {})
    _maybe_emit_meta_change(log, metadata or {})
    _flush_pending_messages(conversation)
    _write_latest_pointer(conversation.session_id)
    return conversation.session_id


def end_session(conversation: ConversationMemory, reason: str = "new") -> str:
    """Emit a ``session_end`` event and clear the latest pointer.

    Called by ``/new`` so the just-finished session stays discoverable via
    ``/sessions`` / ``/continue <id>`` but is NOT resumed by bare
    ``/continue``.

    Returns the session_id ended, or ``""`` if there was nothing to end.
    """
    if not conversation.messages:
        return ""

    log = conversation._event_log
    if log is None:
        # No log was ever written (very short session) — nothing to do.
        _clear_latest_pointer()
        return ""

    # Flush any last-minute pending events BEFORE writing the end marker so
    # the tail of the log is the session_end event.
    try:
        _flush_pending_messages(conversation)
        log.emit({"kind": "session_end", "reason": reason})
    except OSError:
        # Best-effort — /new clearing should never fail user-visibly.
        pass
    _clear_latest_pointer()
    return conversation.session_id


def load_session(
    conversation: ConversationMemory,
    query: Optional[str] = None,
) -> dict:
    """Load a session via the event log.

    Args:
        conversation: ConversationMemory to populate.
        query: session_id / prefix / saved_at prefix / "latest" / None.
               None means "use latest.txt" (equivalent to query="latest").

    Returns a metadata dict:
        {
            "loaded": bool,
            "session_id": str,
            "saved_at": iso-str,
            "message_count": int,
            "metadata": dict,
            "event_log_seq": int | None,  # last seq after load
            "error": str,                 # only when loaded is False
        }
    """
    ensure_session_dir()

    target_sid = _resolve_query_to_sid(query)
    if target_sid is None:
        return {"loaded": False, "error": f"No session matched {query!r}"}

    log_path = event_log_path(target_sid, config.SESSIONS_DIR)
    if not os.path.isfile(log_path):
        return {"loaded": False, "error": f"No event log for session {target_sid}"}

    # Replay into a fresh ConversationMemory, then splice into the caller's.
    try:
        replayed = replay_from_events(target_sid)
    except Exception as e:
        return {"loaded": False, "error": f"Replay failed: {e}"}

    if not replayed.messages:
        return {"loaded": False, "error": "Session has no visible messages"}

    conversation.clear()
    conversation.session_id = replayed.session_id
    conversation.messages = list(replayed.messages)
    # Reuse the log handle opened during replay — no double-open.
    conversation.attach_event_log(
        replayed._event_log,
        recorded_seq=len(conversation.messages),
    )

    summary = _summarize_log(target_sid)
    return {
        "loaded": True,
        "session_id": target_sid,
        "saved_at": summary["saved_at"],
        "message_count": len(conversation.messages),
        "metadata": summary["metadata"],
        "event_log_seq": len(conversation._event_log) - 1
            if conversation._event_log is not None and len(conversation._event_log) > 0
            else None,
    }


def has_saved_session() -> bool:
    """True iff latest.txt points at an existing, non-ended log."""
    sid = _read_latest_pointer()
    if not sid:
        return False
    log_path = event_log_path(sid, config.SESSIONS_DIR)
    return os.path.isfile(log_path)


def list_sessions(limit: Optional[int] = None) -> list[dict]:
    """List all sessions (newest first) by walking ``events/``.

    Each entry has:
        path (log file), filename, session_id, saved_at, message_count,
        metadata, is_latest, ended (bool — saw a session_end event).

    Bad / unreadable files are silently skipped.
    """
    ensure_session_dir()

    events_dir = os.path.join(config.SESSIONS_DIR, "events")
    try:
        names = os.listdir(events_dir)
    except OSError:
        return []

    latest_sid = _read_latest_pointer()

    out: list[dict] = []
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        sid = name[:-len(".jsonl")]
        full = os.path.join(events_dir, name)
        try:
            info = _summarize_log(sid)
        except OSError:
            continue
        out.append({
            "path": full,
            "filename": name,
            "session_id": sid,
            "saved_at": info["saved_at"],
            "message_count": info["message_count"],
            "metadata": info["metadata"],
            "is_latest": (sid == latest_sid),
            "ended": info["ended"],
        })

    out.sort(key=lambda e: e["saved_at"], reverse=True)
    if limit:
        out = out[:limit]
    return out


def find_session(query: str) -> Optional[dict]:
    """Resolve a user-supplied query to a session entry.

    Accepted forms (first match wins):
      * ``"latest"`` → the latest.txt pointer target.
      * exact session_id.
      * unique session_id prefix (≥4 chars).
      * unique saved_at prefix (e.g. ``20260430`` — matches normalized
        ``YYYYMMDD_HHMMSS`` of the last event ts).

    Returns the entry dict (same shape as ``list_sessions``) or ``None``.
    """
    if not query:
        return None
    q = query.strip()
    if not q:
        return None

    entries = list_sessions()
    if not entries:
        return None

    if q.lower() == "latest":
        for e in entries:
            if e["is_latest"]:
                return e
        return None  # no latest pointer set

    # Exact session_id
    for e in entries:
        if e["session_id"] == q:
            return e

    # Prefix matches — require ≥4 chars to avoid accidents
    if len(q) >= 4:
        sid_matches = [e for e in entries if e["session_id"].startswith(q)]
        if len(sid_matches) == 1:
            return sid_matches[0]
        # saved_at prefix (normalized "20260430_215012" / "20260430" etc.)
        ts_matches = [
            e for e in entries
            if _normalize_ts(e["saved_at"]).startswith(q)
        ]
        if len(ts_matches) == 1:
            return ts_matches[0]

    return None


def delete_session(query: Optional[str] = None) -> bool:
    """Delete a session.

    - ``query=None``: just clear the latest.txt pointer (do NOT delete any
      log file). This is the backward-compatible behavior used by ``/new``
      in the old design. ``end_session`` is now preferred.
    - Otherwise: resolve ``query`` to a session_id; delete its log file and,
      if it was the latest pointer target, clear the pointer.

    Returns True if something was removed.
    """
    if query is None:
        return _clear_latest_pointer()

    entry = find_session(query)
    if entry is None:
        return False
    try:
        os.remove(entry["path"])
    except FileNotFoundError:
        return False
    if entry["is_latest"]:
        _clear_latest_pointer()
    return True


def replay_from_events(session_id: str) -> ConversationMemory:
    """Reconstruct a ConversationMemory view from the event log alone.

    Walks every event in ``data/sessions/events/<session_id>.jsonl`` and
    rebuilds the live view:
      - ``session_start`` / ``session_meta`` / ``session_end`` → metadata
        only, don't touch the view.
      - ``message`` → append ``event["message"]`` to view.
      - ``compaction`` → drop ``drop_count`` head messages, prepend
        ``replacement_messages``.

    The returned object has its event log attached at
    ``recorded_seq=len(view)`` so future appends pick up cleanly.

    Raises ``FileNotFoundError`` if no log exists.
    """
    path = event_log_path(session_id, config.SESSIONS_DIR)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No event log for session {session_id}: {path}")

    log = open_event_log(session_id, config.SESSIONS_DIR)
    view: list = []
    for evt in log.get_events():
        kind = evt.get("kind")
        if kind == "message":
            msg = evt.get("message")
            if isinstance(msg, dict):
                view.append(msg)
        elif kind == "compaction":
            drop = max(0, int(evt.get("drop_count", 0)))
            replacement = evt.get("replacement_messages") or []
            kept = view[drop:]
            view = list(replacement) + list(kept)
        # session_start / session_meta / session_end / unknown → ignore
        # for view reconstruction.

    c = ConversationMemory()
    c.session_id = session_id
    c.messages = view
    c.attach_event_log(log, recorded_seq=len(view))
    return c


# ─── Log attach / emit helpers ───────────────────────────────────────────────

def _ensure_log_attached(
    conversation: ConversationMemory,
    metadata: dict,
) -> EventLog:
    """Attach an EventLog if not already, emitting session_start on creation."""
    log = conversation._event_log
    if log is not None:
        return log

    log = open_event_log(conversation.session_id, config.SESSIONS_DIR)
    # Ensure seq 0 is a session_start event for brand-new logs.
    if len(log) == 0:
        snapshot = _copy_metadata(metadata)
        log.emit({
            "kind": "session_start",
            "metadata": snapshot,
        })
        # Seed the meta cache so the next checkpoint doesn't re-scan.
        setattr(log, "_session_meta_cache", snapshot)
    conversation.attach_event_log(log, recorded_seq=0)
    return log


def _maybe_emit_meta_change(log: EventLog, metadata: dict) -> None:
    """If metadata differs from the most recent start/meta event, emit session_meta.

    Caches the last known metadata on the log instance to avoid an O(N) scan
    of every event on every checkpoint. The cache is seeded by a one-shot
    reverse scan on first call after attach.
    """
    if not metadata:
        return
    cached = getattr(log, "_session_meta_cache", _META_CACHE_UNSET)
    if cached is _META_CACHE_UNSET:
        cached = _last_metadata_event(log) or {}
    if _metadata_equal(cached, metadata):
        return
    snapshot = _copy_metadata(metadata)
    log.emit({"kind": "session_meta", "metadata": snapshot})
    setattr(log, "_session_meta_cache", snapshot)


def _flush_pending_messages(conversation: ConversationMemory) -> None:
    """Flush any message events the conversation hasn't recorded yet."""
    log = conversation._event_log
    if log is None:
        return
    pending = conversation.pending_message_events()
    if pending:
        log.emit_many(pending)
        conversation.mark_recorded()


def _last_metadata_event(log: EventLog) -> Optional[dict]:
    """Return the metadata dict of the newest session_start/session_meta event."""
    for evt in reversed(log.get_events()):
        if evt.get("kind") in ("session_start", "session_meta"):
            md = evt.get("metadata")
            return md if isinstance(md, dict) else {}
    return None


def _metadata_equal(a: dict, b: dict) -> bool:
    """Shallow equality on the metadata keys we care about."""
    keys = set(a.keys()) | set(b.keys())
    return all(a.get(k) == b.get(k) for k in keys)


def _copy_metadata(md: dict) -> dict:
    """Shallow-copy metadata, coercing non-JSON-safe values to str."""
    out: dict[str, Any] = {}
    for k, v in (md or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


# ─── Summaries (for /list) ───────────────────────────────────────────────────

def _summarize_log(session_id: str) -> dict:
    """Walk a session log once and produce a /list-ready summary.

    Returns:
        {"saved_at": iso-str, "message_count": int,
         "metadata": dict, "ended": bool}

    Scans the file a single time — counts cost one JSON parse per line, which
    for typical sessions (≤few thousand events) is sub-millisecond.
    """
    path = event_log_path(session_id, config.SESSIONS_DIR)
    saved_at = ""
    metadata: dict = {}
    message_count = 0
    ended = False

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = evt.get("kind")
                ts = evt.get("ts")
                if ts:
                    saved_at = ts
                if kind == "message":
                    message_count += 1
                elif kind == "compaction":
                    drop = max(0, int(evt.get("drop_count", 0)))
                    repl = evt.get("replacement_messages") or []
                    message_count = max(0, message_count - drop) + len(repl)
                elif kind in ("session_start", "session_meta"):
                    md = evt.get("metadata")
                    if isinstance(md, dict):
                        metadata = md
                elif kind == "session_end":
                    ended = True
    except OSError:
        pass

    # Fallback saved_at if the log has no ts (shouldn't happen — event_log
    # stamps ts on every emit) — use mtime.
    if not saved_at:
        try:
            mtime = os.path.getmtime(path)
            saved_at = datetime.fromtimestamp(
                mtime, tz=ASIA_SHANGHAI_TZ,
            ).isoformat(timespec="seconds")
        except OSError:
            saved_at = ""

    return {
        "saved_at": saved_at,
        "message_count": message_count,
        "metadata": metadata,
        "ended": ended,
    }


# ─── Query resolution ────────────────────────────────────────────────────────

def _resolve_query_to_sid(query: Optional[str]) -> Optional[str]:
    """Resolve a user query to a concrete session_id on disk.

    None / "" / "latest" → latest.txt target (if it exists).
    Otherwise try find_session.
    """
    if query is None or not query.strip() or query.strip().lower() == "latest":
        sid = _read_latest_pointer()
        if sid:
            log_path = event_log_path(sid, config.SESSIONS_DIR)
            if os.path.isfile(log_path):
                return sid
        # Fallback: newest non-ended session by saved_at (recovery path when
        # latest.txt is missing or stale).
        for entry in list_sessions():
            if not entry["ended"]:
                return entry["session_id"]
        return None

    entry = find_session(query.strip())
    return entry["session_id"] if entry else None


def _normalize_ts(iso_ts: str) -> str:
    """'2026-04-30T21:50:12+08:00' → '20260430_215012' for prefix matching."""
    if not iso_ts:
        return ""
    out = iso_ts
    # Drop timezone suffix (+0800 / -0500 / Z) first.
    for sep in ("+", "Z"):
        if sep in out[1:]:  # skip a leading '-' on negative years (n/a here)
            out = out.split(sep)[0]
    # Drop the '-' / ':' separators and turn 'T' into '_'.
    for ch in ("-", ":"):
        out = out.replace(ch, "")
    out = out.replace("T", "_")
    return out


# ─── latest.txt pointer ──────────────────────────────────────────────────────

def _latest_pointer_path() -> str:
    return os.path.join(config.SESSIONS_DIR, _LATEST_POINTER_BASENAME)


def _read_latest_pointer() -> str:
    """Return the session_id in latest.txt, or '' if missing/invalid."""
    try:
        with open(_latest_pointer_path(), "r", encoding="utf-8") as f:
            sid = f.read().strip()
        # Defensive: sanity-check it looks like a UUID-ish token.
        if sid and all(c.isalnum() or c == "-" for c in sid) and len(sid) <= 64:
            return sid
        return ""
    except OSError:
        return ""


def _write_latest_pointer(session_id: str) -> None:
    """Atomic write of latest.txt — tmp file + fsync + rename."""
    if not session_id:
        return
    ensure_session_dir()
    path = _latest_pointer_path()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(session_id + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _clear_latest_pointer() -> bool:
    """Remove latest.txt. Returns True if something was removed."""
    try:
        os.remove(_latest_pointer_path())
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False
