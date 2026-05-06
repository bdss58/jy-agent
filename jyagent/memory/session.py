# Session persistence — save/load conversation state across sessions.
#
# Stores the conversation as JSON so the user can resume where they left off
# with /continue.  Sessions are saved on every graceful exit and can be loaded
# on startup.

import json
import os
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from .. import config
from .conversation import ConversationMemory, _new_session_id
from .event_log import EventLog, event_log_path, open_event_log


ASIA_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def ensure_session_dir() -> None:
    os.makedirs(config.SESSIONS_DIR, exist_ok=True)


def _flush_log_pending(conversation: ConversationMemory) -> Optional[int]:
    """Flush any unrecorded messages to the attached event log.

    Returns the seq of the last event in the log after flushing (or
    ``None`` if no log is attached).  Raises on log I/O failure — callers
    must NOT proceed to write the snapshot if this raises (snapshot ahead
    of log violates the source-of-truth invariant).
    """
    log = conversation._event_log
    if log is None:
        return None
    pending = conversation.pending_message_events()
    if pending:
        log.emit_many(pending)
        conversation.mark_recorded()
    return (len(log) - 1) if len(log) > 0 else None


def _build_payload(conversation: ConversationMemory, metadata: Optional[dict] = None) -> dict:
    """Build the JSON-serialisable session payload."""
    now = datetime.now(ASIA_SHANGHAI_TZ)
    last_seq: Optional[int] = None
    log = getattr(conversation, "_event_log", None)
    if log is not None and len(log) > 0:
        last_seq = len(log) - 1
    return {
        "version": 1,
        "session_id": conversation.session_id,
        "saved_at": now.isoformat(timespec="seconds"),
        "message_count": len(conversation.messages),
        "metadata": metadata or {},
        "messages": conversation.messages,
        # Source-of-truth pointer into the event log.  Old loaders that
        # don't know this field will safely ignore it.
        "last_event_seq": last_seq,
    }


def checkpoint_session(conversation: ConversationMemory, metadata: Optional[dict] = None) -> str:
    """Lightweight per-turn durability write. Overwrites ``latest.json`` only.

    Unlike :func:`save_session` this does NOT create a timestamped archive and
    does NOT prune. It is meant to be called after every turn boundary so that
    a crash between user input and graceful exit does not lose the conversation.

    **Order matters**: we flush pending events to the source-of-truth log
    FIRST, then write the snapshot.  If the log flush raises, we propagate
    (caller wraps in try/except) — writing a snapshot ahead of the log would
    violate the "log is truth" invariant.

    Returns the file path written, or ``""`` if the conversation is empty.
    """
    ensure_session_dir()

    if not conversation.messages:
        return ""

    # 1. Lazy-attach a log if none yet — first checkpoint of a fresh session.
    if conversation._event_log is None:
        log = open_event_log(conversation.session_id, config.SESSIONS_DIR)
        # Cursor = current view length: messages already in self.messages
        # at attach time are assumed covered (true for fresh sessions where
        # nothing's been recorded yet AND the messages list is empty here is
        # excluded above; for snapshot-resume the loader attaches with
        # cursor = len(messages) explicitly).
        conversation.attach_event_log(log, recorded_seq=0)

    # 2. Flush log first (source of truth).  If this raises, NO snapshot.
    _flush_log_pending(conversation)

    # 3. Snapshot (the view cache).
    payload = _build_payload(conversation, metadata)
    _atomic_write(config.LATEST_SESSION_FILE, payload)
    return config.LATEST_SESSION_FILE


def save_session(conversation: ConversationMemory, metadata: Optional[dict] = None) -> str:
    """Save conversation to disk. Returns the file path written.

    Saves to two locations:
      1. ``data/sessions/latest.json`` — always overwritten (for /continue)
      2. ``data/sessions/<timestamp>.json`` — archived copy
    """
    ensure_session_dir()

    if not conversation.messages:
        return ""

    # Log first, then snapshot (same invariant as checkpoint_session).
    if conversation._event_log is None:
        conversation.attach_event_log(
            open_event_log(conversation.session_id, config.SESSIONS_DIR),
            recorded_seq=0,
        )
    _flush_log_pending(conversation)

    payload = _build_payload(conversation, metadata)

    # Write latest (always)
    _atomic_write(config.LATEST_SESSION_FILE, payload)

    # Write timestamped archive
    ts_name = payload["saved_at"].replace("-", "").replace(":", "").replace("T", "_").rstrip("Z")
    archive_path = os.path.join(config.SESSIONS_DIR, f"{ts_name}.json")
    _atomic_write(archive_path, payload)

    # Prune old archives (keep most recent 20)
    _prune_archives(keep=20)

    return config.LATEST_SESSION_FILE


def archive_session(conversation: ConversationMemory, metadata: Optional[dict] = None) -> str:
    """Archive conversation to a timestamped file WITHOUT updating latest.json.

    Use this when the user explicitly starts a new conversation (/new) — the old
    session should be recoverable but /continue should NOT resume it.

    Returns the archive file path, or "" if there was nothing to save.
    """
    ensure_session_dir()

    if not conversation.messages:
        return ""

    # Log first, then archive snapshot.
    if conversation._event_log is None:
        conversation.attach_event_log(
            open_event_log(conversation.session_id, config.SESSIONS_DIR),
            recorded_seq=0,
        )
    _flush_log_pending(conversation)

    payload = _build_payload(conversation, metadata)

    ts_name = payload["saved_at"].replace("-", "").replace(":", "").replace("T", "_").rstrip("Z")
    archive_path = os.path.join(config.SESSIONS_DIR, f"{ts_name}.json")
    _atomic_write(archive_path, payload)

    _prune_archives(keep=20)

    return archive_path


def load_session(conversation: ConversationMemory, path: Optional[str] = None) -> dict:
    """Load a saved session into the conversation.

    Args:
        conversation: The ConversationMemory to populate.
        path: File path to load. Defaults to latest.json.

    Returns:
        Metadata dict with saved_at, message_count, session_id, loaded (bool).
    """
    path = path or config.LATEST_SESSION_FILE
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"loaded": False, "error": str(e)}

    version = payload.get("version", 0)
    if version != 1:
        return {"loaded": False, "error": f"Unknown session version: {version}"}

    messages = payload.get("messages", [])
    if not messages:
        return {"loaded": False, "error": "Session file has no messages"}

    session_id = payload.get("session_id") or _new_session_id()

    # Clear and reload (clear() also detaches any prior event log).
    conversation.clear()
    conversation.session_id = session_id
    for msg in messages:
        conversation.messages.append(msg)

    # Attach the per-session event log if one exists on disk.  Cursor =
    # len(messages) because every message in the snapshot is presumed
    # already represented in the log (either as "message" events or rolled
    # into an earlier "compaction" event).
    log_path = event_log_path(session_id, config.SESSIONS_DIR)
    log_status = "absent"
    log_ahead_recovered = False
    if os.path.isfile(log_path):
        log = open_event_log(session_id, config.SESSIONS_DIR)
        snap_seq = payload.get("last_event_seq")

        # Detect log-ahead-of-snapshot: a crash between log flush and
        # snapshot write left the log with events the snapshot doesn't
        # know about.  Recover by replaying the log — those events are
        # the source of truth, and silently keeping the stale snapshot
        # would orphan them forever (next checkpoint would re-mark
        # `last_event_seq` to current log tail, hiding the gap).
        if snap_seq is not None and len(log) > snap_seq + 1:
            recovered = replay_from_events(session_id)
            conversation.messages = list(recovered.messages)
            # replay_from_events opened its own log handle; reuse that.
            conversation.attach_event_log(
                recovered._event_log,
                recorded_seq=len(conversation.messages),
            )
            log_status = "log_ahead_recovered"
            log_ahead_recovered = True
        else:
            conversation.attach_event_log(log, recorded_seq=len(messages))
            log_status = "synced"
    # else: leave log unattached.  First checkpoint will create one
    # transparently for future appends to be durable.

    return {
        "loaded": True,
        "session_id": session_id,
        "saved_at": payload.get("saved_at", "unknown"),
        "message_count": len(conversation.messages),
        "metadata": payload.get("metadata", {}),
        "event_log_status": log_status,
        "event_log_seq": len(conversation._event_log) - 1
            if conversation._event_log is not None and len(conversation._event_log) > 0
            else None,
        "log_ahead_recovered": log_ahead_recovered,
    }


def has_saved_session(path: Optional[str] = None) -> bool:
    """Check if a saved session file exists."""
    path = path or config.LATEST_SESSION_FILE
    return os.path.isfile(path)


def list_sessions(limit: Optional[int] = None) -> list[dict]:
    """List saved sessions sorted by saved_at (newest first).

    Returns a list of dicts with keys:
      ``path``, ``filename``, ``session_id``, ``saved_at``,
      ``message_count``, ``metadata``, ``is_latest``.

    Bad / unreadable session files are silently skipped.
    """
    ensure_session_dir()
    latest_real_path = ""
    try:
        # latest.json may be a regular file (atomic rename); resolve to detect dupes
        if os.path.isfile(config.LATEST_SESSION_FILE):
            latest_real_path = os.path.realpath(config.LATEST_SESSION_FILE)
    except OSError:
        latest_real_path = ""

    out: list[dict] = []
    seen_paths: set[str] = set()
    try:
        names = os.listdir(config.SESSIONS_DIR)
    except OSError:
        return out

    # Always include latest.json first if present
    candidates: list[str] = []
    if os.path.isfile(config.LATEST_SESSION_FILE):
        candidates.append(config.LATEST_SESSION_FILE)
    for name in names:
        if not name.endswith(".json") or name == "latest.json":
            continue
        candidates.append(os.path.join(config.SESSIONS_DIR, name))

    for full in candidates:
        try:
            with open(full, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        # Dedupe latest.json against its archive twin (same content, different name)
        try:
            real = os.path.realpath(full)
        except OSError:
            real = full
        sid = payload.get("session_id") or ""
        saved_at = payload.get("saved_at") or ""
        # If the latest file and an archive both have the same session_id +
        # saved_at, prefer the latest entry (already added) and skip the dup.
        dup_key = (sid, saved_at)
        if dup_key in seen_paths:
            continue
        seen_paths.add(dup_key)
        out.append({
            "path": full,
            "filename": os.path.basename(full),
            "session_id": sid,
            "saved_at": saved_at,
            "message_count": payload.get("message_count", len(payload.get("messages", []))),
            "metadata": payload.get("metadata", {}),
            "is_latest": (real == latest_real_path) or (full == config.LATEST_SESSION_FILE),
        })

    out.sort(key=lambda e: e["saved_at"], reverse=True)
    if limit:
        out = out[:limit]
    return out


def find_session(query: str) -> Optional[dict]:
    """Resolve a user-supplied query to a session entry.

    Accepted query forms (in order of precedence):
      * ``"latest"`` → latest.json (if it exists)
      * exact ``session_id`` match
      * unique session_id prefix match (≥4 chars)
      * exact filename match (with or without .json extension)
      * unique filename / saved_at prefix match (e.g. ``20260430``)

    Returns the session entry dict (same shape as ``list_sessions``) or
    ``None`` if no unambiguous match is found.
    """
    if not query:
        return None
    q = query.strip()
    if not q:
        return None

    if q.lower() == "latest":
        if has_saved_session():
            entries = list_sessions()
            for e in entries:
                if e["is_latest"]:
                    return e
            return entries[0] if entries else None
        return None

    entries = list_sessions()
    if not entries:
        return None

    # Exact session_id
    for e in entries:
        if e["session_id"] == q:
            return e

    # Exact filename (with or without .json)
    q_fn = q if q.endswith(".json") else q + ".json"
    for e in entries:
        if e["filename"] == q_fn or e["filename"] == q:
            return e

    # Prefix matches — require ≥4 chars to avoid accidental hits
    if len(q) >= 4:
        sid_matches = [e for e in entries if e["session_id"].startswith(q)]
        if len(sid_matches) == 1:
            return sid_matches[0]
        # Filename / saved_at prefix (timestamp like "20260430")
        fn_matches = [
            e for e in entries
            if e["filename"].startswith(q) or e["saved_at"].replace("-", "").replace(":", "").startswith(q)
        ]
        if len(fn_matches) == 1:
            return fn_matches[0]

    return None


def delete_session(path: Optional[str] = None) -> bool:
    """Delete a saved session file."""
    path = path or config.LATEST_SESSION_FILE
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False


# ─── Replay from event log ────────────────────────────────────────────────────

def replay_from_events(session_id: str) -> ConversationMemory:
    """Reconstruct a :class:`ConversationMemory` view from the event log alone.

    Walks every event in ``data/sessions/events/<session_id>.jsonl`` and
    rebuilds the live view, applying each compaction event in sequence:
      - ``"kind": "message"`` → append ``event["message"]`` to view.
      - ``"kind": "compaction"`` → drop the first ``drop_count`` view
        messages, replace them with ``event["replacement_messages"]``.

    The result has its event log attached at ``recorded_seq=len(view)``
    so future appends pick up cleanly.

    Raises ``FileNotFoundError`` if no log exists.  Useful for recovery
    after a snapshot loss or a "log ahead of snapshot" diagnosis.
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
            drop = int(evt.get("drop_count", 0))
            replacement = evt.get("replacement_messages") or []
            kept = view[drop:]
            view = list(replacement) + list(kept)
        # Unknown kinds: ignore (forward-compatibility).

    c = ConversationMemory()
    c.session_id = session_id
    c.messages = view
    c.attach_event_log(log, recorded_seq=len(view))
    return c


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _atomic_write(path: str, payload: dict) -> None:
    """Write JSON atomically via temp file + rename, with fsync.

    Symmetry with the event log: both sides flush+fsync at the boundary so
    "snapshot ahead of log" or "log ahead of snapshot" can only happen
    across a process crash between the two writes, never from a buffered-
    but-unflushed write surviving in cache.
    """
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=None)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # tmpfs / non-syncable FS — best-effort
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _prune_archives(keep: int = 20) -> None:
    """Remove old session archives, keeping the most recent ones."""
    try:
        files = []
        for f in os.listdir(config.SESSIONS_DIR):
            if f.endswith('.json') and f != "latest.json":
                full = os.path.join(config.SESSIONS_DIR, f)
                files.append((os.path.getmtime(full), full))
        files.sort(reverse=True)
        for _, path in files[keep:]:
            try:
                os.remove(path)
            except OSError:
                pass
    except OSError:
        pass
