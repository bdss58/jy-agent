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


ASIA_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def ensure_session_dir() -> None:
    os.makedirs(config.SESSIONS_DIR, exist_ok=True)


def _build_payload(conversation: ConversationMemory, metadata: Optional[dict] = None) -> dict:
    """Build the JSON-serialisable session payload."""
    now = datetime.now(ASIA_SHANGHAI_TZ)
    return {
        "version": 1,
        "session_id": conversation.session_id,
        "saved_at": now.isoformat(timespec="seconds"),
        "message_count": len(conversation.messages),
        "metadata": metadata or {},
        "messages": conversation.messages,
    }


def save_session(conversation: ConversationMemory, metadata: Optional[dict] = None) -> str:
    """Save conversation to disk. Returns the file path written.

    Saves to two locations:
      1. ``data/sessions/latest.json`` — always overwritten (for /continue)
      2. ``data/sessions/<timestamp>.json`` — archived copy
    """
    ensure_session_dir()

    if not conversation.messages:
        return ""

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

    # Clear and reload
    conversation.clear()
    conversation.session_id = session_id
    for msg in messages:
        conversation.messages.append(msg)

    return {
        "loaded": True,
        "session_id": session_id,
        "saved_at": payload.get("saved_at", "unknown"),
        "message_count": len(messages),
        "metadata": payload.get("metadata", {}),
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


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _atomic_write(path: str, payload: dict) -> None:
    """Write JSON atomically via temp file + rename."""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=None)
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
