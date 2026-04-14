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
from .conversation import ConversationMemory


ASIA_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def ensure_session_dir() -> None:
    os.makedirs(config.SESSIONS_DIR, exist_ok=True)


def _build_payload(conversation: ConversationMemory, metadata: Optional[dict] = None) -> dict:
    """Build the JSON-serialisable session payload."""
    now = datetime.now(ASIA_SHANGHAI_TZ)
    return {
        "version": 1,
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
        Metadata dict with saved_at, message_count, loaded (bool).
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

    # Clear and reload
    conversation.clear()
    for msg in messages:
        conversation.messages.append(msg)

    return {
        "loaded": True,
        "saved_at": payload.get("saved_at", "unknown"),
        "message_count": len(messages),
        "metadata": payload.get("metadata", {}),
    }


def has_saved_session(path: Optional[str] = None) -> bool:
    """Check if a saved session file exists."""
    path = path or config.LATEST_SESSION_FILE
    return os.path.isfile(path)


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
