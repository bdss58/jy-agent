# Session persistence — save/load conversation state across sessions.
#
# Stores the conversation as JSON so the user can resume where they left off
# with /continue.  Sessions are saved on every graceful exit and can be loaded
# on startup.

import json
import os
from datetime import datetime, timezone
from typing import Optional

from ..config import SESSIONS_DIR, LATEST_SESSION_FILE
from .conversation import ConversationMemory


def ensure_session_dir() -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def save_session(conversation: ConversationMemory, metadata: Optional[dict] = None) -> str:
    """Save conversation to disk. Returns the file path written.

    Saves to two locations:
      1. ``data/sessions/latest.json`` — always overwritten (for /continue)
      2. ``data/sessions/<timestamp>.json`` — archived copy
    """
    ensure_session_dir()

    if not conversation.messages:
        return ""

    now = datetime.now(timezone.utc)
    payload = {
        "version": 1,
        "saved_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message_count": len(conversation.messages),
        "metadata": metadata or {},
        "messages": conversation.messages,
    }

    # Write latest (always)
    _atomic_write(LATEST_SESSION_FILE, payload)

    # Write timestamped archive
    ts_name = now.strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(SESSIONS_DIR, f"{ts_name}.json")
    _atomic_write(archive_path, payload)

    # Prune old archives (keep most recent 20)
    _prune_archives(keep=20)

    return LATEST_SESSION_FILE


def load_session(conversation: ConversationMemory, path: Optional[str] = None) -> dict:
    """Load a saved session into the conversation.

    Args:
        conversation: The ConversationMemory to populate.
        path: File path to load. Defaults to latest.json.

    Returns:
        Metadata dict with saved_at, message_count, loaded (bool).
    """
    path = path or LATEST_SESSION_FILE
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
    path = path or LATEST_SESSION_FILE
    return os.path.isfile(path)


def delete_session(path: Optional[str] = None) -> bool:
    """Delete a saved session file."""
    path = path or LATEST_SESSION_FILE
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
        for f in os.listdir(SESSIONS_DIR):
            if f.endswith('.json') and f != "latest.json":
                full = os.path.join(SESSIONS_DIR, f)
                files.append((os.path.getmtime(full), full))
        files.sort(reverse=True)
        for _, path in files[keep:]:
            try:
                os.remove(path)
            except OSError:
                pass
    except OSError:
        pass
