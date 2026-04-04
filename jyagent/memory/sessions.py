# Session summaries — compressed history of past conversation sessions.

import json
import os
import time
import threading
from datetime import datetime

from ..config import MEMORY_DIR, SESSIONS_FILE, MAX_SESSIONS
from .utils import atomic_write, load_json, ensure_dirs


class SessionSummaries:
    """Summaries of past conversation sessions."""

    def __init__(self):
        self.sessions = load_json(SESSIONS_FILE, [])

    def add_summary(self, summary: str, topics: list = None):
        entry = {
            "summary": summary,
            "topics": topics or [],
            "timestamp": datetime.now().isoformat(),
        }
        self.sessions.append(entry)
        if len(self.sessions) > MAX_SESSIONS:
            self.sessions = self.sessions[-MAX_SESSIONS:]
        self.save()

    def save(self):
        atomic_write(SESSIONS_FILE, self.sessions)

    def get_recent(self, n: int = 5) -> list:
        return self.sessions[-n:]

    def to_prompt_text(self, max_chars: int = 600) -> str:
        recent = self.get_recent(5)
        if not recent:
            return ""
        lines = []
        total = 0
        for session in reversed(recent):
            ts = session.get("timestamp", "unknown")[:10]
            topics = ", ".join(session.get("topics", []))
            line = f"- [{ts}] {session['summary']}"
            if topics:
                line += f" (topics: {topics})"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION LIFECYCLE — Fast exit, deferred summarization
# ═══════════════════════════════════════════════════════════════════════════════

PENDING_SESSION_FILE = os.path.join(MEMORY_DIR, "_pending_session.json")

_session_start_time = None
_runtime_owner_ref = None


def on_session_start(runtime_owner=None) -> None:
    """Called when a new agent session begins. Processes any pending session from last exit."""
    global _session_start_time, _runtime_owner_ref
    _session_start_time = time.time()
    _runtime_owner_ref = runtime_owner
    _process_pending_session_background()


def on_session_end(runtime_owner, conversation_messages: list) -> None:
    """Called when a session ends. Only writes files — NO API calls.

    Saves raw conversation to _pending_session.json for the next session to summarize.
    This makes Ctrl-C exit nearly instant.
    """
    if not conversation_messages or len(conversation_messages) < 4:
        return

    try:
        msg_count = len(conversation_messages)
        duration_min = 0
        if _session_start_time:
            duration_min = int((time.time() - _session_start_time) / 60)

        formatted = []
        for msg in conversation_messages[-20:]:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))[:500]
            formatted.append({"role": role, "content": content})

        pending = {
            "messages": formatted,
            "msg_count": msg_count,
            "duration_min": duration_min,
            "timestamp": datetime.now().isoformat(),
        }

        ensure_dirs()
        with open(PENDING_SESSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(pending, f, ensure_ascii=False)

    except Exception:
        try:
            sessions = SessionSummaries()
            msg_count = len(conversation_messages)
            duration = ""
            if _session_start_time:
                mins = int((time.time() - _session_start_time) / 60)
                duration = f" ({mins} min)" if mins > 0 else ""
            sessions.add_summary(f"Session with {msg_count} messages{duration}")
        except Exception:
            pass


def _process_pending_session_background() -> None:
    """Check for a pending session file and summarize it in a background thread."""
    if not os.path.exists(PENDING_SESSION_FILE):
        return

    try:
        with open(PENDING_SESSION_FILE, 'r', encoding='utf-8') as f:
            pending = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        try:
            os.remove(PENDING_SESSION_FILE)
        except OSError:
            pass
        return

    def _summarize():
        try:
            _generate_session_summary(pending, _runtime_owner_ref)
            # Success — now safe to delete the pending file
            try:
                os.remove(PENDING_SESSION_FILE)
            except OSError:
                pass
        except Exception:
            try:
                sessions = SessionSummaries()
                msg_count = pending.get("msg_count", 0)
                duration_min = pending.get("duration_min", 0)
                duration = f" ({duration_min} min)" if duration_min > 0 else ""
                sessions.add_summary(f"Session with {msg_count} messages{duration}")
                # Fallback succeeded — safe to delete
                try:
                    os.remove(PENDING_SESSION_FILE)
                except OSError:
                    pass
            except Exception:
                pass  # Leave file for next session to retry

    t = threading.Thread(target=_summarize, daemon=True, name="pending-session-summarizer")
    t.start()


def _generate_session_summary(pending: dict, runtime_owner=None) -> None:
    """Generate a session summary using the API. Called from background thread."""
    messages = pending.get("messages", [])
    msg_count = pending.get("msg_count", len(messages))
    duration_min = pending.get("duration_min", 0)

    if not messages:
        return

    conversation_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages
    )

    if runtime_owner is None:
        from ..config import get_active_model_spec
        from ..runtime import RuntimeOwner

        runtime_owner = RuntimeOwner(get_active_model_spec())

    response_text = runtime_owner.complete_text(
        (
            "Summarize this conversation in one short sentence. "
            "Also list 2-5 topic keywords.\n"
            "Return JSON: {\"summary\": \"...\", \"topics\": [...]}\n\n"
            + conversation_text
        ),
        max_output_tokens=256,
    )

    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        response_text = "\n".join(lines)

    result = json.loads(response_text)
    summary_text = result.get("summary", "")
    topics = result.get("topics", [])

    if summary_text:
        sessions = SessionSummaries()
        duration = f" ({duration_min} min)" if duration_min > 0 else ""
        sessions.add_summary(
            f"Session with {msg_count} messages{duration}: {summary_text}",
            topics=topics,
        )
