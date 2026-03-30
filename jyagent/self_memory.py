# Memory module — Claude Code-style: MEMORY.md index + topic files + user profile + session summaries
#
# Layout: data/memory/MEMORY.md (index), topics/*.md (on-demand), user_profile.json, session_summaries.json

import json
import os
import time
import tempfile
from typing import Any, Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED — Directory and atomic write utilities
# ═══════════════════════════════════════════════════════════════════════════════

MEMORY_DIR = os.path.join("data", "memory")
TOPICS_DIR = os.path.join(MEMORY_DIR, "topics")

# Configurable via environment variable
# Token-based threshold (estimated): ~4 chars per token
COMPACT_TOKEN_THRESHOLD = int(os.environ.get("AGENT_COMPACT_TOKEN_THRESHOLD", "80000"))
SUMMARIZE_KEEP_RECENT = int(os.environ.get("AGENT_SUMMARIZE_KEEP_RECENT", "6"))

# Legacy message-count threshold (fallback / secondary trigger)
SUMMARIZE_THRESHOLD = int(os.environ.get("AGENT_SUMMARIZE_THRESHOLD", "20"))

# Self-memory file paths
PROFILE_FILE = os.path.join(MEMORY_DIR, "user_profile.json")
MEMORY_MD_FILE = os.path.join(MEMORY_DIR, "MEMORY.md")
SESSIONS_FILE = os.path.join(MEMORY_DIR, "session_summaries.json")

# Limits
MAX_SESSIONS = 50
MAX_MEMORY_INDEX_LINES = 200  # Claude Code: first 200 lines of MEMORY.md
MAX_MEMORY_INDEX_BYTES = 25 * 1024  # Claude Code: or first 25KB
MAX_MEMORY_PROMPT_CHARS = 5000  # Total budget for memory in system prompt

# Token estimation ratio: ~1 token per 4 chars for mixed en/zh text
CHARS_PER_TOKEN = 4


def _ensure_dirs():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(TOPICS_DIR, exist_ok=True)


def _atomic_write(filepath: str, data: Any):
    """Atomically write JSON data to file."""
    _ensure_dirs()
    dir_for_tmp = os.path.dirname(filepath) or MEMORY_DIR
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_for_tmp, suffix=".tmp")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, filepath)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def _load_json(filepath: str, default=None):
    """Load JSON from file, returning default if not found."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Estimate token count from text. ~4 chars per token for mixed en/zh."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: dict) -> int:
    """Estimate token count for a single conversation message.

    Handles various content formats:
    - string content
    - list of content blocks (text, tool_use, tool_result)
    - nested structures
    """
    content = message.get("content", "")

    if isinstance(content, str):
        # +4 for role/formatting overhead
        return estimate_tokens(content) + 4

    if isinstance(content, list):
        total = 4  # overhead
        for block in content:
            if isinstance(block, str):
                total += estimate_tokens(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif block_type == "tool_use":
                    total += estimate_tokens(block.get("name", ""))
                    total += estimate_tokens(str(block.get("input", {})))
                elif block_type == "tool_result":
                    total += estimate_tokens(str(block.get("content", "")))
                else:
                    total += estimate_tokens(str(block))
            else:
                total += estimate_tokens(str(block))
        return total

    # Fallback
    return estimate_tokens(str(content)) + 4


def estimate_conversation_tokens(messages: list) -> int:
    """Estimate total tokens for a list of conversation messages."""
    return sum(estimate_message_tokens(msg) for msg in messages)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION MEMORY — In-memory chat history
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """In-memory conversation history."""

    def __init__(self):
        self.messages = []

    def add_message(self, role: str, content: Any) -> None:
        self.messages.append({"role": role, "content": content})

    def get_history(self) -> list:
        return self.messages.copy()

    def get_recent(self, n: int = 10) -> list:
        return self.messages[-n:]

    def clear(self) -> None:
        self.messages = []

    def estimated_tokens(self) -> int:
        """Estimate total tokens in current conversation."""
        return estimate_conversation_tokens(self.messages)

    def __len__(self) -> int:
        return len(self.messages)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENT MEMORY — File-backed key-value store (used by evolver)
# ═══════════════════════════════════════════════════════════════════════════════

class PersistentMemory:
    """File-backed key-value store with atomic writes."""

    def __init__(self, store_dir: str = MEMORY_DIR):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def save(self, key: str, data: Any) -> None:
        filepath = os.path.join(self.store_dir, f"{key}.json")
        _atomic_write(filepath, data)

    def load(self, key: str) -> Optional[Any]:
        filepath = os.path.join(self.store_dir, f"{key}.json")
        return _load_json(filepath, default=None)

    def list_keys(self) -> list[str]:
        if not os.path.exists(self.store_dir):
            return []
        keys = []
        for filename in os.listdir(self.store_dir):
            if filename.endswith('.json') and not filename.startswith('_'):
                keys.append(filename[:-5])
        return sorted(keys)

    def delete(self, key: str) -> bool:
        filepath = os.path.join(self.store_dir, f"{key}.json")
        try:
            os.remove(filepath)
            return True
        except FileNotFoundError:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION COMPACTION (inspired by Claude Code's /compact)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Key design improvements over simple summarization:
#   1. Token-based trigger: estimates token count instead of message count
#   2. Structured summary: preserves files modified, key decisions, pending tasks
#   3. Memory re-injection: re-reads MEMORY.md and skills after compaction
#      (like Claude Code's "CLAUDE.md fully survives compaction")
#

# The structured compact prompt — requests specific preservation of important context
COMPACT_PROMPT = """\
You are compacting a conversation to free up context space. \
Analyze the conversation below and produce a STRUCTURED summary that preserves critical information.

Your summary MUST include these sections (skip any section that has no relevant content):

## Task Context
What is the user working on? What's the overall goal?

## Files Modified
List every file that was created, modified, or read during this conversation, with a brief note on what was done:
- `path/to/file.py` — added function X, fixed bug Y

## Key Decisions
Important choices made and WHY (e.g., "Chose approach A over B because..."):
- Decision: ...
- Rationale: ...

## Technical Details
Code patterns, configurations, error messages, or technical facts that would be needed to continue the work:
- ...

## Current State
What was accomplished? What's the current status?

## Pending Tasks
What remains to be done? Any next steps the user mentioned?
- [ ] ...

---
CONVERSATION TO COMPACT:

{conversation}
"""


def compact_conversation(
    conversation: ConversationMemory,
    client,
    keep_recent: int = None,
    custom_instruction: str = "",
) -> dict:
    """Compact (summarize) old messages using a structured prompt.

    This is the core compaction function — called by summarize_if_needed()
    automatically, or can be invoked manually via /compact.

    Args:
        conversation: The conversation memory to compact.
        client: Anthropic API client.
        keep_recent: Number of recent messages to preserve verbatim.
        custom_instruction: Optional user instruction (e.g., "Focus on the API changes").

    Returns:
        dict with keys:
            - "compacted": bool — whether compaction actually happened
            - "before_tokens": int — estimated tokens before
            - "after_tokens": int — estimated tokens after
            - "summary": str — the structured summary (empty if not compacted)
    """
    if keep_recent is None:
        keep_recent = SUMMARIZE_KEEP_RECENT

    before_tokens = conversation.estimated_tokens()

    if len(conversation) < 4:
        return {"compacted": False, "before_tokens": before_tokens,
                "after_tokens": before_tokens, "summary": ""}

    messages_to_compact = conversation.messages[:-keep_recent]
    recent_messages = conversation.messages[-keep_recent:]

    if not messages_to_compact:
        return {"compacted": False, "before_tokens": before_tokens,
                "after_tokens": before_tokens, "summary": ""}

    try:
        # Format messages for the compact prompt
        formatted = _format_messages_for_compact(messages_to_compact)

        # Build the compact prompt
        prompt = COMPACT_PROMPT.format(conversation=formatted)

        # Add custom instruction if provided
        if custom_instruction:
            prompt += f"\n\nADDITIONAL INSTRUCTION: {custom_instruction}"

        response = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            timeout=120,
        )

        summary = ""
        for block in response.content:
            if block.type == "text":
                summary += block.text

        if not summary.strip():
            return {"compacted": False, "before_tokens": before_tokens,
                    "after_tokens": before_tokens, "summary": ""}

        # Replace old messages with the structured summary + recent messages
        # The summary goes as a "user" message so Claude can reference it
        conversation.messages = [
            {
                "role": "user",
                "content": (
                    "[Conversation compacted — structured summary of previous "
                    f"{len(messages_to_compact)} messages below]\n\n{summary}"
                ),
            },
            # Need an assistant acknowledgment to maintain valid message alternation
            {
                "role": "assistant",
                "content": (
                    "Understood. I have the compacted context above. "
                    "I'll continue with full awareness of our previous work."
                ),
            },
        ] + recent_messages

        after_tokens = conversation.estimated_tokens()

        return {
            "compacted": True,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "summary": summary,
        }

    except Exception as e:
        # On failure, fall back to simple truncation — keep recent only
        # but don't lose everything
        return {"compacted": False, "before_tokens": before_tokens,
                "after_tokens": before_tokens, "summary": "",
                "error": str(e)}


def _format_messages_for_compact(messages: list) -> str:
    """Format messages into readable text for the compact prompt.

    Handles various content types: strings, tool_use blocks, tool_result blocks.
    Truncates very long tool results to avoid blowing up the compact prompt itself.
    """
    MAX_TOOL_RESULT_CHARS = 2000  # Truncate long tool outputs in compact input

    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            lines.append(f"**{role}**: {content}")

        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        name = block.get("name", "?")
                        inp = str(block.get("input", {}))
                        if len(inp) > 500:
                            inp = inp[:500] + "..."
                        parts.append(f"[Tool call: {name}({inp})]")
                    elif btype == "tool_result":
                        result = str(block.get("content", ""))
                        if len(result) > MAX_TOOL_RESULT_CHARS:
                            result = result[:MAX_TOOL_RESULT_CHARS] + f"... ({len(result)} chars total, truncated)"
                        parts.append(f"[Tool result: {result}]")
                    else:
                        s = str(block)
                        if len(s) > 500:
                            s = s[:500] + "..."
                        parts.append(s)
                else:
                    parts.append(str(block))
            lines.append(f"**{role}**: {' '.join(parts)}")
        else:
            lines.append(f"**{role}**: {str(content)}")

    return "\n\n".join(lines)


def summarize_if_needed(
    conversation: ConversationMemory,
    client,
    system_prompt_rebuilder=None,
    threshold_tokens: int = None,
    keep_recent: int = None,
) -> Optional[dict]:
    """Auto-compact when conversation exceeds token threshold.

    Triggers compaction when EITHER:
      - Estimated tokens exceed threshold (primary, token-based)
      - Message count exceeds legacy threshold (secondary, as safety net)

    After compaction, calls system_prompt_rebuilder (if provided) to re-inject
    MEMORY.md, skills, and other persistent context — like Claude Code's
    "CLAUDE.md fully survives compaction" behavior.

    Args:
        conversation: The conversation memory.
        client: Anthropic API client.
        system_prompt_rebuilder: Optional callback that returns refreshed system prompt
            context (memory + skills). Called after compaction to signal the agent
            to rebuild its system prompt from disk.
        threshold_tokens: Token threshold for triggering compaction.
        keep_recent: Number of recent messages to keep.

    Returns:
        Compaction result dict if compaction happened, None otherwise.
    """
    if threshold_tokens is None:
        threshold_tokens = COMPACT_TOKEN_THRESHOLD
    if keep_recent is None:
        keep_recent = SUMMARIZE_KEEP_RECENT

    estimated = conversation.estimated_tokens()

    # Primary trigger: token-based
    token_trigger = estimated >= threshold_tokens
    # Secondary trigger: message-count (safety net for extreme cases)
    count_trigger = len(conversation) >= SUMMARIZE_THRESHOLD * 2  # 40+ messages

    if not token_trigger and not count_trigger:
        return None

    import sys
    trigger = "tokens" if token_trigger else "messages"
    sys.stdout.write(
        f"\033[2m  ⚡ Auto-compacting conversation "
        f"({trigger}: ~{estimated} tokens, {len(conversation)} msgs)...\033[0m\n"
    )
    sys.stdout.flush()

    result = compact_conversation(conversation, client, keep_recent)

    if result.get("compacted"):
        sys.stdout.write(
            f"\033[2m  ✅ Compacted: ~{result['before_tokens']} → "
            f"~{result['after_tokens']} tokens "
            f"(saved ~{result['before_tokens'] - result['after_tokens']} tokens)\033[0m\n"
        )
        sys.stdout.flush()

        # Signal that memory/skills should be re-injected
        # This is the "CLAUDE.md fully survives compaction" behavior
        if system_prompt_rebuilder:
            try:
                system_prompt_rebuilder()
            except Exception:
                pass

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# USER PROFILE — Facts about the user that persist indefinitely
# ═══════════════════════════════════════════════════════════════════════════════

class UserProfile:
    """Persistent user profile: name, role, tech stack, preferences, etc."""

    def __init__(self):
        self.data = _load_json(PROFILE_FILE, {
            "name": None,
            "role": None,
            "tech_stack": [],
            "os": None,
            "preferences": {},
            "projects": [],
            "communication_style": None,
            "custom_facts": {},
            "last_updated": None,
        })

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                if key in ("tech_stack", "projects") and isinstance(value, str):
                    existing = self.data.get(key, [])
                    if value not in existing:
                        existing.append(value)
                    self.data[key] = existing
                elif key == "preferences" and isinstance(value, dict):
                    self.data.setdefault("preferences", {}).update(value)
                elif key == "custom_facts" and isinstance(value, dict):
                    self.data.setdefault("custom_facts", {}).update(value)
                else:
                    self.data[key] = value
        self.data["last_updated"] = datetime.now().isoformat()
        self.save()

    def save(self):
        _atomic_write(PROFILE_FILE, self.data)

    def to_prompt_text(self) -> str:
        parts = []
        if self.data.get("name"):
            parts.append(f"User's name: {self.data['name']}")
        if self.data.get("role"):
            parts.append(f"Role: {self.data['role']}")
        if self.data.get("os"):
            parts.append(f"OS: {self.data['os']}")
        if self.data.get("tech_stack"):
            parts.append(f"Tech stack: {', '.join(self.data['tech_stack'])}")
        if self.data.get("projects"):
            parts.append(f"Projects: {', '.join(self.data['projects'])}")
        if self.data.get("preferences"):
            prefs = "; ".join(f"{k}: {v}" for k, v in self.data["preferences"].items())
            parts.append(f"Preferences: {prefs}")
        if self.data.get("communication_style"):
            parts.append(f"Communication style: {self.data['communication_style']}")
        if self.data.get("custom_facts"):
            facts = "; ".join(f"{k}: {v}" for k, v in self.data["custom_facts"].items())
            parts.append(f"Other facts: {facts}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY.md + TOPICS — Claude Code-style memory system
# ═══════════════════════════════════════════════════════════════════════════════

def read_memory_md() -> str:
    """Read the MEMORY.md index file. Returns empty string if not found."""
    try:
        with open(MEMORY_MD_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def read_memory_index() -> str:
    """Read MEMORY.md with Claude Code limits: first 200 lines or 25KB."""
    content = read_memory_md()
    if not content:
        return ""

    # Apply line limit
    lines = content.split("\n")
    if len(lines) > MAX_MEMORY_INDEX_LINES:
        content = "\n".join(lines[:MAX_MEMORY_INDEX_LINES])
        content += f"\n... ({len(lines) - MAX_MEMORY_INDEX_LINES} more lines, use read_file to see full MEMORY.md)"

    # Apply byte limit
    if len(content.encode('utf-8')) > MAX_MEMORY_INDEX_BYTES:
        # Truncate to fit
        while len(content.encode('utf-8')) > MAX_MEMORY_INDEX_BYTES:
            content = content[:len(content) - 200]
        content += "\n... (truncated at 25KB, use read_file to see full MEMORY.md)"

    return content


def write_memory_md(content: str):
    """Write content to MEMORY.md."""
    _ensure_dirs()
    with open(MEMORY_MD_FILE, 'w', encoding='utf-8') as f:
        f.write(content)


def append_memory_md(text: str):
    """Append a line to the end of MEMORY.md."""
    _ensure_dirs()
    existing = read_memory_md()
    if existing and not existing.endswith("\n"):
        existing += "\n"
    with open(MEMORY_MD_FILE, 'a', encoding='utf-8') as f:
        if not existing:
            f.write(f"# Agent Memory\n\n{text}\n")
        else:
            f.write(f"{text}\n")


def forget_from_memory_md(keyword: str) -> int:
    """Remove lines containing keyword from MEMORY.md. Returns count removed."""
    content = read_memory_md()
    if not content:
        return 0
    lines = content.split("\n")
    keyword_lower = keyword.lower()
    new_lines = [l for l in lines if keyword_lower not in l.lower()]
    removed = len(lines) - len(new_lines)
    if removed > 0:
        write_memory_md("\n".join(new_lines))
    return removed


# ─── Topic file operations ────────────────────────────────────────────────────

def list_topics() -> list[str]:
    """List all topic files in the topics directory."""
    _ensure_dirs()
    topics = []
    if os.path.exists(TOPICS_DIR):
        for f in sorted(os.listdir(TOPICS_DIR)):
            if f.endswith('.md'):
                topics.append(f[:-3])  # Remove .md extension
    return topics


def read_topic(name: str) -> str:
    """Read a topic file. Returns empty string if not found."""
    filepath = os.path.join(TOPICS_DIR, f"{name}.md")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def write_topic(name: str, content: str):
    """Write content to a topic file."""
    _ensure_dirs()
    filepath = os.path.join(TOPICS_DIR, f"{name}.md")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def delete_topic(name: str) -> bool:
    """Delete a topic file. Returns True if deleted."""
    filepath = os.path.join(TOPICS_DIR, f"{name}.md")
    try:
        os.remove(filepath)
        return True
    except FileNotFoundError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION SUMMARIES — Compressed history of past sessions
# ═══════════════════════════════════════════════════════════════════════════════

class SessionSummaries:
    """Summaries of past conversation sessions."""

    def __init__(self):
        self.sessions = _load_json(SESSIONS_FILE, [])

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
        _atomic_write(SESSIONS_FILE, self.sessions)

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
#
# Design: Ctrl-C should exit instantly. No API calls on the exit path.
#   - on_session_end: writes raw conversation to _pending_session.json (file I/O only)
#   - on_session_start: picks up pending session, summarizes in background thread
#

PENDING_SESSION_FILE = os.path.join(MEMORY_DIR, "_pending_session.json")

_session_start_time = None


def on_session_start():
    """Called when a new agent session begins. Processes any pending session from last exit."""
    global _session_start_time
    _session_start_time = time.time()
    # Process pending session from previous exit (in background)
    _process_pending_session_background()


def on_session_end(client, conversation_messages: list):
    """Called when a session ends. Only writes files — NO API calls.

    Saves raw conversation to _pending_session.json for the next session to summarize.
    This makes Ctrl-C exit nearly instant.
    """
    if not conversation_messages or len(conversation_messages) < 4:
        return

    try:
        # Compute metadata
        msg_count = len(conversation_messages)
        duration_min = 0
        if _session_start_time:
            duration_min = int((time.time() - _session_start_time) / 60)

        # Save only the last 20 messages (truncated) for summarization
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

        _ensure_dirs()
        # Direct write is fine — this is a temp file, atomicity not critical
        with open(PENDING_SESSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(pending, f, ensure_ascii=False)

    except Exception:
        # Absolute worst case: save a bare-bones summary without API call
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


def _process_pending_session_background():
    """Check for a pending session file and summarize it in a background thread."""
    if not os.path.exists(PENDING_SESSION_FILE):
        return

    try:
        with open(PENDING_SESSION_FILE, 'r', encoding='utf-8') as f:
            pending = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # Corrupt or missing — just remove it
        try:
            os.remove(PENDING_SESSION_FILE)
        except OSError:
            pass
        return

    # Remove the pending file immediately so we don't process it again
    try:
        os.remove(PENDING_SESSION_FILE)
    except OSError:
        pass

    import threading

    def _summarize():
        try:
            _generate_session_summary(pending)
        except Exception:
            # Fallback: save bare metadata
            try:
                sessions = SessionSummaries()
                msg_count = pending.get("msg_count", 0)
                duration_min = pending.get("duration_min", 0)
                duration = f" ({duration_min} min)" if duration_min > 0 else ""
                sessions.add_summary(f"Session with {msg_count} messages{duration}")
            except Exception:
                pass

    t = threading.Thread(target=_summarize, daemon=True, name="pending-session-summarizer")
    t.start()


def _generate_session_summary(pending: dict):
    """Generate a session summary using the API. Called from background thread."""
    import anthropic

    messages = pending.get("messages", [])
    msg_count = pending.get("msg_count", len(messages))
    duration_min = pending.get("duration_min", 0)
    timestamp = pending.get("timestamp", "")

    if not messages:
        return

    conversation_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                "Summarize this conversation in one short sentence. "
                "Also list 2-5 topic keywords.\n"
                "Return JSON: {\"summary\": \"...\", \"topics\": [...]}\n\n"
                + conversation_text
            )
        }]
    )

    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text += block.text

    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
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



# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT INJECTION — Build memory context for the system prompt
# ═══════════════════════════════════════════════════════════════════════════════

def build_memory_context(query: str = "") -> str:
    """Build a memory context string to inject into the system prompt.

    Layout:
      1. User Profile (always, structured)
      2. MEMORY.md index (always, first 200 lines / 25KB)
      3. Recent Session Summaries (always, last 5)

    Topic files are NOT injected here — agent reads them on-demand via read_file.
    """
    sections = []

    # 1. User Profile
    profile = UserProfile()
    profile_text = profile.to_prompt_text()
    if profile_text:
        sections.append(f"## User Profile\n{profile_text}")

    # 2. MEMORY.md index (with Claude Code limits)
    memory_index = read_memory_index()
    if memory_index:
        sections.append(f"## Agent Memory (MEMORY.md)\n{memory_index}")

    # 3. Session Summaries
    sessions = SessionSummaries()
    session_text = sessions.to_prompt_text()
    if session_text:
        sections.append(f"## Recent Sessions\n{session_text}")

    if not sections:
        return ""

    full_text = "\n\n".join(sections)
    if len(full_text) > MAX_MEMORY_PROMPT_CHARS:
        full_text = full_text[:MAX_MEMORY_PROMPT_CHARS] + "\n... (memory truncated)"

    # Build topic file listing for agent awareness
    topics = list_topics()
    topic_listing = ""
    if topics:
        topic_listing = "\n\nTopic files available (read with `read_file`): " + \
            ", ".join(f"data/memory/topics/{t}.md" for t in topics)

    return f"""
═══ SELF-USE MEMORY (automatically maintained) ═══
{full_text}{topic_listing}
═══ END MEMORY ═══

Memory instructions:
- MEMORY.md is the index. Keep it concise (under 200 lines). 
- Move detailed knowledge to topic files in data/memory/topics/<name>.md
- Read topic files on-demand with read_file when you need details.
- To remember something: use manage_memory tool, or directly write files.
- To reorganize: rewrite MEMORY.md and topic files with write_file.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MANUAL MEMORY MANAGEMENT — For manage_memory tool
# ═══════════════════════════════════════════════════════════════════════════════

def remember(text: str, category: str = "") -> str:
    """Add something to MEMORY.md."""
    prefix = f"- [{category}] " if category else "- "
    append_memory_md(prefix + text)
    return f"Remembered: {text}"


def forget(keyword: str) -> str:
    """Remove lines matching a keyword from MEMORY.md."""
    removed = forget_from_memory_md(keyword)
    if removed > 0:
        return f"Forgot {removed} line(s) matching '{keyword}'"
    return f"No memories found matching '{keyword}'"


def show_memory() -> str:
    """Display all stored memories in a readable format."""
    parts = []

    profile = UserProfile()
    profile_text = profile.to_prompt_text()
    if profile_text:
        parts.append(f"📋 USER PROFILE:\n{profile_text}")

    memory_md = read_memory_md()
    if memory_md:
        line_count = len(memory_md.split("\n"))
        display = memory_md[:2000]
        if len(memory_md) > 2000:
            display += f"\n... ({len(memory_md)} total chars, {line_count} lines)"
        parts.append(f"🧠 MEMORY.md ({line_count} lines):\n{display}")

    topics = list_topics()
    if topics:
        topic_lines = []
        for t in topics:
            content = read_topic(t)
            size = len(content)
            lines = len(content.split("\n"))
            topic_lines.append(f"  📄 {t}.md ({lines} lines, {size} chars)")
        parts.append(f"📂 TOPIC FILES ({len(topics)} topics):\n" + "\n".join(topic_lines))

    sessions = SessionSummaries()
    if sessions.sessions:
        sess_lines = []
        for s in sessions.sessions[-5:]:
            ts = s.get("timestamp", "?")[:10]
            sess_lines.append(f"  [{ts}] {s['summary']}")
        parts.append(f"📅 RECENT SESSIONS ({len(sessions.sessions)} total):\n" + "\n".join(sess_lines))

    if not parts:
        return "🧠 Memory is empty. I'll start learning about you as we interact!"

    return "🧠 SELF-USE MEMORY\n" + "\n\n".join(parts)
