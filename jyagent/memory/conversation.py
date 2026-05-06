# In-memory conversation history and token estimation helpers.

from typing import Any, Optional
from uuid import uuid4

from ..config import CHARS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    """Estimate token count from text. ~4 chars per token for mixed en/zh."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: dict) -> int:
    """Estimate token count for a single conversation message."""
    if message.get("role") == "tool_result":
        total = 4
        total += estimate_tokens(message.get("tool_name", ""))
        total += estimate_tokens(str(message.get("content", "")))
        return total

    content = message.get("content", "")

    if isinstance(content, str):
        return estimate_tokens(content) + 4

    if isinstance(content, list):
        total = 4
        for block in content:
            if isinstance(block, str):
                total += estimate_tokens(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif block_type in {"tool_use", "tool_call"}:
                    total += estimate_tokens(block.get("name", ""))
                    total += estimate_tokens(str(block.get("arguments", block.get("input", {}))))
                elif block_type == "thinking":
                    total += estimate_tokens(block.get("thinking", ""))
                elif block_type == "tool_result":
                    total += estimate_tokens(str(block.get("content", "")))
                else:
                    total += estimate_tokens(str(block))
            else:
                total += estimate_tokens(str(block))
        return total

    return estimate_tokens(str(content)) + 4


def estimate_conversation_tokens(messages: list) -> int:
    """Estimate total tokens for a list of conversation messages."""
    return sum(estimate_message_tokens(msg) for msg in messages)


def _new_session_id() -> str:
    return str(uuid4())


class ConversationMemory:
    """In-memory conversation history (the live "view" fed to the LLM).

    The durable source of truth is the per-session :class:`EventLog`
    (jyagent/memory/event_log.py).  This object is just the working view:
    compaction may rewrite ``self.messages`` in place, but the underlying
    event log is append-only and preserves pre-compaction context.

    Lifecycle is managed externally (jyagent/agent.py + memory/session.py):
    ``ConversationMemory`` does NOT auto-create or attach an event log.
    See :meth:`attach_event_log`.
    """

    def __init__(self):
        self.session_id = _new_session_id()
        self.messages = []
        # Event-log binding — set by attach_event_log().  Optional.
        self._event_log = None  # type: ignore[assignment]
        # How many of self.messages have already been recorded as
        # "kind":"message" events in self._event_log.  Compaction resets
        # this to len(self.messages) after emitting a "kind":"compaction"
        # event with the synthetic replacement_messages.
        self._recorded_seq: int = 0

    def add_message(self, role: str, content: Any) -> None:
        self.messages.append({"role": role, "content": content})

    def get_history(self) -> list:
        return self.messages.copy()

    def get_recent(self, n: int = 10) -> list:
        return self.messages[-n:]

    def clear(self) -> None:
        # Detach (don't close — caller owns the lifecycle).  A new log will
        # be attached by the next checkpoint or by /continue.
        self.session_id = _new_session_id()
        self.messages = []
        self._event_log = None
        self._recorded_seq = 0

    def estimated_tokens(self) -> int:
        """Estimate total tokens in current conversation."""
        return estimate_conversation_tokens(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    # ─── event-log binding ────────────────────────────────────────────────

    def attach_event_log(self, log, recorded_seq: Optional[int] = None) -> None:
        """Bind an :class:`EventLog` so future checkpoints flush new messages.

        ``recorded_seq`` is how many messages in ``self.messages`` are
        already represented in the log (either as "message" events or
        rolled into a prior "compaction" event's ``replacement_messages``).
        Defaults to ``len(self.messages)`` — i.e. assume the current view
        is fully covered by the log so far (correct for both fresh
        sessions and snapshot-resumed sessions).
        """
        self._event_log = log
        self._recorded_seq = (
            len(self.messages) if recorded_seq is None else recorded_seq
        )

    def detach_event_log(self) -> None:
        self._event_log = None
        self._recorded_seq = 0

    def pending_message_events(self) -> list[dict]:
        """Return event dicts for messages not yet in the log."""
        if self._event_log is None:
            return []
        pending = self.messages[self._recorded_seq:]
        return [{"kind": "message", "message": m} for m in pending]

    def mark_recorded(self, count: Optional[int] = None) -> None:
        """Advance the recorded cursor to len(messages) (or explicit count)."""
        self._recorded_seq = (
            len(self.messages) if count is None else count
        )
