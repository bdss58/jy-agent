# In-memory conversation history.

from typing import Any
from .utils import estimate_conversation_tokens


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
