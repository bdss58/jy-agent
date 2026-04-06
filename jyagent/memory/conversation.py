# In-memory conversation history and token estimation helpers.

from typing import Any

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
