# Shared utilities for the memory subsystem.

import json
import os
import tempfile
from typing import Any

from ..config import MEMORY_DIR, TOPICS_DIR, CHARS_PER_TOKEN


def ensure_dirs() -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(TOPICS_DIR, exist_ok=True)


def atomic_write(filepath: str, data: Any) -> None:
    """Atomically write JSON data to file."""
    ensure_dirs()
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


def load_json(filepath: str, default=None):
    """Load JSON from file, returning default if not found."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


# ─── Token estimation ─────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Estimate token count from text. ~4 chars per token for mixed en/zh."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: dict) -> int:
    """Estimate token count for a single conversation message."""
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

    return estimate_tokens(str(content)) + 4


def estimate_conversation_tokens(messages: list) -> int:
    """Estimate total tokens for a list of conversation messages."""
    return sum(estimate_message_tokens(msg) for msg in messages)
