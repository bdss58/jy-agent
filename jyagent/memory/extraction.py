# Proactive memory extraction — auto-extract facts from conversation turns.
#
# After each assistant response, scan the latest user<>assistant exchange for
# user preferences, corrections, stated facts, and environment details worth
# remembering.  Runs in a background thread to avoid blocking the main loop.

import sys
import threading
from typing import Optional

from ..config import CHARS_PER_TOKEN
from .operations import read_memory_md, remember


# Minimum user message length to trigger extraction (skip short commands)
_MIN_USER_MSG_CHARS = 30

# Maximum chars of exchange to send for analysis
_MAX_EXCHANGE_CHARS = 4000

# Cooldown: extract every N user messages (not every turn)
_EXTRACTION_INTERVAL = 4

# Module-level state
_messages_since_extraction = 0
_extraction_lock = threading.Lock()

EXTRACTION_PROMPT = """\
You are a memory extraction system for an AI agent. Analyze the conversation \
exchange below and extract ONLY facts worth remembering long-term.

Extract these types (if present):
- **correction**: User corrected the agent about something
- **preference**: User stated a preference (tools, style, workflow, language)
- **user_stated**: User shared a fact about themselves, their environment, or their project
- **tip**: A technical insight or gotcha discovered during the exchange

Rules:
- Return ONLY new facts not already in the existing memory (shown below).
- Each fact on its own line, prefixed with the type: [correction] ..., [preference] ..., etc.
- Be extremely selective — only extract facts useful in FUTURE sessions.
- If there is NOTHING worth extracting, return exactly: NONE
- Maximum 3 facts per exchange.
- Keep each fact concise (one line, under 120 chars).

EXISTING MEMORY (do not duplicate):
{existing_memory}

---
EXCHANGE TO ANALYZE:

User: {user_message}
Assistant: {assistant_message}
"""


def _extract_text(content) -> str:
    """Pull plain text from a message content field (str or list-of-blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def should_extract(user_message: str) -> bool:
    """Decide whether this turn warrants extraction."""
    global _messages_since_extraction
    with _extraction_lock:
        _messages_since_extraction += 1
        if _messages_since_extraction < _EXTRACTION_INTERVAL:
            return False
        if len(user_message) < _MIN_USER_MSG_CHARS:
            return False
        # Reset counter — we'll extract this turn
        _messages_since_extraction = 0
        return True


def extract_and_remember(runtime_owner, user_message: str, assistant_message: str) -> None:
    """Run extraction in background thread. Non-blocking, best-effort."""
    # Quick pre-filter: skip if messages are too short to contain extractable facts
    if len(user_message) < _MIN_USER_MSG_CHARS:
        return

    def _do_extract():
        try:
            existing = read_memory_md()
            # Truncate to keep prompt small
            if len(existing) > 2000:
                existing = existing[:2000] + "\n..."

            user_text = user_message[:_MAX_EXCHANGE_CHARS]
            asst_text = assistant_message[:_MAX_EXCHANGE_CHARS]

            prompt = EXTRACTION_PROMPT.format(
                existing_memory=existing,
                user_message=user_text,
                assistant_message=asst_text,
            )

            result = runtime_owner.complete_text(prompt, max_output_tokens=256)

            if not result or not result.strip() or result.strip().upper() == "NONE":
                return

            # Parse and remember each extracted fact
            count = 0
            for line in result.strip().splitlines():
                line = line.strip()
                if not line or line.upper() == "NONE":
                    continue
                # Lines should look like: [correction] User prefers X
                if line.startswith("[") and "]" in line:
                    bracket_end = line.index("]")
                    category = line[1:bracket_end].strip()
                    fact = line[bracket_end + 1:].strip().lstrip("- ")
                    if fact and len(fact) > 10:
                        remember(fact, category)
                        count += 1
                elif line.startswith("- "):
                    fact = line[2:].strip()
                    if fact and len(fact) > 10:
                        remember(fact)
                        count += 1
                if count >= 3:
                    break

            if count > 0:
                sys.stderr.write(f"\033[2m  🧠 Auto-extracted {count} memory fact(s)\033[0m\n")
                sys.stderr.flush()

        except Exception:
            pass  # Best-effort, never crash the agent

    thread = threading.Thread(target=_do_extract, daemon=True, name="memory-extraction")
    thread.start()
