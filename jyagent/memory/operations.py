# MEMORY.md and topic file operations.
# These are the functions called by the manage_memory tool facade.

import os

from ..config import (
    MEMORY_MD_FILE, TOPICS_DIR,
    MAX_MEMORY_INDEX_LINES, MAX_MEMORY_INDEX_BYTES,
)


def ensure_dirs() -> None:
    os.makedirs(os.path.dirname(MEMORY_MD_FILE), exist_ok=True)
    os.makedirs(TOPICS_DIR, exist_ok=True)


# ─── MEMORY.md operations ─────────────────────────────────────────────────────

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

    lines = content.split("\n")
    if len(lines) > MAX_MEMORY_INDEX_LINES:
        content = "\n".join(lines[:MAX_MEMORY_INDEX_LINES])
        content += f"\n... ({len(lines) - MAX_MEMORY_INDEX_LINES} more lines, use read_file to see full MEMORY.md)"

    if len(content.encode('utf-8')) > MAX_MEMORY_INDEX_BYTES:
        while len(content.encode('utf-8')) > MAX_MEMORY_INDEX_BYTES:
            content = content[:len(content) - 200]
        content += "\n... (truncated at 25KB, use read_file to see full MEMORY.md)"

    return content


def write_memory_md(content: str) -> None:
    """Write content to MEMORY.md."""
    ensure_dirs()
    with open(MEMORY_MD_FILE, 'w', encoding='utf-8') as f:
        f.write(content)


def append_memory_md(text: str) -> None:
    """Append a line to the end of MEMORY.md."""
    ensure_dirs()
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
    ensure_dirs()
    topics = []
    if os.path.exists(TOPICS_DIR):
        for f in sorted(os.listdir(TOPICS_DIR)):
            if f.endswith('.md'):
                topics.append(f[:-3])
    return topics


def read_topic(name: str) -> str:
    """Read a topic file. Returns empty string if not found."""
    filepath = os.path.join(TOPICS_DIR, f"{name}.md")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def write_topic(name: str, content: str) -> None:
    """Write content to a topic file."""
    ensure_dirs()
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


# ─── High-level operations (used by manage_memory tool) ──────────────────────

def remember(text: str, category: str = "") -> str:
    """Remember a fact or learning by appending to MEMORY.md."""
    prefix = f"[{category}] " if category else "- "
    append_memory_md(f"{prefix}{text}")
    return f"Remembered: {text[:100]}"


def forget(keyword: str) -> str:
    """Forget memories matching a keyword."""
    removed = forget_from_memory_md(keyword)
    if removed > 0:
        return f"Removed {removed} entries matching '{keyword}'"
    return f"No entries found matching '{keyword}'"


def show_memory() -> str:
    """Show all memory contents."""
    parts = []

    content = read_memory_md()
    if content:
        line_count = len(content.split("\n"))
        display = content[:2000]
        if len(content) > 2000:
            display += f"\n... ({line_count} total lines)"
        parts.append(f"🧠 MEMORY.md ({line_count} lines):\n{display}")

    topics = list_topics()
    if topics:
        topic_lines = []
        for t in topics:
            tc = read_topic(t)
            size = len(tc)
            lines = len(tc.split("\n"))
            topic_lines.append(f"  📄 {t}.md ({lines} lines, {size} chars)")
        parts.append(f"📂 TOPIC FILES ({len(topics)} topics):\n" + "\n".join(topic_lines))

    if not parts:
        return "🧠 Memory is empty. I'll start learning about you as we interact!"

    return "🧠 SELF-USE MEMORY\n" + "\n\n".join(parts)
