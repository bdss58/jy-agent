# MEMORY.md and topic file operations.
# These are the functions called by the manage_memory tool facade.

import os
from datetime import datetime
from zoneinfo import ZoneInfo

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

_FRONTMATTER_SEP = "---"
ASIA_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def _now_iso() -> str:
    """Return current time as ISO 8601 string."""
    return datetime.now(ASIA_SHANGHAI_TZ).isoformat(timespec="seconds")


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Parse YAML-style frontmatter from a topic file.

    Returns (metadata_dict, body_text).  If no frontmatter, returns ({}, raw).
    """
    if not raw.startswith(_FRONTMATTER_SEP):
        return {}, raw

    parts = raw.split(_FRONTMATTER_SEP, 2)  # ['', yaml_block, body]
    if len(parts) < 3:
        return {}, raw

    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    body = parts[2].lstrip("\n")
    return meta, body


def _build_frontmatter(meta: dict) -> str:
    """Serialize metadata dict into YAML frontmatter block."""
    lines = [_FRONTMATTER_SEP]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines.append(_FRONTMATTER_SEP)
    return "\n".join(lines) + "\n"


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


def read_topic_body(name: str) -> str:
    """Read a topic file, stripping frontmatter. Returns just the body."""
    raw = read_topic(name)
    if not raw:
        return ""
    _, body = _parse_frontmatter(raw)
    return body


def read_topic_meta(name: str) -> dict:
    """Read only the frontmatter metadata of a topic file."""
    raw = read_topic(name)
    if not raw:
        return {}
    meta, _ = _parse_frontmatter(raw)
    return meta


def write_topic(name: str, content: str) -> None:
    """Write content to a topic file with created/updated frontmatter.

    If the file already exists, preserves the original ``created`` timestamp
    and updates ``updated``.  New files get both set to now.

    If the caller passes content that already starts with ``---`` frontmatter,
    we merge timestamps into it rather than double-wrapping.
    """
    ensure_dirs()
    filepath = os.path.join(TOPICS_DIR, f"{name}.md")

    now = _now_iso()

    # Preserve original created timestamp if file already exists
    existing_meta = read_topic_meta(name)
    created = existing_meta.get("created", now)

    # If caller included their own frontmatter, strip it and keep their keys
    caller_meta, body = _parse_frontmatter(content)
    if not caller_meta:
        # No frontmatter from caller — treat entire content as body
        body = content

    # Build final metadata: caller keys + timestamps (timestamps win on conflict)
    final_meta = {**caller_meta, "created": created, "updated": now}

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(_build_frontmatter(final_meta))
        f.write(body)
        if body and not body.endswith("\n"):
            f.write("\n")


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
            meta = read_topic_meta(t)
            size = len(tc)
            lines = len(tc.split("\n"))
            updated = meta.get("updated", "")
            ts_suffix = f", updated {updated}" if updated else ""
            topic_lines.append(f"  📄 {t}.md ({lines} lines, {size} chars{ts_suffix})")
        parts.append(f"📂 TOPIC FILES ({len(topics)} topics):\n" + "\n".join(topic_lines))

    if not parts:
        return "🧠 Memory is empty. I'll start learning about you as we interact!"

    return "🧠 SELF-USE MEMORY\n" + "\n\n".join(parts)
