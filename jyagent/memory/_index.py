# Tier 1 — MEMORY.md (the always-loaded index).
#
# Owns: MEMORY.md file CRUD primitives, durable-entry validation,
# protected-section logic (so substring-`forget` and the LLM-driven UPDATE
# pipeline cannot wipe Behavioral Rules / User Profile / User Preferences),
# the keyword-`forget` primitive, and the size-warning soft-cap reporter.
#
# Extracted from ``operations.py`` (2026-05-06) as part of the split that
# made each tier its own module.  Existing callers continue to import via
# ``jyagent.memory.operations`` (which re-exports this module's API).

import os
import re
import threading

from .. import config as _cfg

# Access paths via _cfg.MEMORY_MD_FILE / _cfg.TOPICS_DIR (late-bound) so that
# tests can patch config attributes *after* this module is imported.
MAX_MEMORY_INDEX_LINES = _cfg.MAX_MEMORY_INDEX_LINES
MAX_MEMORY_INDEX_BYTES = _cfg.MAX_MEMORY_INDEX_BYTES
MAX_DURABLE_MEMORY_TEXT_CHARS = 400

_ALLOWED_MEMORY_CATEGORIES = {
    "",
    "correction",
    "preference",
    "gotcha",
    "tip",
    "workflow",
    "user_stated",
    "goal",
}
_MEMORY_CATEGORY_RE = re.compile(r"^[a-z_]+$")


# ─── Concurrency ──────────────────────────────────────────────────────────────
#
# Every read-modify-write on MEMORY.md takes this reentrant lock. The lock is
# reentrant because several call chains nest — e.g. the LLM-driven extraction
# pipeline replaces an existing line by calling ``forget`` and ``remember``
# back-to-back inside one critical section.
#
# Why we need it:
#   - The background extraction thread (extraction.py::_do_extract) and the
#     main thread's synchronous manage_memory calls both mutate MEMORY.md.
#   - Extraction's ADD/UPDATE pipeline does read → think → write; a parallel
#     writer sneaking in between the read and the write silently drops one
#     side's edits.
#   - The UPDATE path is two writes (forget + remember), so without a lock a
#     concurrent reader can see the "old line gone, new line missing"
#     intermediate state and act on it.
#
# Scope: only MEMORY.md mutations — topic and journal files have their own
# semantics (journal uses O_EXCL for the header; topics are overwritten
# wholesale and we accept last-writer-wins for them since the topic name
# itself is the concurrency partition).
_MEMORY_MD_LOCK = threading.RLock()


def ensure_dirs() -> None:
    os.makedirs(os.path.dirname(_cfg.MEMORY_MD_FILE), exist_ok=True)
    os.makedirs(_cfg.TOPICS_DIR, exist_ok=True)
    os.makedirs(_cfg.JOURNAL_DIR, exist_ok=True)

# ─── MEMORY.md operations ─────────────────────────────────────────────────────

def read_memory_md() -> str:
    """Read the MEMORY.md index file. Returns empty string if not found."""
    try:
        with open(_cfg.MEMORY_MD_FILE, 'r', encoding='utf-8') as f:
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
    """Write content to MEMORY.md (acquires the MEMORY.md lock)."""
    ensure_dirs()
    with _MEMORY_MD_LOCK:
        with open(_cfg.MEMORY_MD_FILE, 'w', encoding='utf-8') as f:
            f.write(content)


def append_memory_md(text: str) -> None:
    """Append a line to the end of MEMORY.md.

    If the existing file lacks a trailing newline (common when MEMORY.md was
    hand-edited), prepend one to ``text`` so the new entry doesn't get
    silently glued onto the previous last line — which would corrupt the
    category prefix (`[gotcha] foo[tip] bar` instead of two separate lines).

    Acquires the MEMORY.md lock to serialize against concurrent writers
    (background extraction thread + main-thread tool calls).
    """
    ensure_dirs()
    with _MEMORY_MD_LOCK:
        existing = read_memory_md()
        with open(_cfg.MEMORY_MD_FILE, 'a', encoding='utf-8') as f:
            if not existing:
                f.write(f"# Agent Memory\n\n{text}\n")
            elif not existing.endswith("\n"):
                # Heal the missing terminator before appending.
                f.write(f"\n{text}\n")
            else:
                f.write(f"{text}\n")


def _validate_memory_entry(text: str, category: str = "") -> tuple[str, str]:
    """Validate one durable MEMORY.md entry and return stripped text/category.

    MEMORY.md is injected into the system prompt, so durable entries must stay
    one-line facts/rules rather than arbitrary markdown blocks.
    """
    entry = text.strip()
    cat = category.strip().lower()

    if not entry:
        raise ValueError("memory entry text must be non-empty")
    if len(entry.splitlines()) != 1:
        raise ValueError("memory entry text must be exactly one line")
    if len(entry) > MAX_DURABLE_MEMORY_TEXT_CHARS:
        raise ValueError(
            f"memory entry text must be <= {MAX_DURABLE_MEMORY_TEXT_CHARS} chars"
        )
    if entry.lstrip().startswith(("#", "~~")):
        raise ValueError(
            "memory entry text must not start with a markdown heading or strikethrough"
        )
    if cat and (cat not in _ALLOWED_MEMORY_CATEGORIES or not _MEMORY_CATEGORY_RE.match(cat)):
        allowed = ", ".join(sorted(c for c in _ALLOWED_MEMORY_CATEGORIES if c))
        raise ValueError(f"invalid memory category {category!r}; allowed: {allowed}")

    return entry, cat


# ─── Protected sections (shared with extraction.py UPDATE pipeline) ──────────
#
# Lines under these sections cannot be deleted by substring `forget` and
# cannot be replaced by the LLM-driven UPDATE directive. They encode hard
# agent rules / user identity / preferences that should only ever be edited
# by the human, never by the model from a single conversation turn.
#
# Match key is the heading text lowercased and stripped of leading "#"s.

_PROTECTED_SECTION_HEADERS = frozenset({
    "behavioral rules (critical)",
    "behavioral rules",
    "user preferences",
    "user profile",
})

# Default minimum length for the manual `forget` keyword. Shorter keywords
# match too many lines (e.g. "py" hits every Python rule). Override only
# from internal callers that have already validated uniqueness.
MIN_FORGET_KEYWORD_LEN = 6


def _compute_protected_indices(lines: list[str]) -> set[int]:
    """Return the set of line indices that belong to a protected section.

    A line is protected when:
      - it is itself a markdown ``#`` / ``##`` / ``###`` heading, OR
      - it sits under a section whose heading text (lowercased, ``#``-stripped)
        is in ``_PROTECTED_SECTION_HEADERS``.

    Used by ``forget_from_memory_md`` and ``extraction._replace_line`` so a
    keyword-driven delete can never wipe a Behavioral Rule, the User Profile,
    or section headings.
    """
    protected: set[int] = set()
    inside_protected = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            protected.add(i)
            heading = stripped.lstrip("#").strip().lower()
            inside_protected = heading in _PROTECTED_SECTION_HEADERS
            continue
        if inside_protected:
            protected.add(i)
    return protected


def forget_from_memory_md(
    keyword: str,
    *,
    min_keyword_len: int = MIN_FORGET_KEYWORD_LEN,
    protect_sections: bool = True,
) -> tuple[int, list[str], int]:
    """Remove lines containing ``keyword`` from MEMORY.md.

    Returns ``(removed_count, removed_lines_preview, protected_skipped_count)``
    so the caller can render a useful preview.

    Default safeties (a keyword-substring delete is destructive):
      - ``min_keyword_len`` rejects short keywords (e.g. "py" would match
        every Python-related rule). Pass ``min_keyword_len=0`` only from
        internal call sites that have already proven uniqueness.
      - ``protect_sections=True`` skips ``#`` headings and lines under
        Behavioral Rules / User Profile / User Preferences. There is no
        legitimate use case for substring-deleting a behavioral rule.

    Read-modify-write — guarded by ``_MEMORY_MD_LOCK``.

    Raises ``ValueError`` on empty or too-short keyword.
    """
    if not keyword or not keyword.strip():
        raise ValueError("forget keyword must be non-empty")
    if len(keyword) < min_keyword_len:
        raise ValueError(
            f"forget keyword too short ({len(keyword)} < {min_keyword_len}); "
            f"use a more specific substring to avoid mass deletes"
        )

    with _MEMORY_MD_LOCK:
        content = read_memory_md()
        if not content:
            return (0, [], 0)
        lines = content.split("\n")
        keyword_lower = keyword.lower()

        protected_idx = _compute_protected_indices(lines) if protect_sections else set()

        new_lines: list[str] = []
        removed_preview: list[str] = []
        protected_skipped = 0
        for i, line in enumerate(lines):
            if keyword_lower in line.lower():
                if i in protected_idx:
                    protected_skipped += 1
                    new_lines.append(line)
                    continue
                removed_preview.append(line)
                continue  # drop this line
            new_lines.append(line)

        if removed_preview:
            write_memory_md("\n".join(new_lines))
        return (len(removed_preview), removed_preview, protected_skipped)


# ─── Size-warning helper (soft cap) ──────────────────────────────────────────

def memory_index_size_warning() -> str | None:
    """Return a one-line warning if MEMORY.md is approaching the load cap.

    Soft thresholds (configurable):
      - lines >= MEMORY_INDEX_WARN_LINES (default 150 / hard cap 200)
      - bytes >= MEMORY_INDEX_WARN_BYTES (default 18 KB / hard cap 25 KB)

    Anthropic guidance: "target under 200 lines per CLAUDE.md file. Longer
    files consume more context and reduce adherence." Bloated memory files
    cause the model to ignore actual instructions.

    Returns None if memory is healthy. Returns a printable warning otherwise.
    """
    content = read_memory_md()
    if not content:
        return None
    line_count = len(content.splitlines())
    byte_count = len(content.encode("utf-8"))

    warn_lines = _cfg.MEMORY_INDEX_WARN_LINES
    warn_bytes = _cfg.MEMORY_INDEX_WARN_BYTES
    cap_lines = _cfg.MAX_MEMORY_INDEX_LINES
    cap_bytes = _cfg.MAX_MEMORY_INDEX_BYTES

    over_lines = line_count >= warn_lines
    over_bytes = byte_count >= warn_bytes
    if not (over_lines or over_bytes):
        return None

    bits = []
    if over_lines:
        bits.append(f"{line_count} lines (warn at {warn_lines}, hard cap {cap_lines})")
    if over_bytes:
        bits.append(f"{byte_count} bytes (warn at {warn_bytes}, hard cap {cap_bytes})")
    return (
        "⚠️ MEMORY.md approaching load cap: "
        + "; ".join(bits)
        + ". Move detail into topics/<name>.md (curated) or journal/YYYY-MM.md (chronological)."
    )


# ─── Topic Files Index section (keep MEMORY.md in sync with topic CRUD) ──────
#
# Tier-2 topic files have a one-line "table of contents" inside MEMORY.md so
# the always-loaded prompt can advertise what extended detail is available
# without loading every topic body. Mutating that TOC is a read-modify-write
# on MEMORY.md, so it lives here next to the lock and the file primitives.
#
# These were previously co-located with topic CRUD in ``_topics.py`` and
# reached into ``_index._MEMORY_MD_LOCK`` from outside. Moved here 2026-05-12
# (refactor/memory-cleanup step 1).

TOPIC_INDEX_HEADING = "## Topic Files Index"

_MAX_TOPIC_DESC_CHARS = 120
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_topic_description(raw: str) -> str:
    """Sanitize topic description text before writing into Tier 1 (MEMORY.md).

    Topic bodies are arbitrary markdown, but the topic *index entry* in
    MEMORY.md is always-loaded prompt text. A long or hostile first heading
    would inject unbounded text into the system prompt on every turn.

    Rules:
      - strip whitespace and ASCII control chars
      - strip leading ``#``/``~`` (so a heading body doesn't render as a new
        markdown heading inside the index)
      - collapse internal whitespace to single spaces
      - truncate to ``_MAX_TOPIC_DESC_CHARS`` with an ellipsis
    """
    s = _CONTROL_CHARS_RE.sub("", raw or "")
    s = s.strip()
    s = s.lstrip("#").lstrip("~").strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return "(no description)"
    if len(s) > _MAX_TOPIC_DESC_CHARS:
        s = s[: _MAX_TOPIC_DESC_CHARS - 1].rstrip() + "…"
    return s


def extract_topic_description(body: str) -> str:
    """Extract a short description from topic body for the MEMORY.md index.

    Uses the first ``#`` heading text. Falls back to the first non-empty
    non-frontmatter line. Always passed through ``sanitize_topic_description``
    so the index entry can never inject markdown control chars or unbounded
    text into the always-loaded prompt.
    """
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return sanitize_topic_description(stripped)
        if stripped and not stripped.startswith("---"):
            return sanitize_topic_description(stripped)
    return "(no description)"


def upsert_topic_index_entry(name: str, description: str) -> None:
    """Add or update a topic entry in the ``## Topic Files Index`` section.

    This is a MEMORY.md read-modify-write operation, so the whole sequence is
    guarded by ``_MEMORY_MD_LOCK``. Existing entries are replaced in place so
    the always-loaded topic index does not go stale after topic rewrites.
    """
    entry_marker = f"**{name}.md**"
    new_entry = f"- {entry_marker} — {description}"

    with _MEMORY_MD_LOCK:
        content = read_memory_md()
        if not content:
            write_memory_md(f"# Agent Memory\n\n{TOPIC_INDEX_HEADING}\n{new_entry}\n")
            return

        lines = content.split("\n")
        if TOPIC_INDEX_HEADING in content:
            result: list[str] = []
            in_index = False
            replaced = False
            appended = False

            for line in lines:
                stripped = line.strip()
                if stripped == TOPIC_INDEX_HEADING:
                    in_index = True
                    result.append(line)
                    continue

                if in_index:
                    if line.startswith("- **"):
                        if entry_marker in line:
                            if not replaced:
                                result.append(new_entry)
                                replaced = True
                            continue
                        result.append(line)
                        continue
                    if stripped == "":
                        result.append(line)
                        continue

                    if not replaced and not appended:
                        result.append(new_entry)
                        appended = True
                    in_index = False

                result.append(line)

            if in_index and not replaced and not appended:
                result.append(new_entry)

            write_memory_md("\n".join(result))
            return

        # Section doesn't exist — create it before later repo/project sections
        # when possible, otherwise append at end.
        result = []
        inserted = False
        protected_before_index = {
            "## User Profile",
            "## Behavioral Rules (CRITICAL)",
            "## User Preferences",
            "## Environment",
            TOPIC_INDEX_HEADING,
        }
        for line in lines:
            if (
                not inserted
                and line.strip().startswith("## ")
                and line.strip() not in protected_before_index
            ):
                result.append(TOPIC_INDEX_HEADING)
                result.append(new_entry)
                result.append("")
                inserted = True
            result.append(line)
        if not inserted:
            if result and result[-1] != "":
                result.append("")
            result.append(TOPIC_INDEX_HEADING)
            result.append(new_entry)
        write_memory_md("\n".join(result))


def remove_topic_index_entry(name: str) -> None:
    """Remove a topic entry from the ``## Topic Files Index`` in MEMORY.md."""
    with _MEMORY_MD_LOCK:
        content = read_memory_md()
        entry_marker = f"**{name}.md**"

        if entry_marker not in content:
            return

        lines = content.split("\n")
        new_lines = [line for line in lines if entry_marker not in line]

        # If the section is now empty (only heading + blanks), remove it too
        cleaned = []
        skip_empty_section = False
        for i, line in enumerate(new_lines):
            if line.strip() == TOPIC_INDEX_HEADING:
                # Check if next non-blank line is another ## heading or EOF
                j = i + 1
                while j < len(new_lines) and new_lines[j].strip() == "":
                    j += 1
                if j >= len(new_lines) or new_lines[j].startswith("## "):
                    # Section is empty — skip heading and trailing blanks
                    skip_empty_section = True
                    continue
            if skip_empty_section and line.strip() == "":
                continue
            skip_empty_section = False
            cleaned.append(line)

        write_memory_md("\n".join(cleaned))
