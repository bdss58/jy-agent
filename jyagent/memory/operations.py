# MEMORY.md and topic file operations.
# These are the functions called by the manage_memory tool facade.

import os
import re
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

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
_JOURNAL_LOCK = threading.Lock()


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
    if os.path.exists(_cfg.TOPICS_DIR):
        for f in sorted(os.listdir(_cfg.TOPICS_DIR)):
            if f.endswith('.md'):
                topics.append(f[:-3])
    return topics


# Strict topic-name allowlist. Closes the path-traversal lever — without this,
# a topic name like
# "../../../tmp/escape" would resolve outside TOPICS_DIR for read/write/delete.
# Kept narrow on purpose: even `:` would let a future refactor confuse a
# topic name with a frontmatter key.
_VALID_TOPIC_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,79}$")


def _topic_path(name: str) -> str | None:
    """Return the on-disk path for a topic name, or ``None`` if invalid.

    Validation:
      - Must match ``_VALID_TOPIC_NAME_RE`` (alnum + ``_.-``, ≤80 chars).
      - Must not contain a path separator after normalization (defense in
        depth — the regex already excludes ``/`` and ``\\``).
      - The resolved absolute path must be a direct child of TOPICS_DIR.
    """
    if not name or not _VALID_TOPIC_NAME_RE.match(name):
        return None
    if os.sep in name or (os.altsep and os.altsep in name):
        return None

    candidate = os.path.join(_cfg.TOPICS_DIR, f"{name}.md")
    # Resolve symlinks / `..` defensively even though the regex blocks `..`.
    real_dir = os.path.realpath(_cfg.TOPICS_DIR)
    real_candidate = os.path.realpath(candidate)
    if os.path.dirname(real_candidate) != real_dir:
        return None
    return candidate


def read_topic(name: str) -> str:
    """Read a topic file. Returns empty string if not found or name invalid."""
    filepath = _topic_path(name)
    if filepath is None:
        return ""
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


# Match an ATX header line ("## Foo Bar") at column 0. Used by
# read_topic_section to locate a single section without dragging in the
# `re` import at call time.
_SECTION_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def read_topic_section(name: str, header: str) -> str:
    """Return one section of a topic file, identified by its header text.

    Section boundaries follow markdown convention: a section runs from its
    own header line up to (but excluding) the next header at the same depth
    or shallower. Comparison is case-insensitive and ignores leading ``#``
    characters in the supplied ``header`` argument so callers can pass either
    ``"Tier model"`` or ``"## Tier model"``.

    Returns the empty string if the topic or the section is not found.
    Sub-sections (deeper headers nested inside the matched one) are included.
    """
    body = read_topic_body(name)
    if not body:
        return ""

    # Normalize the requested header — accept "## Foo" / "Foo" / "  Foo  ".
    needle = header.lstrip("#").strip().lower()
    if not needle:
        return ""

    matches = list(_SECTION_HEADER_RE.finditer(body))
    if not matches:
        return ""

    target = None
    for m in matches:
        if m.group(2).strip().lower() == needle:
            target = m
            break
    if target is None:
        return ""

    target_depth = len(target.group(1))
    start = target.start()

    # End at the next header of equal-or-shallower depth.
    end = len(body)
    for m in matches:
        if m.start() <= start:
            continue
        if len(m.group(1)) <= target_depth:
            end = m.start()
            break

    return body[start:end].rstrip()


def list_topic_sections(name: str) -> list[str]:
    """Return the H2/H3 section headers of a topic, in document order.

    Useful when the agent wants to see which sub-sections exist before
    requesting one — keeps section reads informed rather than guessing.
    """
    body = read_topic_body(name)
    if not body:
        return []
    return [
        m.group(2).strip()
        for m in _SECTION_HEADER_RE.finditer(body)
        if 2 <= len(m.group(1)) <= 3
    ]


# ─── Topic index helpers (keep MEMORY.md in sync) ────────────────────────────

_TOPIC_INDEX_HEADING = "## Topic Files Index"


_MAX_TOPIC_DESC_CHARS = 120
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _sanitize_topic_description(raw: str) -> str:
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


def _extract_topic_description(body: str) -> str:
    """Extract a short description from topic body for the MEMORY.md index.

    Uses the first ``#`` heading text. Falls back to the first non-empty
    non-frontmatter line. Always passed through ``_sanitize_topic_description``
    so the index entry can never inject markdown control chars or unbounded
    text into the always-loaded prompt.
    """
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return _sanitize_topic_description(stripped)
        if stripped and not stripped.startswith("---"):
            return _sanitize_topic_description(stripped)
    return "(no description)"


def _upsert_topic_index_entry(name: str, description: str) -> None:
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
            write_memory_md(f"# Agent Memory\n\n{_TOPIC_INDEX_HEADING}\n{new_entry}\n")
            return

        lines = content.split("\n")
        if _TOPIC_INDEX_HEADING in content:
            result: list[str] = []
            in_index = False
            replaced = False
            appended = False

            for line in lines:
                stripped = line.strip()
                if stripped == _TOPIC_INDEX_HEADING:
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
            _TOPIC_INDEX_HEADING,
        }
        for line in lines:
            if (
                not inserted
                and line.strip().startswith("## ")
                and line.strip() not in protected_before_index
            ):
                result.append(_TOPIC_INDEX_HEADING)
                result.append(new_entry)
                result.append("")
                inserted = True
            result.append(line)
        if not inserted:
            if result and result[-1] != "":
                result.append("")
            result.append(_TOPIC_INDEX_HEADING)
            result.append(new_entry)
        write_memory_md("\n".join(result))


def _add_topic_index_entry(name: str, description: str) -> None:
    """Compatibility wrapper: upsert the topic entry in MEMORY.md."""
    _upsert_topic_index_entry(name, description)


def _remove_topic_index_entry(name: str) -> None:
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
            if line.strip() == _TOPIC_INDEX_HEADING:
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


def write_topic(name: str, content: str) -> None:
    """Write content to a topic file with created/updated frontmatter.

    If the file already exists, preserves the original ``created`` timestamp
    and updates ``updated``.  New files get both set to now.

    If the caller passes content that already starts with ``---`` frontmatter,
    we merge timestamps into it rather than double-wrapping.

    For NEW topics (file doesn't exist yet), automatically adds an index entry
    to the ``## Topic Files Index`` section of MEMORY.md.

    Raises ``ValueError`` for invalid topic names — closes the path-traversal
    lever.
    """
    ensure_dirs()
    filepath = _topic_path(name)
    if filepath is None:
        raise ValueError(
            f"invalid topic name {name!r}: must match [A-Za-z0-9][A-Za-z0-9_.-]{{0,79}}"
        )
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

    # Atomic write: topic body + frontmatter go to a temp file first, then
    # os.replace() onto the final path. Without this, `write_topic` can leave
    # a half-written file on crash or race with a concurrent writer. The
    # topic-name regex serialises partitions at the path level — atomicity
    # is the per-partition guarantee.
    serialized = _build_frontmatter(final_meta) + body
    if body and not body.endswith("\n"):
        serialized += "\n"
    from ..utils.files import atomic_write as _atomic_write
    _atomic_write(filepath, serialized)

    # Keep the always-loaded topic index current for both new and rewritten
    # topics. The helper is idempotent and locked against concurrent writers.
    description = _extract_topic_description(body)
    _upsert_topic_index_entry(name, description)


def delete_topic(name: str) -> bool:
    """Delete a topic file and remove its index entry from MEMORY.md.

    Returns True if the file was deleted. Returns False on invalid name OR
    on missing file — both are "topic does not exist" from the caller's POV.
    """
    filepath = _topic_path(name)
    if filepath is None:
        return False
    try:
        os.remove(filepath)
        _remove_topic_index_entry(name)
        return True
    except FileNotFoundError:
        return False


# ─── Journal tier (Tier 3: never auto-loaded) ────────────────────────────────
#
# Why a separate tier? Anthropic Claude Code docs and the consensus across
# Letta/Mem0/LangMem/Zep/A-MEM are unanimous: chronological "what I did today"
# notes do NOT belong in the always-loaded memory file. They cause:
#   - prompt-cache invalidation (mutating the cached prefix → ~12× cost penalty)
#   - context rot (NoLiMa: Claude 3.5 Sonnet effective length only ~4K tokens)
#   - "lost in the middle" attention degradation (Liu et al., TACL 2023)
#
# Journal entries live under data/memory/journal/YYYY-MM.md and are read on
# demand only (e.g. "what did we work on last Tuesday?"). They are NOT injected
# into the system prompt.

_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def _journal_path(month: str | None = None) -> str:
    """Return the journal file path for the given month (YYYY-MM).

    Validates ``month`` to prevent path traversal (``../../etc/passwd``) —
    low-severity in a single-user agent, but cheap to close the door on a
    future caller passing user-controlled strings.
    """
    if month is None:
        month = datetime.now(ASIA_SHANGHAI_TZ).strftime("%Y-%m")
    if not _MONTH_RE.match(month):
        raise ValueError(f"journal month must match YYYY-MM, got {month!r}")
    return os.path.join(_cfg.JOURNAL_DIR, f"{month}.md")


def list_journals() -> list[str]:
    """List all journal months (YYYY-MM strings), newest first."""
    ensure_dirs()
    months = []
    if os.path.exists(_cfg.JOURNAL_DIR):
        for f in os.listdir(_cfg.JOURNAL_DIR):
            if f.endswith(".md") and _MONTH_RE.match(f[:-3]):
                months.append(f[:-3])
    return sorted(months, reverse=True)


def read_journal(month: str | None = None) -> str:
    """Read a single journal month. Empty string if no entries."""
    ensure_dirs()
    path = _journal_path(month)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


# Journal categories live in markdown headers ("## YYYY-MM-DD HH:MM [cat]")
# so a category containing "[", "]", "\n", or other markdown control chars
# breaks the journal structure (and the `## ` header parser used by
# search.py::_split_sections). Validate against a permissive identifier
# regex — more relaxed than _ALLOWED_MEMORY_CATEGORIES because journal
# categories are freeform tags ("ship", "debug", "refactor", "session",
# "memory_revision", "codex_review", ...). Invalid input falls back to
# "note" silently — journal writes are best-effort and we don't want a
# malformed tag to lose the actual entry body.
_JOURNAL_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]{0,30}$")


def _sanitize_journal_category(raw: str | None) -> str:
    """Normalize a journal category for safe use in a markdown header.

    Returns "note" for any input that fails validation. Lowercases the
    input and strips surrounding whitespace before checking the pattern.
    """
    if not raw:
        return "note"
    cat = raw.strip().lower()
    if not cat or not _JOURNAL_CATEGORY_RE.match(cat):
        return "note"
    return cat


def append_journal(text: str, category: str = "note") -> str:
    """Append a dated entry to the current month's journal file.

    Returns the journal-relative path that was written, so callers (and the
    agent) know where the note lives without having to guess.

    The ``category`` is sanitized (``_sanitize_journal_category``) before
    being interpolated into the markdown header — a malformed tag from a
    user-supplied facade call would otherwise break the journal section
    structure that ``search.py::_split_sections`` relies on.

    Concurrency: local threads serialize the header+body append under
    ``_JOURNAL_LOCK``. The header still uses O_CREAT|O_EXCL so a separate
    process racing on the first entry of a fresh month cannot produce a
    duplicate ``# Journal —`` heading.
    """
    ensure_dirs()
    safe_category = _sanitize_journal_category(category)
    now = datetime.now(ASIA_SHANGHAI_TZ)
    month = now.strftime("%Y-%m")
    timestamp = now.strftime("%Y-%m-%d %H:%M")
    path = _journal_path(month)

    with _JOURNAL_LOCK:
        # Race-free header install: whoever wins the O_EXCL create writes the
        # header; everyone else raises FileExistsError and skips straight to the
        # append step.
        header = (
            f"# Journal — {month}\n\n"
            "Tier 3 / append-only / NEVER auto-loaded. Chronological session "
            "notes live here so they don't pollute MEMORY.md (the always-loaded "
            "index) or topic files (curated knowledge).\n\n"
        )
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                os.write(fd, header.encode("utf-8"))
            finally:
                os.close(fd)
        except FileExistsError:
            pass  # another writer already installed the header — safe to append

        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## {timestamp} [{safe_category}]\n{text.rstrip()}\n\n")

    return f"data/memory/journal/{month}.md"


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


# ─── Consolidate (dedup analysis, no LLM call) ───────────────────────────────

def consolidate_memory() -> str:
    """Analyze MEMORY.md for likely-duplicate entries and report groupings.

    Pure heuristic / read-only — does NOT delete anything. The agent can use
    the report to decide which entries to merge or move out manually.

    Heuristics:
      1. Group lines by category tag (`[gotcha]`, `[tip]`, `[note]`, etc.)
      2. Flag categories with >5 entries (consolidation candidates)
      3. Within each large category, flag pairs whose significant-token sets
         overlap by >=4 (likely related topics). Token extraction covers
         ASCII identifiers, version numbers, and CJK words so the heuristic
         works on bilingual memory.
      4. Always flag any line longer than 400 chars (belongs in a topic file)
      5. Always flag any categorized line with a YYYY-MM-DD date in its body
         (likely belongs in the journal tier, not the index)
    """
    content = read_memory_md()
    if not content:
        return "MEMORY.md is empty — nothing to consolidate."

    lines = content.splitlines()
    cat_re = re.compile(r"^\[(?P<cat>[a-z_]+)\]\s*(?P<body>.*)$", re.IGNORECASE)

    by_cat: dict[str, list[tuple[int, str]]] = {}
    long_lines: list[tuple[int, str]] = []
    journal_candidates: list[tuple[int, str, str]] = []  # (line_no, category, preview)
    date_re = re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b")

    for idx, line in enumerate(lines, start=1):
        m = cat_re.match(line.strip())
        if not m:
            continue
        cat = m.group("cat").lower()
        body = m.group("body")
        by_cat.setdefault(cat, []).append((idx, body))
        if len(line) > 400:
            long_lines.append((idx, body[:80]))
        # Dated entries in ANY category are journal candidates — not only [note].
        # A dated [gotcha] or [tip] is equally an activity-log smell.
        if date_re.search(body):
            journal_candidates.append((idx, cat, body[:80]))

    report = ["📊 MEMORY.md consolidation analysis", ""]
    issues_found = False

    # 1. Per-category counts
    report.append("## Categories")
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        if len(by_cat[cat]) > 5:
            report.append(f"  [{cat}]: {len(by_cat[cat])} entries  ⚠️ consider consolidating")
            issues_found = True
        else:
            report.append(f"  [{cat}]: {len(by_cat[cat])} entries")

    # 2. Lines that are too long for the always-loaded tier
    if long_lines:
        issues_found = True
        report.append("")
        report.append("## Lines > 400 chars (move to topics/<name>.md)")
        for ln, preview in long_lines:
            report.append(f"  L{ln}: {preview}…")

    # 3. Dated entries in any category → belong in journal
    if journal_candidates:
        issues_found = True
        report.append("")
        report.append("## Dated entries (move to journal/YYYY-MM.md)")
        for ln, cat, preview in journal_candidates:
            report.append(f"  L{ln} [{cat}]: {preview}")

    # 4. Cheap overlap dedup hints inside large categories.
    # Token extraction covers three shapes:
    #   • ASCII identifiers of 3+ letters with ._- internal punctuation so
    #     version numbers and dotted paths stay intact ("Python3.14",
    #     "jyagent.tools.facades").
    #   • CJK character bigrams (Chinese/Japanese/Korean). CJK has no
    #     inter-word whitespace, so "用户偏好中文" as one "run" is useless for
    #     dedup — instead we emit "用户", "户偏", "偏好", "好中", "中文" and
    #     let the set-overlap heuristic do its job. Stop-words are ASCII-only
    #     (no curated CJK stop list; bigram noise is low in practice).
    ascii_tok = re.compile(r"[A-Za-z][\w.\-]{2,}")
    cjk_run = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]{2,}")

    def _sig_words(s: str) -> set[str]:
        out = {
            tok.lower()
            for tok in ascii_tok.findall(s)
            if tok.lower() not in _STOP_WORDS
        }
        for run in cjk_run.findall(s):
            for i in range(len(run) - 1):
                out.add(run[i : i + 2])
        return out

    overlap_hints: list[str] = []
    for cat, entries in by_cat.items():
        if len(entries) < 4:
            continue
        word_sets = [(ln, body, _sig_words(body)) for ln, body in entries]
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                ln_a, body_a, ws_a = word_sets[i]
                ln_b, body_b, ws_b = word_sets[j]
                shared = ws_a & ws_b
                if len(shared) >= 4:
                    overlap_hints.append(
                        f"  [{cat}] L{ln_a} ↔ L{ln_b} share: "
                        f"{', '.join(sorted(shared)[:6])}"
                    )

    if overlap_hints:
        issues_found = True
        report.append("")
        report.append("## Possible duplicate pairs (>=4 shared content words)")
        report.extend(overlap_hints[:20])
        if len(overlap_hints) > 20:
            report.append(f"  … and {len(overlap_hints) - 20} more pairs")

    warning = memory_index_size_warning()
    if warning:
        issues_found = True
        report.append("")
        report.append(warning)

    if not issues_found:
        report.append("")
        report.append("✅ No obvious consolidation candidates found.")

    return "\n".join(report)


_STOP_WORDS = {
    "with", "from", "that", "this", "have", "when", "then", "than",
    "they", "them", "their", "there", "where", "which", "what",
    "your", "yours", "been", "being", "would", "could", "should", "must",
    "will", "shall", "does", "doesn", "didn", "isn", "wasn", "aren", "weren",
    "about", "above", "below", "between", "after", "before", "during",
    "while", "until", "because", "since", "though", "although",
    "also", "only", "even", "just", "still", "very", "more", "most", "less",
    "least", "other", "another", "some", "many", "each", "every", "both",
    "either", "neither", "such", "same", "different",
    "into", "onto", "over", "under", "through", "across", "around",
    "always", "never", "often", "sometimes", "usually",
    "make", "made", "makes", "making", "take", "taken", "takes", "taking",
    "give", "gave", "given", "gives", "giving", "want", "wants", "need",
    "needs", "see", "saw", "seen", "look", "looks", "looking",
    "good", "better", "best", "bad", "worse", "worst",
    "true", "false", "none", "null",
}


# ─── High-level operations (used by manage_memory tool) ──────────────────────

def remember(text: str, category: str = "", *, suppress_warning: bool = False) -> str:
    """Remember a durable fact or learning by appending to MEMORY.md.

    NOTE: This appends to the always-loaded index. Use only for data-independent
    rules / facts that prevent future mistakes. Ephemeral task notes ("today I
    finished X") belong in the journal tier — call ``append_journal`` instead.

    The return string concatenates a soft-cap warning when MEMORY.md is near
    its load limit. Set ``suppress_warning=True`` for programmatic callers /
    tests that parse the return value.
    """
    entry, cat = _validate_memory_entry(text, category)
    prefix = f"[{cat}] " if cat else "- "
    append_memory_md(f"{prefix}{entry}")
    msg = f"Remembered: {entry[:100]}"
    if not suppress_warning:
        warning = memory_index_size_warning()
        if warning:
            msg += "\n" + warning
    return msg


def forget(keyword: str) -> str:
    """Forget memories matching a keyword.

    Wraps ``forget_from_memory_md`` with default safeties (≥6-char keyword,
    skips lines under Behavioral Rules / User Profile / User Preferences and
    ``#`` headings). Returns a human-readable preview of what was removed
    plus a count of protected lines that were preserved.
    """
    try:
        removed, preview, protected_skipped = forget_from_memory_md(keyword)
    except ValueError as e:
        return f"Refused to forget: {e}"

    suffix = ""
    if protected_skipped:
        suffix = (
            f" ({protected_skipped} protected line(s) preserved — Behavioral "
            f"Rules / User Profile / User Preferences are never deleted by keyword)"
        )

    if removed == 0:
        if protected_skipped:
            return (
                f"No entries removed — all {protected_skipped} match(es) were in "
                "protected sections (Behavioral Rules / User Profile / User Preferences)"
            )
        return f"No entries found matching '{keyword}'"

    # Show up to 3 removed lines so the user can see what was lost.
    sample = "\n".join(f"  - {ln.strip()[:120]}" for ln in preview[:3])
    more = f"\n  …and {removed - 3} more" if removed > 3 else ""
    return (
        f"Removed {removed} entries matching '{keyword}'{suffix}:\n{sample}{more}"
    )



def show_memory() -> str:
    """Show all memory contents."""
    parts = []

    content = read_memory_md()
    if content:
        line_count = len(content.splitlines())
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

    journals = list_journals()
    if journals:
        j_lines = []
        for m in journals[:6]:  # show at most 6 most-recent months
            jc = read_journal(m)
            j_lines.append(f"  📓 {m}.md ({len(jc.split(chr(10)))} lines, {len(jc)} chars)")
        more = "" if len(journals) <= 6 else f"\n  … and {len(journals) - 6} older months"
        parts.append(f"📓 JOURNAL ({len(journals)} months, on-demand only):\n" + "\n".join(j_lines) + more)

    if not parts:
        return "🧠 Memory is empty. I'll start learning about you as we interact!"

    warning = memory_index_size_warning()
    suffix = ("\n\n" + warning) if warning else ""
    return "🧠 SELF-USE MEMORY\n" + "\n\n".join(parts) + suffix
