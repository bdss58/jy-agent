# Tier 2 — Topic files (curated, on-demand extended detail).
#
# Owns: frontmatter parsing/writing, topic CRUD (`list_topics`, `read_topic*`,
# `write_topic`, `delete_topic`, `list_topic_sections`, `read_topic_section`),
# and the helpers that keep MEMORY.md's "Topic Files Index" section in sync
# whenever a topic is created, rewritten, or deleted (`_upsert_topic_index_entry`,
# `_add_topic_index_entry`, `_remove_topic_index_entry`).
#
# Extracted from ``operations.py`` (2026-05-06) as part of the split that
# made each tier its own module.  Existing callers continue to import via
# ``jyagent.memory.operations`` (which re-exports this).
#
# Dependencies: ``_index`` for the MEMORY.md lock + read/write primitives
# (the index-entry helpers do read-modify-write on MEMORY.md), and
# ``jyagent.utils.files.atomic_write`` for crash-safe topic writes.

from __future__ import annotations

import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from .. import config as _cfg
from ._index import _MEMORY_MD_LOCK, read_memory_md, write_memory_md, ensure_dirs


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


