# Shared markdown parsing helpers.
#
# Both Tier-2 (``_topics``) and the BM25 search (``search``) need to
# understand markdown frontmatter and ATX section headers. Before this
# module existed:
#   - ``_topics._parse_frontmatter`` parsed the ``--- … ---`` YAML block
#     using a string-split path, while ``search._strip_frontmatter`` matched
#     the same block with a separate regex. Either could drift.
#   - ``_topics._SECTION_HEADER_RE`` (matches ``#``-``######``) and
#     ``search._H2_H3`` (matches only ``##``/``###``) shared the same
#     concept (ATX header in column 0) with subtly different acceptance.
#
# Now both responsibilities live here. ``_topics`` and ``search`` import
# from this module and keep their old underscore-prefixed names as shim
# re-exports so existing tests / callers continue to work.

from __future__ import annotations

import re

_FRONTMATTER_SEP = "---"

# Regex form of the frontmatter block. Used by ``strip_frontmatter`` for
# zero-allocation cases (search's BM25 indexer hits this in a hot loop).
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)

# All ATX header depths (``#`` to ``######``) at column 0. Used by
# ``read_topic_section`` to locate any-depth header by its text.
SECTION_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Just ``##`` / ``###`` headers — the granularity BM25 chunks at. The
# top-level ``#`` is treated as a file title and stays with the preamble.
H2_H3_HEADER_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Parse YAML-style frontmatter from a markdown body.

    Returns ``(metadata_dict, body_text)``. If no frontmatter, returns
    ``({}, raw)``. Tolerant: only ``key: value`` lines are kept; nested or
    quoted YAML is not supported (and we don't want it — frontmatter here is
    a flat created/updated/tags dict).
    """
    if not raw.startswith(_FRONTMATTER_SEP):
        return {}, raw

    parts = raw.split(_FRONTMATTER_SEP, 2)  # ['', yaml_block, body]
    if len(parts) < 3:
        return {}, raw

    meta: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    body = parts[2].lstrip("\n")
    return meta, body


def build_frontmatter(meta: dict) -> str:
    """Serialize a flat metadata dict into a ``--- … ---`` YAML block.

    Output always ends with a newline so the caller can concatenate the
    body without worrying about glue.
    """
    lines = [_FRONTMATTER_SEP]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines.append(_FRONTMATTER_SEP)
    return "\n".join(lines) + "\n"


def strip_frontmatter(body: str) -> str:
    """Return ``body`` with any leading ``--- … ---`` block removed."""
    m = _FRONTMATTER_RE.match(body)
    return body[m.end():] if m else body


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown body into ``(section_header, section_text)`` pairs.

    Splits on ``##``/``###`` only. The first chunk (before any such header)
    gets section = "". Each header's text is included at the start of its
    chunk so searches that hit the header still surface the full section.
    Returns ``[]`` for an empty body.
    """
    headers: list[tuple[int, int, str]] = []  # (start, depth, text)
    for m in H2_H3_HEADER_RE.finditer(body):
        depth = len(m.group(1))
        headers.append((m.start(), depth, m.group(2).strip()))

    if not headers:
        return [("", body.strip())] if body.strip() else []

    chunks: list[tuple[str, str]] = []
    # Preamble (before the first header)
    if headers[0][0] > 0:
        pre = body[: headers[0][0]].strip()
        if pre:
            chunks.append(("", pre))

    for i, (start, _depth, text) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(body)
        section_text = body[start:end].strip()
        if section_text:
            chunks.append((text, section_text))

    return chunks
