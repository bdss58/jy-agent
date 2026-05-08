# System prompt injection — build memory context for the system prompt.

from ..config import MAX_MEMORY_PROMPT_CHARS
from ._index import read_memory_index
from ._topics import list_topics


# Heading MEMORY.md uses for its own topic index. When present, we skip the
# standalone "Topic files available:" listing below to avoid duplicating the
# same information twice in the always-loaded system prompt.
_TOPIC_INDEX_HEADING = "## Topic Files Index"


def build_memory_context(query: str = "") -> str:
    """Build a memory context string to inject into the system prompt.

    Layout:
      1. MEMORY.md index (always, first 200 lines / 25KB) — includes user profile section
      2. Topic-file discovery hint — ONLY when MEMORY.md lacks its own
         ``## Topic Files Index`` section, so we never duplicate the same
         listing twice in the always-loaded prompt.

    The combined body is capped by ``MAX_MEMORY_PROMPT_CHARS`` AFTER all
    sections have been assembled (fixed 2026-05-06; previously the cap was
    applied before the topic listing, letting the listing bypass it).

    Topic file BODIES are not injected here — the agent reads them on-demand
    via ``read_file`` or ``manage_memory(action='search')``.
    """
    memory_index = read_memory_index()
    if not memory_index:
        return ""

    # Assemble the full body first: MEMORY.md index, then the topic-file
    # discovery hint (only if MEMORY.md doesn't already list them).
    body_parts: list[str] = [f"## Agent Memory (MEMORY.md)\n{memory_index}"]

    topics = list_topics()
    if topics and _TOPIC_INDEX_HEADING not in memory_index:
        listing = "Topic files available (read with `read_file`): " + ", ".join(
            f"data/memory/topics/{t}.md" for t in topics
        )
        body_parts.append(listing)

    full_body = "\n\n".join(body_parts)

    # Cap AFTER assembly so the standalone topic listing (if present) counts
    # toward the budget rather than bypassing it.
    if len(full_body) > MAX_MEMORY_PROMPT_CHARS:
        full_body = full_body[:MAX_MEMORY_PROMPT_CHARS] + "\n... (memory truncated)"

    return f"""
═══ SELF-USE MEMORY (automatically maintained) ═══
{full_body}
═══ END MEMORY ═══

Memory tier model (do not violate):
  Tier 1 — MEMORY.md: ALWAYS LOADED, hard cap 200 lines / 25 KB. Durable rules
           and facts only. Bloat causes attention degradation + cache invalidation.
  Tier 2 — data/memory/topics/<name>.md: curated detail, read on demand.
  Tier 3 — data/memory/journal/YYYY-MM.md: append-only session notes; NEVER
           auto-loaded. The home for "what I did today" / commit-style entries.

Routing:
  - Durable rule (passes "would removing cause a future mistake?") → manage_memory(action='remember')
  - Extended detail / curated knowledge → manage_memory(action='topic', text='write:<name>|<body>')
  - Chronological session note → manage_memory(action='journal', text=..., category=...)
  - Audit MEMORY.md for bloat / dedup → manage_memory(action='consolidate')
  - Read topic / journal on demand with read_file.
"""
