# System prompt injection — build memory context for the system prompt.

from ..config import MAX_MEMORY_PROMPT_CHARS
from .operations import read_memory_index, list_topics


def build_memory_context(query: str = "") -> str:
    """Build a memory context string to inject into the system prompt.

    Layout:
      1. MEMORY.md index (always, first 200 lines / 25KB) — includes user profile section

    Topic files are NOT injected here — agent reads them on-demand via read_file.
    """
    sections = []

    # 1. MEMORY.md index (with Claude Code limits) — includes user profile
    memory_index = read_memory_index()
    if memory_index:
        sections.append(f"## Agent Memory (MEMORY.md)\n{memory_index}")

    if not sections:
        return ""

    full_text = "\n\n".join(sections)
    if len(full_text) > MAX_MEMORY_PROMPT_CHARS:
        full_text = full_text[:MAX_MEMORY_PROMPT_CHARS] + "\n... (memory truncated)"

    # Build topic file listing for agent awareness
    topics = list_topics()
    topic_listing = ""
    if topics:
        topic_listing = "\n\nTopic files available (read with `read_file`): " + \
            ", ".join(f"data/memory/topics/{t}.md" for t in topics)

    return f"""
═══ SELF-USE MEMORY (automatically maintained) ═══
{full_text}{topic_listing}
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
