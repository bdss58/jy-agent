# System prompt injection — build memory context for the system prompt.

from ..config import MAX_MEMORY_PROMPT_CHARS
from .operations import read_memory_index, list_topics
from .sessions import SessionSummaries


def build_memory_context(query: str = "") -> str:
    """Build a memory context string to inject into the system prompt.

    Layout:
      1. MEMORY.md index (always, first 200 lines / 25KB) — includes user profile section
      2. Recent Session Summaries (always, last 5)

    Topic files are NOT injected here — agent reads them on-demand via read_file.
    """
    sections = []

    # 1. MEMORY.md index (with Claude Code limits) — includes user profile
    memory_index = read_memory_index()
    if memory_index:
        sections.append(f"## Agent Memory (MEMORY.md)\n{memory_index}")

    # 2. Session Summaries
    sessions = SessionSummaries()
    session_text = sessions.to_prompt_text()
    if session_text:
        sections.append(f"## Recent Sessions\n{session_text}")

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

Memory instructions:
- MEMORY.md is the index. Keep it concise (under 200 lines). 
- Move detailed knowledge to topic files in data/memory/topics/<name>.md
- Read topic files on-demand with read_file when you need details.
- To remember something: use manage_memory tool, or directly write files.
- To reorganize: rewrite MEMORY.md and topic files with write_file.
"""
