"""Tool entry point for the self-use memory system.

Routes action verbs (remember, forget, show, search, topic, goal, journal,
consolidate) to implementations in ``jyagent.memory``.  Not a thin wrapper
— the action dispatcher, input validation, and error-envelope formatting
all live here; the storage/BM25/journal implementation lives in
``jyagent.memory``.

History: extracted from the former ``tools/facades.py`` (2026-05).
"""
from ..runtime.tools.result import ToolResult


def manage_memory(action: str, text: str = "", category: str = "") -> ToolResult:
    """Manage the agent's self-use memory system. Actions: 'remember' (save a DURABLE learning/fact to MEMORY.md — use sparingly, data-independent rules only), 'forget' (remove memories by keyword), 'show' (display all memories), 'search' (BM25 over topic+journal bodies), 'topic' (manage curated topic files: list/read/write/delete/sections), 'goal' (add/complete a goal), 'journal' (append a dated session note to data/memory/journal/YYYY-MM.md — never auto-loaded, for 'what I did today' style entries), 'consolidate' (analyze MEMORY.md for dedup / bloat candidates — read-only). Three tiers: always-loaded index (MEMORY.md) / curated on-demand (topics/) / chronological on-demand (journal/). To revise an existing rule: write a 'journal' entry recording the change, then 'forget' the old keyword and 'remember' the new fact — keeps Tier 1 lean while preserving audit history in Tier 3."""
    from ..memory import (
        remember, forget, show_memory,
        list_topics, read_topic, write_topic, delete_topic,
        read_topic_section, list_topic_sections,
        append_journal, list_journals, read_journal, consolidate_memory,
    )
    from ..memory.search import search_memory, render_hits

    try:
        if action == "remember":
            if not text:
                return ToolResult("Error: 'text' parameter required for 'remember' action", is_error=True)
            return ToolResult(f"🧠 {remember(text, category)}")

        elif action == "forget":
            if not text:
                return ToolResult("Error: 'text' parameter required for 'forget' action (keyword to match)", is_error=True)
            return ToolResult(f"🧠 {forget(text)}")

        elif action == "search":
            if not text:
                return ToolResult(
                    "Error: 'text' parameter required for 'search' action (query string)",
                    is_error=True,
                )
            # Default to recent journal months only — full-history search is
            # an O(N_months × tokens) cost that grows with project lifetime.
            # Pass `category="all"` to opt into searching every journal month.
            jmonths = None if category.strip().lower() == "all" else 6
            hits = search_memory(text, top_k=5, journal_months=jmonths)
            return ToolResult(render_hits(hits))

        elif action == "show":
            return ToolResult(show_memory())

        elif action == "journal":
            if not text:
                return ToolResult("Error: 'text' parameter required for 'journal' action", is_error=True)
            cat = category or "note"
            path = append_journal(text, cat)
            return ToolResult(f"📓 Journal entry appended to {path} [{cat}]")

        elif action == "consolidate":
            return ToolResult(consolidate_memory())

        elif action == "topic":
            if not text:
                return ToolResult("Error: 'text' parameter required. Formats: 'list', 'read:<name>', 'read:<name>#<section>', 'sections:<name>', 'write:<name>|<content>', 'delete:<name>'", is_error=True)

            if text == "list":
                topics = list_topics()
                if not topics:
                    return ToolResult("📂 No topic files yet. Create with topic action: 'write:<name>|<content>'")
                lines = []
                for t in topics:
                    tc = read_topic(t)
                    lines.append(f"  📄 {t}.md ({len(tc.split(chr(10)))} lines, {len(tc)} chars)")
                return ToolResult("📂 Topic files (" + str(len(topics)) + "):\n" + "\n".join(lines))

            elif text.startswith("read:"):
                name_spec = text[5:].strip()
                # Allow `read:<name>#<section>` to fetch one section instead
                # of the whole file. The `#` separator is markdown-friendly
                # and avoids ambiguity with topic names that contain `:`.
                if "#" in name_spec:
                    name, section = name_spec.split("#", 1)
                    name = name.strip()
                    section = section.strip()
                    body = read_topic_section(name, section)
                    if not body:
                        sections = list_topic_sections(name)
                        avail = ", ".join(sections) if sections else "(no sections)"
                        return ToolResult(
                            f"Section '{section}' not found in '{name}'. Available: {avail}",
                            is_error=True,
                        )
                    return ToolResult(f"📄 Topic: {name}.md#{section}\n\n{body}")
                name = name_spec
                content = read_topic(name)
                if not content:
                    return ToolResult(f"Topic '{name}' not found. Available: {', '.join(list_topics()) or 'none'}", is_error=True)
                return ToolResult(f"📄 Topic: {name}.md" + "\n\n" + content)

            elif text.startswith("sections:"):
                name = text[9:].strip()
                sections = list_topic_sections(name)
                if not sections:
                    if not read_topic(name):
                        return ToolResult(
                            f"Topic '{name}' not found. Available: {', '.join(list_topics()) or 'none'}",
                            is_error=True,
                        )
                    return ToolResult(f"📄 {name}.md has no ## or ### sections")
                lines = [f"  • {s}" for s in sections]
                return ToolResult(
                    f"📄 {name}.md sections ({len(sections)}):\n" + "\n".join(lines)
                )

            elif text.startswith("write:"):
                rest = text[6:]
                if "|" not in rest:
                    return ToolResult("Error: format is 'write:<name>|<content>'", is_error=True)
                name, content = rest.split("|", 1)
                name = name.strip()
                content = content.strip()
                try:
                    write_topic(name, content)
                except ValueError as e:
                    return ToolResult(f"Error: {e}", is_error=True)
                return ToolResult(f"📄 Topic '{name}.md' written ({len(content)} chars)")

            elif text.startswith("delete:"):
                name = text[7:].strip()
                if delete_topic(name):
                    return ToolResult(f"📄 Topic '{name}.md' deleted")
                return ToolResult(f"Topic '{name}' not found", is_error=True)

            else:
                return ToolResult("Error: Unknown topic command. Use: 'list', 'read:<name>', 'read:<name>#<section>', 'sections:<name>', 'write:<name>|<content>', 'delete:<name>'", is_error=True)

        elif action == "goal":
            if not text:
                return ToolResult("Error: 'text' parameter required", is_error=True)
            if text.lower().startswith("done:"):
                return ToolResult(f"🧠 {forget(text[5:].strip())}")
            return ToolResult(f"🧠 {remember(text, 'goal')}")

        else:
            return ToolResult(f"Error: Unknown action '{action}'. Valid: remember, forget, show, search, topic, goal, journal, consolidate", is_error=True)

    except Exception as e:
        return ToolResult(f"Error managing memory: {e}", is_error=True)
