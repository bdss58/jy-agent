# Facade tools: manage_memory, manage_skills
# These are thin wrappers that delegate to the memory/ and skills modules.

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


def manage_skills(action: str, name: str = "", description: str = "",
                  instructions: str = "", resource_path: str = "") -> ToolResult:
    """Manage Agent Skills (agentskills.io). Actions:
    'load' (one-shot: return full SKILL.md body as tool result — PREFERRED for model use),
    'pin' (session-long: keep instructions injected on every turn — use only when user asks),
    'list' (show all skills), 'deactivate' (un-pin; no name = un-pin all),
    'info' (show skill details), 'create' (create new skill), 'delete' (remove skill),
    'resources' (list skill files), 'read' (read a skill resource file),
    'reload' (re-scan skills directory)."""
    from ..skills import get_skill_manager
    from ..config import MAX_TOOL_RESULT_CHARS

    try:
        mgr = get_skill_manager()

        if action == "list":
            catalog = mgr.get_catalog()
            if not catalog:
                return ToolResult("📦 No skills found. Create one with manage_skills(action='create', ...)")
            lines = ["📦 Agent Skills:"]
            for entry in catalog:
                status = "📌 PINNED" if entry["pinned"] else "  📦"
                lines.append(f"  {status} {entry['name']}: {entry['description'][:100]}")
            lines.append(f"\n  Total: {len(catalog)} skills, {sum(1 for e in catalog if e['pinned'])} pinned")
            return ToolResult("\n".join(lines))

        elif action == "load":
            # One-shot: return the SKILL.md body as the tool result text so it
            # enters conversation history exactly once. Spec-conformant
            # progressive disclosure (agentskills.io / Claude Code).
            if not name:
                return ToolResult("Error: 'name' parameter required", is_error=True)
            skill = mgr.get_skill(name)
            if not skill:
                return ToolResult(
                    f"Error: Skill '{name}' not found. Use manage_skills(action='list').",
                    is_error=True,
                )
            # Short-circuit if the user already pinned this skill — its body is
            # already injected on every turn, calling load would double-bill.
            if name in mgr.get_pinned_skills():
                return ToolResult(
                    f"📌 Skill '{name}' is already pinned. Its full instructions "
                    f"are attached to every user message — no need to load again. "
                    f"Use manage_skills(action='deactivate', name='{name}') to un-pin."
                )

            body = skill.get("body", "")
            # Defensively escape closing tags inside the body so a SKILL.md
            # containing literal "</instructions>" or "</skill>" can't break
            # out of the wrapper. Replace with a visible sentinel so authors
            # can spot the substitution if they ever inspect the rendered
            # tool result. (Safety > authoring convenience.)
            def _safe_body(s: str) -> str:
                return (s.replace("</instructions>", "<\u200b/instructions>")
                         .replace("</skill>",        "<\u200b/skill>"))

            body = _safe_body(body)
            # Reserve ~1 KB headroom for the wrapper XML so the result does not
            # collide with MAX_TOOL_RESULT_CHARS (currently 8000) and silently
            # truncate the tail of SKILL.md.
            wrapper_budget = 1024
            body_cap = max(MAX_TOOL_RESULT_CHARS - wrapper_budget, 1000)
            truncated = len(body) > body_cap
            body_part = body[:body_cap]

            parts = [f'<skill name="{name}">']
            if skill.get("allowed_tools"):
                parts.append(f"<allowed_tools>{', '.join(skill['allowed_tools'])}</allowed_tools>")
            resources = mgr.list_resources(name)
            if resources:
                parts.append(f"<resources>{', '.join(resources)}</resources>")
            parts.append("<instructions>")
            parts.append(body_part)
            if truncated:
                parts.append(
                    f"[... SKILL.md body truncated from {len(body)} to {body_cap} chars; "
                    f"read remaining content via manage_skills(action='read', name='{name}', resource_path=...)]"
                )
            parts.append("</instructions>")
            parts.append("</skill>")
            return ToolResult("\n".join(parts))

        elif action == "pin":
            if not name:
                return ToolResult("Error: 'name' parameter required", is_error=True)
            success = mgr.pin(name)
            if success:
                return ToolResult(
                    f"📌 Skill '{name}' pinned. Its full instructions will be "
                    f"prepended to every user message until deactivated. For one-shot "
                    f"use prefer manage_skills(action='load', name='{name}')."
                )
            return ToolResult(
                f"Error: Skill '{name}' not found. Use manage_skills(action='list').",
                is_error=True,
            )

        elif action == "deactivate":
            if not name:
                # No name → un-pin all (Codex-suggested bonus).
                pinned = mgr.get_pinned_skills()
                if not pinned:
                    return ToolResult("📦 No skills are currently pinned.")
                mgr.unpin_all()
                return ToolResult(f"📦 Un-pinned {len(pinned)} skill(s): {', '.join(pinned)}.")
            success = mgr.unpin(name)
            if success:
                return ToolResult(f"📦 Skill '{name}' un-pinned.")
            return ToolResult(f"Error: Skill '{name}' is not pinned.", is_error=True)

        elif action == "info":
            if not name:
                return ToolResult("Error: 'name' parameter required", is_error=True)
            skill = mgr.get_skill(name)
            if not skill:
                return ToolResult(f"Error: Skill '{name}' not found.", is_error=True)
            is_pinned = name in mgr.get_pinned_skills()
            body = skill.get("body", "")
            lines = [
                f"📦 Skill: {skill['name']}",
                f"   Description: {skill['description']}",
                f"   Pinned: {'📌 Yes' if is_pinned else '❌ No'}",
                f"   Path: {skill.get('path', 'N/A')}",
                f"   Instructions ({len(body)} chars):",
                "   " + body[:500],
            ]
            if len(body) > 500:
                lines.append(f"   ... ({len(body) - 500} more chars)")
            return ToolResult("\n".join(lines))

        elif action == "create":
            if not name:
                return ToolResult("Error: 'name' parameter required", is_error=True)
            if not description:
                return ToolResult("Error: 'description' parameter required", is_error=True)
            if not instructions:
                return ToolResult("Error: 'instructions' parameter required", is_error=True)
            return ToolResult(mgr.create_skill(name, description, instructions))

        elif action == "delete":
            if not name:
                return ToolResult("Error: 'name' parameter required", is_error=True)
            return ToolResult(mgr.delete_skill(name))

        elif action == "resources":
            if not name:
                return ToolResult("Error: 'name' parameter required", is_error=True)
            # Distinguish "skill not found" from "skill has no resources":
            # list_resources() returns [] for both, so check existence first.
            if not mgr.get_skill(name):
                return ToolResult(f"Error: Skill '{name}' not found.", is_error=True)
            resources = mgr.list_resources(name)
            if not resources:
                return ToolResult(f"📦 Skill '{name}' has no resource files.")
            lines = [f"📦 Resources for '{name}':"]
            for r in resources:
                lines.append(f"  📄 {r}")
            return ToolResult("\n".join(lines))

        elif action == "read":
            if not name:
                return ToolResult("Error: 'name' parameter required", is_error=True)
            if not resource_path:
                return ToolResult("Error: 'resource_path' parameter required", is_error=True)
            content = mgr.read_resource(name, resource_path)
            if content is None:
                return ToolResult(f"Error: Resource '{resource_path}' not found in skill '{name}'.", is_error=True)
            return ToolResult(content)

        elif action == "reload":
            mgr.discover()
            catalog = mgr.get_catalog()
            return ToolResult(f"🔄 Skills reloaded. Found {len(catalog)} skills.")

        else:
            return ToolResult(
                f"Error: Unknown action '{action}'. Valid: list, load, pin, "
                f"deactivate, info, create, delete, resources, read, reload.",
                is_error=True,
            )

    except Exception as e:
        return ToolResult(f"Error managing skills: {e}", is_error=True)
