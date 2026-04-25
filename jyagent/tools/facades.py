# Facade tools: manage_memory, manage_skills
# These are thin wrappers that delegate to the memory/ and skills modules.

from ..runtime.tools.result import ToolResult


def manage_memory(action: str, text: str = "", category: str = "") -> ToolResult:
    """Manage the agent's self-use memory system. Actions: 'remember' (save a DURABLE learning/fact to MEMORY.md — use sparingly, data-independent rules only), 'forget' (remove memories by keyword), 'show' (display all memories), 'topic' (manage curated topic files: list/read/write/delete), 'goal' (add/complete a goal), 'note' (DEPRECATED alias for journal), 'journal' (append a dated session note to data/memory/journal/YYYY-MM.md — never auto-loaded, for 'what I did today' style entries), 'consolidate' (analyze MEMORY.md for dedup / bloat candidates — read-only). Three tiers: always-loaded index (MEMORY.md) / curated on-demand (topics/) / chronological on-demand (journal/)."""
    from ..memory.operations import (
        remember, forget, show_memory,
        list_topics, read_topic, write_topic, delete_topic,
        append_journal, list_journals, read_journal, consolidate_memory,
    )

    try:
        if action == "remember":
            if not text:
                return ToolResult("Error: 'text' parameter required for 'remember' action", is_error=True)
            return ToolResult(f"🧠 {remember(text, category)}")

        elif action == "forget":
            if not text:
                return ToolResult("Error: 'text' parameter required for 'forget' action (keyword to match)", is_error=True)
            return ToolResult(f"🧠 {forget(text)}")

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
                return ToolResult("Error: 'text' parameter required. Formats: 'list', 'read:<name>', 'write:<name>|<content>', 'delete:<name>'", is_error=True)

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
                name = text[5:].strip()
                content = read_topic(name)
                if not content:
                    return ToolResult(f"Topic '{name}' not found. Available: {', '.join(list_topics()) or 'none'}", is_error=True)
                return ToolResult(f"📄 Topic: {name}.md" + "\n\n" + content)

            elif text.startswith("write:"):
                rest = text[6:]
                if "|" not in rest:
                    return ToolResult("Error: format is 'write:<name>|<content>'", is_error=True)
                name, content = rest.split("|", 1)
                name = name.strip()
                content = content.strip()
                write_topic(name, content)
                return ToolResult(f"📄 Topic '{name}.md' written ({len(content)} chars)")

            elif text.startswith("delete:"):
                name = text[7:].strip()
                if delete_topic(name):
                    return ToolResult(f"📄 Topic '{name}.md' deleted")
                return ToolResult(f"Topic '{name}' not found", is_error=True)

            else:
                return ToolResult("Error: Unknown topic command. Use: 'list', 'read:<name>', 'write:<name>|<content>', 'delete:<name>'", is_error=True)

        elif action == "goal":
            if not text:
                return ToolResult("Error: 'text' parameter required", is_error=True)
            if text.lower().startswith("done:"):
                return ToolResult(f"🧠 {forget(text[5:].strip())}")
            return ToolResult(f"🧠 {remember(text, 'goal')}")

        elif action == "note":
            # 'note' used to mean "append [note] to MEMORY.md" — that was an
            # anti-pattern (chronological cruft in the always-loaded index, with
            # prompt-cache invalidation cost). Redirect to the journal tier and
            # tell the caller. Old call sites still work.
            if not text:
                return ToolResult("Error: 'text' parameter required", is_error=True)
            cat = category or "note"
            path = append_journal(text, cat)
            return ToolResult(
                f"📓 Note routed to journal: {path} [{cat}] "
                "(action='note' is now an alias for action='journal'; "
                "use action='remember' only for durable rules in MEMORY.md)"
            )

        else:
            return ToolResult(f"Error: Unknown action '{action}'. Valid: remember, forget, show, topic, goal, note, journal, consolidate", is_error=True)

    except Exception as e:
        return ToolResult(f"Error managing memory: {e}", is_error=True)


def manage_skills(action: str, name: str = "", description: str = "",
                  instructions: str = "", resource_path: str = "") -> ToolResult:
    """Manage Agent Skills (agentskills.io). Actions: 'list' (show all skills), 'activate'/'deactivate' (control which skills are loaded into context), 'info' (show skill details), 'create' (create new skill), 'delete' (remove skill), 'resources' (list skill files), 'read' (read a skill resource file), 'reload' (re-scan skills directory)."""
    from ..skills import get_skill_manager

    try:
        mgr = get_skill_manager()

        if action == "list":
            catalog = mgr.get_catalog()
            if not catalog:
                return ToolResult("📦 No skills found. Create one with manage_skills(action='create', ...)")
            lines = ["📦 Agent Skills:"]
            for entry in catalog:
                status = "✅ ACTIVE" if entry["active"] else "  📦"
                lines.append(f"  {status} {entry['name']}: {entry['description'][:100]}")
            lines.append(f"\n  Total: {len(catalog)} skills, {sum(1 for e in catalog if e['active'])} active")
            return ToolResult("\n".join(lines))

        elif action == "activate":
            if not name:
                return ToolResult("Error: 'name' parameter required", is_error=True)
            success = mgr.activate(name)
            if success:
                return ToolResult(f"✅ Skill '{name}' activated. Its instructions will be included in the system prompt.")
            return ToolResult(f"Error: Skill '{name}' not found. Use manage_skills(action='list') to see available skills.", is_error=True)

        elif action == "deactivate":
            if not name:
                return ToolResult("Error: 'name' parameter required", is_error=True)
            success = mgr.deactivate(name)
            if success:
                return ToolResult(f"📦 Skill '{name}' deactivated.")
            return ToolResult(f"Error: Skill '{name}' not found or not active.", is_error=True)

        elif action == "info":
            if not name:
                return ToolResult("Error: 'name' parameter required", is_error=True)
            skill = mgr.get_skill(name)
            if not skill:
                return ToolResult(f"Error: Skill '{name}' not found.", is_error=True)
            is_active = name in mgr.get_active_skills()
            body = skill.get("body", "")
            lines = [
                f"📦 Skill: {skill['name']}",
                f"   Description: {skill['description']}",
                f"   Active: {'✅ Yes' if is_active else '❌ No'}",
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
            resources = mgr.list_resources(name)
            if resources is None:
                return ToolResult(f"Error: Skill '{name}' not found.", is_error=True)
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
            return ToolResult(f"Error: Unknown action '{action}'. Valid: list, activate, deactivate, info, create, delete, resources, read, reload", is_error=True)

    except Exception as e:
        return ToolResult(f"Error managing skills: {e}", is_error=True)
