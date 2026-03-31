# Facade tools: manage_memory, manage_skills
# These are thin wrappers that delegate to the memory/ and skills modules.

def manage_memory(action: str, text: str = "", category: str = "") -> str:
    """Manage the agent's self-use memory system. Actions: 'remember' (save a learning/fact), 'forget' (remove memories by keyword), 'show' (display all memories), 'topic' (manage topic files: list/read/write/delete), 'goal' (add/complete a goal), 'note' (add a working note). This tool lets you proactively remember things about the user for future sessions."""
    from ..memory.operations import (
        remember, forget, show_memory,
        list_topics, read_topic, write_topic, delete_topic,
    )

    try:
        if action == "remember":
            if not text:
                return "Error: 'text' parameter required for 'remember' action"
            return f"🧠 {remember(text, category)}"

        elif action == "forget":
            if not text:
                return "Error: 'text' parameter required for 'forget' action (keyword to match)"
            return f"🧠 {forget(text)}"

        elif action == "show":
            return show_memory()

        elif action == "topic":
            if not text:
                return "Error: 'text' parameter required. Formats: 'list', 'read:<name>', 'write:<name>|<content>', 'delete:<name>'"

            if text == "list":
                topics = list_topics()
                if not topics:
                    return "📂 No topic files yet. Create with topic action: 'write:<name>|<content>'"
                lines = []
                for t in topics:
                    tc = read_topic(t)
                    lines.append(f"  📄 {t}.md ({len(tc.split(chr(10)))} lines, {len(tc)} chars)")
                return "📂 Topic files (" + str(len(topics)) + "):\n" + "\n".join(lines)

            elif text.startswith("read:"):
                name = text[5:].strip()
                content = read_topic(name)
                if not content:
                    return f"Topic '{name}' not found. Available: {', '.join(list_topics()) or 'none'}"
                return f"📄 Topic: {name}.md" + "\n\n" + content

            elif text.startswith("write:"):
                rest = text[6:]
                if "|" not in rest:
                    return "Error: format is 'write:<name>|<content>'"
                name, content = rest.split("|", 1)
                name = name.strip()
                content = content.strip()
                write_topic(name, content)
                return f"📄 Topic '{name}.md' written ({len(content)} chars)"

            elif text.startswith("delete:"):
                name = text[7:].strip()
                if delete_topic(name):
                    return f"📄 Topic '{name}.md' deleted"
                return f"Topic '{name}' not found"

            else:
                return "Error: Unknown topic command. Use: 'list', 'read:<name>', 'write:<name>|<content>', 'delete:<name>'"

        elif action == "goal":
            if not text:
                return "Error: 'text' parameter required"
            if text.lower().startswith("done:"):
                return f"🧠 {forget(text[5:].strip())}"
            return f"🧠 {remember(text, 'goal')}"

        elif action == "note":
            if not text:
                return "Error: 'text' parameter required"
            return f"🧠 {remember(text, 'note')}"

        else:
            return f"Error: Unknown action '{action}'. Valid: remember, forget, show, topic, goal, note"

    except Exception as e:
        return f"Error managing memory: {e}"


def manage_skills(action: str, name: str = "", description: str = "",
                  instructions: str = "", resource_path: str = "") -> str:
    """Manage Agent Skills (agentskills.io). Actions: 'list' (show all skills), 'activate'/'deactivate' (control which skills are loaded into context), 'info' (show skill details), 'create' (create new skill), 'delete' (remove skill), 'resources' (list skill files), 'read' (read a skill resource file), 'reload' (re-scan skills directory)."""
    from ..skills import get_skill_manager

    try:
        mgr = get_skill_manager()

        if action == "list":
            catalog = mgr.get_catalog()
            if not catalog:
                return "📦 No skills found. Create one with manage_skills(action='create', ...)"
            lines = ["📦 Agent Skills:"]
            for entry in catalog:
                status = "✅ ACTIVE" if entry["active"] else "  📦"
                lines.append(f"  {status} {entry['name']}: {entry['description'][:100]}")
            lines.append(f"\n  Total: {len(catalog)} skills, {sum(1 for e in catalog if e['active'])} active")
            return "\n".join(lines)

        elif action == "activate":
            if not name:
                return "Error: 'name' parameter required"
            success = mgr.activate(name)
            if success:
                return f"✅ Skill '{name}' activated. Its instructions will be included in the system prompt."
            return f"Error: Skill '{name}' not found. Use manage_skills(action='list') to see available skills."

        elif action == "deactivate":
            if not name:
                return "Error: 'name' parameter required"
            success = mgr.deactivate(name)
            if success:
                return f"📦 Skill '{name}' deactivated."
            return f"Error: Skill '{name}' not found or not active."

        elif action == "info":
            if not name:
                return "Error: 'name' parameter required"
            skill = mgr.get_skill(name)
            if not skill:
                return f"Error: Skill '{name}' not found."
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
            return "\n".join(lines)

        elif action == "create":
            if not name:
                return "Error: 'name' parameter required"
            if not description:
                return "Error: 'description' parameter required"
            if not instructions:
                return "Error: 'instructions' parameter required"
            return mgr.create_skill(name, description, instructions)

        elif action == "delete":
            if not name:
                return "Error: 'name' parameter required"
            return mgr.delete_skill(name)

        elif action == "resources":
            if not name:
                return "Error: 'name' parameter required"
            resources = mgr.list_resources(name)
            if resources is None:
                return f"Error: Skill '{name}' not found."
            if not resources:
                return f"📦 Skill '{name}' has no resource files."
            lines = [f"📦 Resources for '{name}':"]
            for r in resources:
                lines.append(f"  📄 {r}")
            return "\n".join(lines)

        elif action == "read":
            if not name:
                return "Error: 'name' parameter required"
            if not resource_path:
                return "Error: 'resource_path' parameter required"
            content = mgr.read_resource(name, resource_path)
            if content is None:
                return f"Error: Resource '{resource_path}' not found in skill '{name}'."
            return content

        elif action == "reload":
            mgr.reload()
            catalog = mgr.get_catalog()
            return f"🔄 Skills reloaded. Found {len(catalog)} skills."

        else:
            return f"Error: Unknown action '{action}'. Valid: list, activate, deactivate, info, create, delete, resources, read, reload"

    except Exception as e:
        return f"Error managing skills: {e}"
