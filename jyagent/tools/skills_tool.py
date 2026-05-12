"""Tool entry point for Agent Skills (agentskills.io).

Routes action verbs (load, pin, list, deactivate, info, create, delete,
resources, read, reload) to the SkillManager in ``jyagent.skills``.  Not a
thin wrapper — verb routing, input validation, error envelopes, and
skill-body excerption all live here.

History: extracted from the former ``tools/facades.py`` (2026-05).
"""
from ..runtime.tools.result import ToolResult


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
            # Defensively escape closing wrapper tags inside the body so a
            # SKILL.md containing literal "</instructions>" or "</skill>"
            # can't break out of the wrapper. Shared with the pin-path —
            # see jyagent.skills.safe_skill_body.
            from ..skills import safe_skill_body
            body = safe_skill_body(body)
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
