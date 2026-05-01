# Agent Skills engine — implements the Agent Skills open standard (agentskills.io)
#
# Provides:
#   - SkillManager: discovers, parses, and manages SKILL.md files
#   - Progressive disclosure: advertise → load → read resources
#   - System prompt injection: XML format per agentskills.io spec
#   - LLM-based auto-activation: model-assisted skill routing
#   - CLI integration: /skills command
#
# Directory layout:
#   skills/                    ← default skills root
#   ├── browser-automation/
#   │   ├── SKILL.md
#   │   ├── scripts/
#   │   └── references/
#   └── web-research/
#       └── SKILL.md

import os
import re
import glob
import html
import sys
from typing import Optional

from .config import (
    MAX_INSTRUCTIONS_CHARS,
    MAX_RESOURCE_CHARS,
)


# ─── Constants ────────────────────────────────────────────────────────────────

# Resolve to <repo_root>/skills.  This file lives at jyagent/skills.py, so
# we need TWO dirname calls: jyagent/ → <repo_root>.
#
# **Audit checklist when moving this file:** every `__file__`-relative path
# is depth-coupled to the file's location.  See data/memory/MEMORY.md
# "Durable Gotchas" — the reverse of this fix happened in 2026-04 when
# skills.py moved from jyagent/ → jyagent/runtime/ and silently returned
# zero skills until the dirname count was bumped to three.
DEFAULT_SKILLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "skills",
)
SKILL_FILENAME = "SKILL.md"

# Progressive disclosure budgets
MAX_CATALOG_TOKENS_PER_SKILL = 150   # ~name + description for advertising

# ─── SKILL.md Parser ─────────────────────────────────────────────────────────

def parse_skill_md(filepath: str) -> Optional[dict]:
    """
    Parse a SKILL.md file into a structured dict.
    
    Returns:
        {
            "name": str,
            "description": str,
            "license": str | None,
            "compatibility": str | None,
            "metadata": dict | None,
            "allowed_tools": list[str] | None,
            "body": str,                 # markdown body (instructions)
            "path": str,                 # absolute path to SKILL.md
            "skill_dir": str,            # parent directory of the skill
        }
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return None

    # Split YAML frontmatter from body
    frontmatter, body = _split_frontmatter(content)
    if frontmatter is None:
        return None

    # Parse YAML frontmatter (lightweight, no PyYAML dependency)
    meta = _parse_yaml_frontmatter(frontmatter)
    if not meta:
        return None

    # Validate required fields
    name = meta.get("name")
    description = meta.get("description")
    if not name or not description:
        return None

    # Validate name format: lowercase, hyphens, 1-64 chars
    if not re.match(r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?$', name) or len(name) > 64:
        return None
    if '--' in name:
        return None

    # Parse allowed-tools
    allowed_tools = None
    raw_tools = meta.get("allowed-tools") or meta.get("allowed_tools")
    if raw_tools and isinstance(raw_tools, str):
        allowed_tools = raw_tools.split()

    # Parse metadata
    metadata = meta.get("metadata")
    if metadata and not isinstance(metadata, dict):
        metadata = None

    skill_dir = os.path.dirname(os.path.abspath(filepath))

    return {
        "name": name,
        "description": description[:1024],
        "license": meta.get("license"),
        "compatibility": meta.get("compatibility"),
        "metadata": metadata,
        "allowed_tools": allowed_tools,
        "body": body.strip(),
        "path": os.path.abspath(filepath),
        "skill_dir": skill_dir,
    }


def _split_frontmatter(content: str) -> tuple:
    """Split YAML frontmatter (between ---) from markdown body."""
    content = content.lstrip()
    if not content.startswith('---'):
        return None, content

    # Find the closing ---
    end_match = re.search(r'\n---\s*\n', content[3:])
    if not end_match:
        # Try end of string
        end_match = re.search(r'\n---\s*$', content[3:])
        if not end_match:
            return None, content

    fm_start = 3  # skip opening ---
    fm_end = fm_start + end_match.start()
    frontmatter = content[fm_start:fm_end].strip()
    body = content[fm_start + end_match.end():].strip()

    return frontmatter, body


def _parse_yaml_frontmatter(text: str) -> dict:
    """
    Lightweight YAML parser for SKILL.md frontmatter.
    Handles:
      - Flat key-value pairs: key: value
      - Block scalars: key: >- (with indented continuation lines joined as string)
      - Nested maps: key:\n  subkey: value (collected as dict)
      - Quoted strings
    No external dependencies required.
    """
    result = {}
    lines = text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty/comment lines
        if not stripped or stripped.startswith('#'):
            i += 1
            continue

        # Must be a top-level key: value line
        kv = _parse_yaml_kv(stripped)
        if not kv:
            i += 1
            continue

        key, value = kv
        top_indent = len(line) - len(line.lstrip())

        # Case 1: Block scalar (>-, >, |, |-)
        if value in ('>-', '>', '|', '|-'):
            fold = value.startswith('>')
            chomp = value.endswith('-')
            # Collect all indented continuation lines
            block_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                # Empty line within block
                if not next_line.strip():
                    block_lines.append('')
                    i += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= top_indent:
                    break  # Back to top-level or less indented
                block_lines.append(next_line.strip())
                i += 1

            # Remove trailing empty lines if chomp
            if chomp:
                while block_lines and block_lines[-1] == '':
                    block_lines.pop()

            if fold:
                # >  or >- : fold newlines into spaces
                result[key] = ' '.join(line for line in block_lines if line)
            else:
                # |  or |- : preserve newlines
                result[key] = '\n'.join(block_lines)
            continue

        # Case 2: Empty value — might be a nested map
        if value == '' or value is None:
            nested = {}
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if not next_line.strip():
                    i += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= top_indent:
                    break  # Back to top-level
                sub_kv = _parse_yaml_kv(next_line.strip())
                if sub_kv:
                    nested[sub_kv[0]] = sub_kv[1]
                i += 1

            result[key] = nested if nested else ''
            continue

        # Case 3: Simple key: value
        result[key] = value
        i += 1

    return result


def _parse_yaml_kv(line: str) -> Optional[tuple]:
    """Parse a single YAML key: value line."""
    match = re.match(r'^([a-zA-Z_-]+)\s*:\s*(.*)', line)
    if not match:
        return None

    key = match.group(1).strip()
    value = match.group(2).strip()

    # Handle multiline string indicators — return them as-is so caller can handle
    if value in ('>-', '>', '|', '|-'):
        return (key, value)

    # Strip quotes
    if len(value) >= 2:
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

    return (key, value)


# ─── Skill Manager ───────────────────────────────────────────────────────────

class SkillManager:
    """
    Discovers, manages, and provides progressive disclosure for Agent Skills.

    Two ways a skill enters context:
      * **load** (one-shot) — facade returns the SKILL.md body as a tool
        result so it lives in conversation history exactly once. The manager
        does NOT track load events; that channel is stateless.
      * **pin** (session-long) — `pin(name)` adds the skill to `_pinned`.
        `build_pinned_bodies_block()` then re-emits its body as a tail block
        on every turn until `unpin(name)` (or `unpin_all()`).

    Usage:
        manager = SkillManager()
        manager.discover()                        # scan skills/ directory
        catalog = manager.get_catalog()           # advertise: name + description
        manager.pin("web-search")                 # session-long pin
        bodies = manager.build_pinned_bodies_block()  # injected per-turn

    Back-compat: the old method names ``activate`` / ``deactivate`` /
    ``deactivate_all`` / ``get_active_skills`` / ``build_active_bodies_block``
    are kept as deprecated thin aliases pointing at the new names.
    """

    def __init__(self, skills_dir: str = DEFAULT_SKILLS_DIR):
        self.skills_dir = os.path.abspath(skills_dir)
        self._skills: dict[str, dict] = {}       # name → parsed skill
        self._pinned: set[str] = set()           # session-pinned skill names
        # NOTE: there is NO automatic skill router.  The model sees the
        # catalog (Stage 1) in the system prompt and self-disposes via
        # ``manage_skills(action='load'|'pin', ...)``.  Load is one-shot
        # (no manager state); pin is session-long (tracked in ``_pinned``).
        # Eval tooling that needs "would query X trigger skill Y?" is in
        # skills/create-skill/scripts/test_trigger.py, which inlines its
        # own one-shot router rather than coupling to this class.

    def discover(self) -> list[str]:
        """
        Scan the skills directory for SKILL.md files.
        Returns list of discovered skill names.
        """
        self._skills.clear()
        # Drop stale pins from skills that no longer exist (e.g. after
        # delete/rename). Re-population happens below; any names that
        # remain pinned and still parse correctly stay pinned.
        prev_pinned = set(self._pinned)
        self._pinned.clear()
        discovered = []

        if not os.path.isdir(self.skills_dir):
            return discovered

        # Pattern 1: skills/<name>/SKILL.md
        for skill_md in glob.glob(os.path.join(self.skills_dir, "*", SKILL_FILENAME)):
            skill = parse_skill_md(skill_md)
            if skill:
                self._skills[skill["name"]] = skill
                discovered.append(skill["name"])

        # Pattern 2: skills/SKILL.md (single-skill, no subdirectory)
        root_skill = os.path.join(self.skills_dir, SKILL_FILENAME)
        if os.path.isfile(root_skill):
            skill = parse_skill_md(root_skill)
            if skill and skill["name"] not in self._skills:
                self._skills[skill["name"]] = skill
                discovered.append(skill["name"])

        # Restore pins for skills that survived the rediscovery.
        for n in prev_pinned:
            if n in self._skills:
                self._pinned.add(n)

        return discovered

    def get_skill(self, name: str) -> Optional[dict]:
        """Get a parsed skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """List all discovered skill names."""
        return sorted(self._skills.keys())

    def get_catalog(self) -> list[dict]:
        """
        Stage 1: Advertise — return lightweight catalog for system prompt injection.
        Each entry is ~100 tokens (name + description only).
        """
        catalog = []
        for name in sorted(self._skills.keys()):
            skill = self._skills[name]
            catalog.append({
                "name": skill["name"],
                "description": skill["description"],
                "pinned": name in self._pinned,
            })
        return catalog

    def pin(self, name: str) -> bool:
        """
        Pin a skill for the whole session: its full SKILL.md body will be
        prepended to every user message via ``build_pinned_bodies_block()``
        until ``unpin(name)`` (or ``unpin_all()``).
        """
        if name in self._skills:
            self._pinned.add(name)
            return True
        return False

    def unpin(self, name: str) -> bool:
        """Stop session-pinning a skill."""
        if name in self._pinned:
            self._pinned.discard(name)
            return True
        return False

    def unpin_all(self) -> None:
        """Un-pin every currently-pinned skill."""
        self._pinned.clear()

    def get_pinned_skills(self) -> list[str]:
        """Return names of currently pinned skills."""
        return sorted(self._pinned)

    # ── Deprecated aliases (kept for back-compat with external callers) ──
    # New code should use pin/unpin/unpin_all/get_pinned_skills directly.
    # These thin shims preserve the old naming used by /skill CLI handler
    # historical and any saved sessions/tests written against the prior API.

    def activate(self, name: str) -> bool:
        """DEPRECATED alias of ``pin``."""
        return self.pin(name)

    def deactivate(self, name: str) -> bool:
        """DEPRECATED alias of ``unpin``."""
        return self.unpin(name)

    def deactivate_all(self) -> None:
        """DEPRECATED alias of ``unpin_all``."""
        self.unpin_all()

    def get_active_skills(self) -> list[str]:
        """DEPRECATED alias of ``get_pinned_skills``."""
        return self.get_pinned_skills()

    def get_instructions(self, name: str) -> Optional[str]:
        """
        Get the full SKILL.md body (instructions) for a skill.
        Returns None if skill not found.
        """
        skill = self._skills.get(name)
        if not skill:
            return None
        body = skill.get("body", "")
        return body[:MAX_INSTRUCTIONS_CHARS]

    def read_resource(self, name: str, relative_path: str) -> Optional[str]:
        """
        Stage 3: Read — load a resource file (scripts/, references/, assets/).
        Returns file content or None.
        """
        skill = self._skills.get(name)
        if not skill:
            return None

        skill_dir = os.path.realpath(skill["skill_dir"])
        # Security: prevent path traversal and symlink escapes
        resource_path = os.path.realpath(os.path.join(skill_dir, relative_path))
        try:
            if os.path.commonpath([skill_dir, resource_path]) != skill_dir:
                return None
        except ValueError:
            return None

        try:
            with open(resource_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content[:MAX_RESOURCE_CHARS]
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            return None

    def list_resources(self, name: str) -> list[str]:
        """List available resource files for a skill."""
        skill = self._skills.get(name)
        if not skill:
            return []

        skill_dir = skill["skill_dir"]
        resources = []
        for subdir in ("scripts", "references", "assets"):
            subdir_path = os.path.join(skill_dir, subdir)
            if os.path.isdir(subdir_path):
                for root, dirs, files in os.walk(subdir_path):
                    for f in files:
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, skill_dir)
                        resources.append(rel)
        return sorted(resources)

    # ── 改进1: XML prompt format (agentskills.io spec) ────────────────────

    def build_catalog_block(self) -> str:
        """
        Stage 1 — return the lightweight skill catalog as XML.

        Stable across turns (only changes when the on-disk skills/ directory
        changes), so this is safe to inject into the SYSTEM PROMPT where it
        will be cached by Anthropic's prompt cache. Active-state is
        intentionally omitted from this block so per-turn activation diffs
        do NOT invalidate the prefix cache.
        """
        if not self._skills:
            return ""

        lines = ["<available_skills>"]
        lines.append(
            "Use manage_skills(action='load', name=X) to bring skill X's full "
            "instructions into context as a tool result. Do NOT call load again "
            "for the same skill if its instructions are already visible above. "
            "Use action='pin' instead only when the user explicitly asks to "
            "keep a skill on for the whole session. Users can also pin via /skill <name>."
        )
        for name in sorted(self._skills.keys()):
            skill = self._skills[name]
            lines.append("<skill>")
            lines.append(f"<name>{html.escape(name)}</name>")
            # Description is already capped to 1024 chars at parse time
            # (agentskills.io spec §description). Don't re-truncate here — the
            # description IS the discovery signal for progressive disclosure,
            # and a 200-char cap was clipping trigger keywords mid-sentence.
            lines.append(f"<description>{html.escape(skill['description'])}</description>")
            lines.append("</skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def build_pinned_bodies_block(self) -> str:
        """
        Stage 2 — return full instruction bodies for currently PINNED skills.

        Empty string when no skill is pinned. Caller should attach this as a
        TAIL message block (e.g. prepend to the last user message's content)
        rather than concatenating into the system prompt — that way pin
        changes do not invalidate the system-prompt prefix cache.

        Note: skills brought in via ``manage_skills(action='load', ...)`` do
        NOT appear here — those return their body as a tool result and live
        in conversation history, not in the per-turn pin block.
        """
        if not self._pinned:
            return ""

        lines = []
        for name in sorted(self._pinned):
            skill = self._skills.get(name)
            if not skill:
                continue
            body = skill.get("body", "")
            if not body:
                continue

            if lines:
                lines.append("")
            lines.append(f'<pinned_skill name="{html.escape(name)}">')

            if skill.get("allowed_tools"):
                tools_str = ", ".join(skill["allowed_tools"])
                lines.append(f"<allowed_tools>{html.escape(tools_str)}</allowed_tools>")

            resources = self.list_resources(name)
            if resources:
                lines.append(f"<resources>{html.escape(', '.join(resources))}</resources>")

            lines.append("<instructions>")
            lines.append(body[:MAX_INSTRUCTIONS_CHARS])
            lines.append("</instructions>")
            lines.append("</pinned_skill>")

        return "\n".join(lines)

    def build_active_bodies_block(self) -> str:
        """DEPRECATED alias of ``build_pinned_bodies_block``."""
        return self.build_pinned_bodies_block()

    def build_prompt_context(self) -> str:
        """
        Legacy combined renderer: catalog + pinned bodies in one string.

        Kept for back-compat with callers that inject the whole thing into a
        single buffer (e.g. tests, ad-hoc tooling). The main agent loop now
        uses ``build_catalog_block`` and ``build_pinned_bodies_block``
        separately so they can be placed in cache-friendly positions.
        """
        if not self._skills:
            return ""

        catalog = self.build_catalog_block()
        bodies = self.build_pinned_bodies_block()
        if catalog and bodies:
            return catalog + "\n\n" + bodies
        return catalog or bodies

    def create_skill(self, name: str, description: str, instructions: str,
                     metadata: Optional[dict] = None) -> str:
        """
        Create a new skill directory with SKILL.md.
        Returns success message or error.
        """
        # Validate name
        if not re.match(r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?$', name) or len(name) > 64:
            return f"Error: Invalid skill name '{name}'. Must be lowercase alphanumeric with hyphens, 1-64 chars."
        if '--' in name:
            return "Error: Skill name must not contain consecutive hyphens."
        if name in self._skills:
            return f"Error: Skill '{name}' already exists."

        # Create directory structure
        skill_dir = os.path.join(self.skills_dir, name)
        os.makedirs(skill_dir, exist_ok=True)
        os.makedirs(os.path.join(skill_dir, "scripts"), exist_ok=True)
        os.makedirs(os.path.join(skill_dir, "references"), exist_ok=True)
        os.makedirs(os.path.join(skill_dir, "assets"), exist_ok=True)

        # Build SKILL.md
        fm_lines = [
            "---",
            f"name: {name}",
            "description: >-",
            f"  {description}",
        ]
        if metadata:
            fm_lines.append("metadata:")
            for k, v in metadata.items():
                fm_lines.append(f"  {k}: \"{v}\"")
        fm_lines.append("---")
        fm_lines.append("")
        fm_lines.append(instructions)

        skill_md_path = os.path.join(skill_dir, SKILL_FILENAME)
        with open(skill_md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(fm_lines))

        # Re-discover to pick up the new skill
        self.discover()

        return f"✅ Skill '{name}' created at {skill_dir}/"

    def delete_skill(self, name: str) -> str:
        """Delete a skill directory."""
        if name not in self._skills:
            return f"Error: Skill '{name}' not found."

        import shutil
        skill_dir = self._skills[name]["skill_dir"]
        try:
            shutil.rmtree(skill_dir)
            self._pinned.discard(name)
            del self._skills[name]
            return f"✅ Skill '{name}' deleted."
        except Exception as e:
            return f"Error deleting skill: {e}"


# ─── Singleton ────────────────────────────────────────────────────────────────

_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    """Get or create the global SkillManager singleton."""
    global _manager
    if _manager is None:
        _manager = SkillManager()
        _manager.discover()
    return _manager


def init_skills(skills_dir: str = DEFAULT_SKILLS_DIR) -> SkillManager:
    """Initialize (or reinitialize) the skill manager with a given directory."""
    global _manager
    _manager = SkillManager(skills_dir)
    _manager.discover()
    return _manager
