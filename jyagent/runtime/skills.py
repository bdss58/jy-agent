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
import json
import sys
import time
from typing import Optional

from ..config import (
    MAX_INSTRUCTIONS_CHARS,
    MAX_RESOURCE_CHARS,
    SKILL_ROUTER_TIMEOUT,
    get_skill_router_model_spec,
)


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills")
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
    
    Usage:
        manager = SkillManager()
        manager.discover()                        # scan skills/ directory
        catalog = manager.get_catalog()           # advertise: name + description
        manager.activate("web-research")          # load: full instructions
        context = manager.build_prompt_context()  # inject into system prompt
    """

    def __init__(self, skills_dir: str = DEFAULT_SKILLS_DIR):
        self.skills_dir = os.path.abspath(skills_dir)
        self._skills: dict[str, dict] = {}       # name → parsed skill
        self._active: set[str] = set()            # currently loaded skill names
        self._auto_activate: bool = True          # auto-activate on relevant query

    def discover(self) -> list[str]:
        """
        Scan the skills directory for SKILL.md files.
        Returns list of discovered skill names.
        """
        self._skills.clear()
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
                "active": name in self._active,
            })
        return catalog

    def activate(self, name: str) -> bool:
        """
        Stage 2: Load — mark a skill as active so its full instructions
        are included in the system prompt.
        """
        if name in self._skills:
            self._active.add(name)
            return True
        return False

    def deactivate(self, name: str) -> bool:
        """Deactivate a skill (stop injecting its instructions)."""
        if name in self._active:
            self._active.discard(name)
            return True
        return False

    def deactivate_all(self):
        """Deactivate all skills."""
        self._active.clear()

    def get_active_skills(self) -> list[str]:
        """Return names of currently active skills."""
        return sorted(self._active)

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

    # ── Diff-based skill routing (per-turn re-evaluation) ──────────────────

    def auto_activate_for_query(self, query: str, runtime_owner=None,
                                recent_messages: list | None = None) -> list[str]:
        """
        Re-evaluate which skills should be active for this turn.

        Unlike the old additive approach, this considers the FULL catalog
        every turn and produces a diff: skills can be added AND removed.
        Recent conversation messages give the router context for follow-up
        queries like "test it" or "now fix the description".

        Strategy:
          1. Try LLM diff-based routing (full catalog + recent history)
          2. Fallback to keyword matching on full catalog if LLM fails
          3. On fallback failure, keep current active set unchanged

        Returns list of skill names in the new active set (not just newly added).
        """
        if not self._auto_activate:
            return list(self._active)

        if not self._skills:
            return []

        prev_active = set(self._active)

        # Try LLM routing first (evaluates full catalog)
        result = self._route_llm(
            query, recent_messages=recent_messages, runtime_owner=runtime_owner,
        )
        if result is not None:
            return result

        # Fallback: keyword matching on full catalog
        result = self._route_keywords(query)
        if result is not None:
            return result

        # Both failed — keep current set unchanged (graceful degrade)
        return list(self._active)

    def _route_llm(self, query: str, *, recent_messages: list | None = None,
                   runtime_owner=None) -> Optional[list[str]]:
        """
        LLM diff-based router: decide the complete active skill set for this turn.

        Sends the full skill catalog, currently active set, recent conversation
        context, and current query. Returns the new active set, or None on failure.
        """
        if runtime_owner is None:
            try:
                from ..llm import LLMOwner
                runtime_owner = LLMOwner(get_skill_router_model_spec())
            except Exception:
                return None

        # Build skill catalog
        catalog_lines = []
        for name, skill in sorted(self._skills.items()):
            catalog_lines.append(f"- {name}: {skill['description'][:200]}")
        catalog_text = "\n".join(catalog_lines)

        # Build recent conversation context (prior turns for multi-turn continuity)
        # Caller already filters to user/assistant and excludes current query.
        history_text = ""
        if recent_messages:
            history_parts = []
            for msg in recent_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text from content blocks
                    content = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if content and role in ("user", "assistant"):
                    content = content[:200]  # truncate for token budget
                    history_parts.append(f"  {role}: {content}")
            if history_parts:
                history_text = "Recent conversation:\n" + "\n".join(history_parts) + "\n\n"

        # Current active set (so router knows what's loaded)
        active_text = ""
        if self._active:
            active_text = f"Currently active skills: {sorted(self._active)}\n\n"

        routing_prompt = (
            "You are a skill router. Given a user query, conversation context, "
            "and available skills, decide which skills should be active for this turn.\n\n"
            "Skills can be ADDED (newly relevant) or REMOVED (no longer needed). "
            "Evaluate the FULL catalog — don't just keep everything active.\n\n"
            f"Available skills:\n{catalog_text}\n\n"
            f"{active_text}"
            f"{history_text}"
            f"Current user query: {query}\n\n"
            "Return a JSON array of skill names that should be active for this turn. "
            "Return [] if no skills are needed. "
            "ONLY output the JSON array, nothing else."
        )

        try:
            t0 = time.time()
            text = runtime_owner.complete_text(
                routing_prompt,
                max_output_tokens=100,
                model_spec=get_skill_router_model_spec(runtime_owner.model_spec),
                timeout=SKILL_ROUTER_TIMEOUT,
                # Router is a cheap utility call — no extended thinking.
                # This also avoids validate_anthropic_reasoning rejecting
                # models < Claude 4.6 when adaptive thinking is configured
                # via env for the main agent.
                reasoning=None,
            )
            elapsed = time.time() - t0

            # Parse response
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
                text = text.strip()

            selected = json.loads(text)
            if not isinstance(selected, list):
                return None

            # Validate: only accept known skill names
            new_active = set()
            for name in selected:
                if isinstance(name, str) and name in self._skills:
                    new_active.add(name)

            # Compute diff for logging
            prev_active = set(self._active)
            added = new_active - prev_active
            removed = prev_active - new_active

            # Apply the new active set
            self._active = new_active

            # Log the diff
            parts = []
            if added:
                parts.append(f"+{sorted(added)}")
            if removed:
                parts.append(f"-{sorted(removed)}")
            if not parts:
                parts.append("no change")
            diff_str = " ".join(parts)

            print(f"\033[2m  ⚡ Skill router ({elapsed:.1f}s): "
                  f"{sorted(new_active)} ({diff_str})\033[0m", file=sys.stderr)

            return sorted(new_active)

        except Exception as e:
            # Surface router failures so silent keyword fallback isn't
            # mistaken for "router is working". First-line-only, dim, to
            # stderr so the UX stays calm but debuggable.
            msg = str(e).split("\n", 1)[0][:240]
            print(f"\033[2m  ⚡ Skill router failed ({type(e).__name__}): "
                  f"{msg}\033[0m", file=sys.stderr)
            return None

    def _route_keywords(self, query: str) -> Optional[list[str]]:
        """
        Fallback: keyword matching on the full catalog.

        Unlike the old version, this evaluates ALL skills (not just inactive)
        and replaces the active set entirely — same diff semantics as LLM router.
        """
        query_lower = query.lower()

        stopwords = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in',
                    'and', 'or', 'on', 'it', 'this', 'that', 'use', 'when',
                    'with', 'from', 'by', 'at', 'be', 'as', 'do', 'if', 'so',
                    'what', 'how', 'why', 'can', 'will', 'my', 'your', 'me'}
        query_words = set(re.findall(r'[a-z]{2,}', query_lower))

        # If query is very short (≤3 words), keyword matching is unreliable —
        # keep current active set unchanged rather than guess wrong
        if len(query_words) <= 3 and self._active:
            print(f"\033[2m  ⚡ Skill keywords: query too short, keeping "
                  f"{sorted(self._active)}\033[0m", file=sys.stderr)
            return list(self._active)

        new_active = set()
        for name, skill in self._skills.items():
            desc_lower = skill["description"].lower()
            desc_words = set(re.findall(r'[a-z]{2,}', desc_lower))
            meaningful_overlap = (desc_words & query_words) - stopwords

            name_words = set(name.replace('-', ' ').split())
            name_overlap = name_words & query_words

            if len(meaningful_overlap) >= 2 or len(name_overlap) >= 1:
                new_active.add(name)

        prev_active = set(self._active)
        added = new_active - prev_active
        removed = prev_active - new_active
        self._active = new_active

        parts = []
        if added:
            parts.append(f"+{sorted(added)}")
        if removed:
            parts.append(f"-{sorted(removed)}")
        if not parts:
            parts.append("no change")
        diff_str = " ".join(parts)

        print(f"\033[2m  ⚡ Skill keywords: "
              f"{sorted(new_active)} ({diff_str})\033[0m", file=sys.stderr)

        return sorted(new_active)

    # ── 改进1: XML prompt format (agentskills.io spec) ────────────────────

    def build_prompt_context(self, query: str = "", runtime_owner=None,
                             recent_messages: list | None = None) -> str:
        """
        Build the skills context section for system prompt injection.
        
        Uses the official agentskills.io XML format:
        - <available_skills>: catalog of all skills (always included)
        - <active_skill>: full instructions for activated skills
        
        Implements progressive disclosure:
        - Stage 1 (advertise): name + description in <skill> elements ~100 tokens/skill
        - Stage 2 (load): full SKILL.md body in <active_skill> elements
        """
        if not self._skills:
            return ""

        # Route skills for this turn (diff-based: can add AND remove)
        if query:
            self.auto_activate_for_query(
                query, runtime_owner=runtime_owner,
                recent_messages=recent_messages,
            )

        lines = []

        # ── Stage 1: Catalog — <available_skills> XML ──
        lines.append("<available_skills>")
        lines.append("Use /skill <name> to manually activate, or they auto-activate on relevant queries.")
        for name in sorted(self._skills.keys()):
            skill = self._skills[name]
            active_attr = ' status="active"' if name in self._active else ''
            lines.append(f"<skill{active_attr}>")
            lines.append(f"<name>{html.escape(name)}</name>")
            lines.append(f"<description>{html.escape(skill['description'][:200])}</description>")
            lines.append("</skill>")
        lines.append("</available_skills>")

        # ── Stage 2: Active skill instructions ──
        for name in sorted(self._active):
            skill = self._skills.get(name)
            if not skill:
                continue

            body = skill.get("body", "")
            if not body:
                continue

            lines.append("")
            lines.append(f'<active_skill name="{html.escape(name)}">')

            # Optional: allowed tools
            if skill.get("allowed_tools"):
                tools_str = ", ".join(skill["allowed_tools"])
                lines.append(f"<allowed_tools>{html.escape(tools_str)}</allowed_tools>")

            # Optional: available resources
            resources = self.list_resources(name)
            if resources:
                lines.append(f"<resources>{html.escape(', '.join(resources))}</resources>")

            lines.append("<instructions>")
            lines.append(body[:MAX_INSTRUCTIONS_CHARS])
            lines.append("</instructions>")
            lines.append("</active_skill>")

        return "\n".join(lines)

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
            self._active.discard(name)
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
