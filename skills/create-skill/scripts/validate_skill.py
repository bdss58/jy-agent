#!/usr/bin/env python3
"""Validate a SKILL.md file against the Agent Skills specification.

Usage:
    python scripts/validate_skill.py <skill-directory>
    python scripts/validate_skill.py --help

Checks:
  - SKILL.md exists and is parseable
  - name field: format, length, matches directory name
  - description field: length, has trigger guidance
  - Body: line count, has decision tree or structured sections
  - Directory structure: standard subdirs present
  - Resources referenced in body actually exist
"""

import os
import re
import sys
from pathlib import Path


def _split_frontmatter(content: str):
    """Split YAML frontmatter from markdown body."""
    content = content.lstrip()
    if not content.startswith('---'):
        return None, content
    end = re.search(r'\n---\s*\n', content[3:])
    if not end:
        end = re.search(r'\n---\s*$', content[3:])
        if not end:
            return None, content
    fm = content[3:3 + end.start()].strip()
    body = content[3 + end.end():].strip()
    return fm, body


def _parse_yaml_value(text: str) -> str:
    """Strip quotes from a YAML value."""
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text


def _parse_simple_yaml(text: str) -> dict:
    """Parse flat key-value YAML (enough for frontmatter validation)."""
    result = {}
    current_key = None
    current_value_lines = []

    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        indent = len(line) - len(line.lstrip())

        # Continuation line for block scalar:
        # If we're collecting a block value and this line is indented,
        # it's a continuation — even if it contains ':' (like "TRIGGER on:")
        if current_key and indent > 0:
            current_value_lines.append(stripped)
            continue

        # Save previous key
        if current_key and current_value_lines:
            result[current_key] = ' '.join(current_value_lines)
            current_key = None
            current_value_lines = []

        if ':' in stripped:
            key, _, val = stripped.partition(':')
            key = key.strip()
            val = val.strip()
            if val in ('>-', '>', '|', '|-', ''):
                current_key = key
                current_value_lines = []
            else:
                result[key] = _parse_yaml_value(val)

    if current_key and current_value_lines:
        result[current_key] = ' '.join(current_value_lines)

    return result


class ValidationResult:
    def __init__(self):
        self.errors = []    # Fatal — skill won't work
        self.warnings = []  # Quality issues
        self.info = []      # Suggestions

    def error(self, msg):
        self.errors.append(f"❌ {msg}")

    def warn(self, msg):
        self.warnings.append(f"⚠️  {msg}")

    def note(self, msg):
        self.info.append(f"💡 {msg}")

    @property
    def passed(self):
        return len(self.errors) == 0

    def summary(self):
        lines = []
        if self.errors:
            lines.append("ERRORS:")
            lines.extend(f"  {e}" for e in self.errors)
        if self.warnings:
            lines.append("WARNINGS:")
            lines.extend(f"  {w}" for w in self.warnings)
        if self.info:
            lines.append("SUGGESTIONS:")
            lines.extend(f"  {i}" for i in self.info)
        if self.passed:
            lines.insert(0, f"✅ Validation passed ({len(self.warnings)} warnings, {len(self.info)} suggestions)")
        else:
            lines.insert(0, f"❌ Validation FAILED ({len(self.errors)} errors, {len(self.warnings)} warnings)")
        return "\n".join(lines)


def validate_skill(skill_dir: str) -> ValidationResult:
    """Validate a skill directory. Returns ValidationResult."""
    result = ValidationResult()
    skill_path = Path(skill_dir).resolve()
    dir_name = skill_path.name

    # --- Check SKILL.md exists ---
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        result.error(f"SKILL.md not found in {skill_path}")
        return result

    content = skill_md.read_text(encoding='utf-8')

    # --- Parse frontmatter ---
    fm_text, body = _split_frontmatter(content)
    if fm_text is None:
        result.error("No YAML frontmatter found (must start with ---)")
        return result

    meta = _parse_simple_yaml(fm_text)

    # --- Validate 'name' ---
    name = meta.get('name', '')
    if not name:
        result.error("Missing required field: name")
    else:
        if len(name) > 64:
            result.error(f"name too long: {len(name)} chars (max 64)")
        if not re.match(r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?$', name):
            result.error(f"name '{name}' invalid: must be lowercase alphanumeric + hyphens, no leading/trailing hyphens")
        if '--' in name:
            result.error(f"name '{name}' contains consecutive hyphens")
        if name != dir_name:
            result.error(f"name '{name}' doesn't match directory name '{dir_name}'")

    # --- Validate 'description' ---
    desc = meta.get('description', '')
    if not desc:
        result.error("Missing required field: description")
    else:
        if len(desc) > 1024:
            result.error(f"description too long: {len(desc)} chars (max 1024)")
        if len(desc) < 50:
            result.warn(f"description very short ({len(desc)} chars) — may not trigger well")

        # Check for trigger guidance
        desc_lower = desc.lower()
        has_trigger = any(kw in desc_lower for kw in [
            'trigger', 'use this skill', 'use when', 'activate when',
            'whenever the user', 'whenever you'
        ])
        has_anti_trigger = any(kw in desc_lower for kw in [
            'do not trigger', 'don\'t trigger', 'not trigger',
            'do not use', 'don\'t use'
        ])

        if not has_trigger:
            result.warn("description lacks trigger guidance — add 'Use this skill when...' or 'TRIGGER on:'")
        if not has_anti_trigger:
            result.note("Consider adding DO NOT TRIGGER guidance to prevent false activations")

    # --- Validate body ---
    body_lines = body.split('\n')
    line_count = len(body_lines)

    if line_count == 0:
        result.warn("SKILL.md body is empty — add instructions")
    elif line_count > 500:
        result.warn(f"SKILL.md body is {line_count} lines (recommended < 500). Move details to references/")
    elif line_count > 300:
        result.note(f"SKILL.md body is {line_count} lines — getting long, consider splitting")

    # Check for decision tree / structured routing
    has_decision_tree = any(kw in body for kw in [
        '├─', '└─', 'Decision Tree', 'decision tree',
        'What kind of', 'Choose Your Approach', 'Which', 'Route'
    ])
    if not has_decision_tree and line_count > 50:
        result.warn("No decision tree found — consider adding routing logic for different task types")

    # Check for anti-patterns section
    has_anti_patterns = any(kw in body for kw in ['❌', 'Anti-Pattern', 'anti-pattern', "Don't", "Don't"])
    if not has_anti_patterns and line_count > 100:
        result.note("Consider adding Anti-Patterns section (❌/✅ pairs)")

    # --- Check directory structure ---
    for subdir in ['scripts', 'references', 'assets']:
        subdir_path = skill_path / subdir
        if not subdir_path.exists():
            result.note(f"Optional directory '{subdir}/' not present")

    # --- Check resource references ---
    # Find references like [text](./references/foo.md) or (references/foo.md)
    ref_pattern = re.compile(r'\((?:\./)?([^)]+\.(?:md|py|sh|json|txt))\)')
    for match in ref_pattern.finditer(body):
        ref_path = match.group(1)
        full_ref = skill_path / ref_path
        if not full_ref.exists():
            result.warn(f"Referenced file '{ref_path}' not found in skill directory")

    # --- Check for explain-the-why ---
    body_upper = body.upper()
    always_count = len(re.findall(r'\bALWAYS\b', body_upper)) - len(re.findall(r'\balways\b', body))
    never_count = len(re.findall(r'\bNEVER\b', body_upper)) - len(re.findall(r'\bnever\b', body))
    caps_count = always_count + never_count
    if caps_count >= 3:
        result.warn(f"Found {caps_count} ALL-CAPS ALWAYS/NEVER — consider explaining the 'why' instead of rigid rules")

    return result


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)

    skill_dir = sys.argv[1]
    result = validate_skill(skill_dir)
    print(result.summary())
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
