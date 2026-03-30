"""
Evolution Engine — enables the agent to iteratively improve its own source code.

Reads current source, asks Claude for improvements, validates, backs up, and hot-reloads.
Delegates prompt construction and evaluation to evolution_strategy.py when available,
with built-in fallback prompts for safety.

v3: Migrated from boot project. Now lives inside jyagent package.
"""

import importlib
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Optional

from .validator import validate_single_module, get_top_level_names_from_source

# Source directory is the package directory itself (jyagent/)
GENERATED_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(GENERATED_DIR, "backups")
MANIFEST_PATH = os.path.join(GENERATED_DIR, "manifest.json")
MAX_EVOLUTIONS_PER_SESSION = 5

_evolution_count = 0


def _load_valid_modules() -> list[str]:
    """Load the list of evolvable modules from manifest.json, with fallback."""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                manifest = json.load(f)
            return [
                name for name, meta in manifest.get("modules", {}).items()
                if meta.get("evolvable", True)
            ]
        except (json.JSONDecodeError, OSError):
            pass
    return ["tools", "self_memory", "planner", "agent"]


def update_manifest(module_name: str, new_code: str) -> None:
    """Update manifest.json with the new module's exports after evolution."""
    if not os.path.exists(MANIFEST_PATH):
        return

    try:
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)

        exports = get_top_level_names_from_source(new_code)
        if module_name not in manifest.get("modules", {}):
            manifest.setdefault("modules", {})[module_name] = {
                "exports": exports,
                "evolvable": True,
            }
        else:
            manifest["modules"][module_name]["exports"] = exports

        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
    except (json.JSONDecodeError, OSError, SyntaxError):
        pass  # non-fatal: manifest update is best-effort


def read_module_source(module_name: str) -> str:
    """Read the current source code of a generated module."""
    path = os.path.join(GENERATED_DIR, f"{module_name}.py")
    if not os.path.exists(path):
        return f"Error: Module {module_name} not found"
    with open(path, "r") as f:
        return f.read()


def get_evolution_history(persistent_memory: Any) -> list:
    """Load evolution history from persistent memory."""
    history = persistent_memory.load("_evolution_history")
    return history if history is not None else []


def _save_evolution_record(persistent_memory: Any, record: dict) -> None:
    """Append an evolution record to persistent memory."""
    history = get_evolution_history(persistent_memory)
    history.append(record)
    persistent_memory.save("_evolution_history", history)


def _backup_module(module_name: str) -> str:
    """Back up current module source to backups directory. Returns backup path."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    source = read_module_source(module_name)
    existing = [f for f in os.listdir(BACKUP_DIR) if f.startswith(f"{module_name}_v")]
    version = len(existing) + 1
    backup_path = os.path.join(BACKUP_DIR, f"{module_name}_v{version}.py")
    with open(backup_path, "w") as f:
        f.write(source)
    return backup_path


def _rollback_module(module_name: str, backup_path: str) -> None:
    """Restore a module from a backup file."""
    with open(backup_path, "r") as f:
        source = f.read()
    target = os.path.join(GENERATED_DIR, f"{module_name}.py")
    with open(target, "w") as f:
        f.write(source)


# ---------------------------------------------------------------------------
# Built-in fallback prompts (used when evolution_strategy.py is unavailable)
# ---------------------------------------------------------------------------

def _default_evolution_prompt(
    module_name: str,
    current_source: str,
    feedback: str,
    interaction_log: str,
    history_summary: str,
) -> str:
    return f"""You are improving an AI agent's generated module: {module_name}.py

## Current Source Code
```python
{current_source}
```

## Feedback / Weakness Identified
{feedback}

## Recent Interaction Log
{interaction_log[:3000]}

## Previous Evolution History
{history_summary or "No previous evolutions."}

## Instructions
- Output ONLY a single Python code block with the improved version of {module_name}.py
- Preserve ALL existing function signatures and exports (do not remove any)
- Keep all relative imports (from .xxx import ...)
- Only use standard library + anthropic package
- Focus on the specific feedback given
- Add a comment at the top: # evolved v{{version}} — {{one-line changelog}}

```python
# Your improved code here
```"""


def _default_evaluation_prompt(interaction_log: str, sources_text: str) -> str:
    return f"""Analyze this AI agent's recent interactions and source code. Identify the single most impactful improvement.

## Recent Interactions
{interaction_log[:3000]}

## Current Source Code
{sources_text}

Respond in this exact JSON format (no other text):
{{"module": "<module_name>", "weakness": "specific description", "suggestion": "concrete improvement"}}
"""


def _get_evolution_prompt(
    module_name: str,
    current_source: str,
    feedback: str,
    interaction_log: str,
    history_summary: str,
) -> str:
    """Get evolution prompt from strategy module, falling back to built-in."""
    try:
        from .evolution_strategy import build_evolution_prompt
        return build_evolution_prompt(
            module_name, current_source, feedback, interaction_log, history_summary
        )
    except (ImportError, Exception):
        return _default_evolution_prompt(
            module_name, current_source, feedback, interaction_log, history_summary
        )


def _get_evaluation_prompt(interaction_log: str, sources_text: str) -> str:
    """Get evaluation prompt from strategy module, falling back to built-in."""
    try:
        from .evolution_strategy import build_evaluation_prompt
        return build_evaluation_prompt(interaction_log, sources_text)
    except (ImportError, Exception):
        return _default_evaluation_prompt(interaction_log, sources_text)


def _parse_evaluation(response_text: str) -> Optional[dict]:
    """Parse evaluation response using strategy module, falling back to json.loads."""
    try:
        from .evolution_strategy import parse_evaluation_result
        return parse_evaluation_result(response_text)
    except (ImportError, Exception):
        try:
            return json.loads(response_text.strip())
        except (json.JSONDecodeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Core evolution logic
# ---------------------------------------------------------------------------

def evolve_module(
    client: Any,
    module_name: str,
    feedback: str,
    interaction_log: str,
    persistent_memory: Any,
) -> tuple[bool, str]:
    """
    Evolve a single generated module based on feedback.

    Returns (success, message) where message is either a changelog or error.
    """
    global _evolution_count

    if _evolution_count >= MAX_EVOLUTIONS_PER_SESSION:
        return (False, f"Evolution limit reached ({MAX_EVOLUTIONS_PER_SESSION} per session)")

    valid_modules = _load_valid_modules()
    if module_name not in valid_modules:
        return (False, f"Cannot evolve '{module_name}'. Valid: {valid_modules}")

    current_source = read_module_source(module_name)
    if current_source.startswith("Error:"):
        return (False, current_source)

    history = get_evolution_history(persistent_memory)
    history_summary = ""
    if history:
        recent = history[-5:]
        history_summary = "\n".join(
            f"- v{h.get('version', '?')}: {h.get('module', '?')} — {h.get('changelog', '?')}"
            for h in recent
        )

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    prompt = _get_evolution_prompt(
        module_name, current_source, feedback, interaction_log, history_summary
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=16384,  # Increased from 8000: modules can be 300+ lines, need room
            system="You are a code improvement engine. Output ONLY the improved Python code block. No explanations.",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return (False, f"API call failed: {e}")

    response_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    # Extract code block
    match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if not match:
        return (False, "No code block found in evolution response")

    new_code = match.group(1).strip()

    # Validate
    success, errors = validate_single_module(f"{module_name}.py", new_code)
    if not success:
        return (False, f"Validation failed: {'; '.join(errors)}")

    # Backup current version
    backup_path = _backup_module(module_name)

    # Write new version
    target_path = os.path.join(GENERATED_DIR, f"{module_name}.py")
    with open(target_path, "w") as f:
        f.write(new_code + "\n")

    # Update manifest with new exports
    update_manifest(module_name, new_code)

    # Try hot-reload
    reload_ok = hot_reload_module(module_name)
    if not reload_ok:
        _rollback_module(module_name, backup_path)
        return (False, "Hot-reload failed, rolled back to previous version")

    # Record evolution
    version = len([f for f in os.listdir(BACKUP_DIR) if f.startswith(f"{module_name}_v")])
    record = {
        "module": module_name,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "feedback": feedback[:200],
        "changelog": f"Evolved {module_name} based on: {feedback[:100]}",
    }
    _save_evolution_record(persistent_memory, record)
    _evolution_count += 1

    return (True, f"Successfully evolved {module_name}.py (v{version}). Backup: {backup_path}")


def hot_reload_module(module_name: str) -> bool:
    """Hot-reload a generated module. Registry-based tools re-register on import."""
    try:
        full_name = f"jyagent.{module_name}"

        # Clear cached module
        for key in list(sys.modules.keys()):
            if key == full_name or key.startswith(f"{full_name}."):
                del sys.modules[key]

        # Re-import (tool modules re-register with the registry on import)
        importlib.import_module(full_name)
        return True
    except Exception:
        return False


def evaluate_performance(
    client: Any, interaction_log: str, current_sources: dict[str, str]
) -> Optional[dict]:
    """
    Ask Claude to evaluate the agent's performance and identify improvement areas.

    Returns dict with keys: module, weakness, suggestion — or None on failure.
    """
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    sources_text = "\n\n".join(
        f"### {name}\n```python\n{src}\n```" for name, src in current_sources.items()
    )

    prompt = _get_evaluation_prompt(interaction_log, sources_text)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,  # Increased from 500: allows more detailed evaluation JSON
            system="You are a code reviewer. Respond ONLY with the JSON object requested.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _parse_evaluation(text)
    except Exception:
        return None
