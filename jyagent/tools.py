# Core tools: run_shell, read_file, write_file, list_directory, evolve_self, add_tool, manage_memory, manage_skills
# Native tool_*.py files are auto-discovered at import time. Runtime tools via add_tool persist as tool_*.py.

import importlib
import importlib.util
import subprocess
import os
import json
import inspect
import glob
import sys
import traceback
from typing import Any
from .registry import get_registry

_client = None


def set_client(c: Any) -> None:
    global _client
    _client = c


# ─── Shared constants (also used by tool_glob_grep) ──────────────────────────

SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env',
    '.mypy_cache', '.pytest_cache', '.tox', '.eggs', '*.egg-info',
    'dist', 'build', '.next', '.nuxt', 'coverage', '.coverage',
    '.idea', '.vscode', '.DS_Store',
}

BINARY_EXTS = {
    '.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', '.o', '.a',
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
    '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.rar', '.7z',
    '.woff', '.woff2', '.ttf', '.eot',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.db', '.sqlite', '.sqlite3',
}


# ─── Utility ──────────────────────────────────────────────────────────────────

def _strip_unsupported_schema_keys(properties: dict) -> dict:
    """Strip keys not supported by Bedrock's JSON Schema validator (e.g., 'default')."""
    unsupported_keys = {"default"}
    cleaned = {}
    for prop_name, prop_def in properties.items():
        if isinstance(prop_def, dict):
            cleaned[prop_name] = {k: v for k, v in prop_def.items() if k not in unsupported_keys}
        else:
            cleaned[prop_name] = prop_def
    return cleaned


# ─── Core tool functions ──────────────────────────────────────────────────────

def run_shell(command: str, timeout: int = 60) -> str:
    """Execute a shell command and return the output. Use the timeout parameter for long-running commands like installs, downloads, or compilations."""
    try:
        effective_timeout = max(1, min(int(timeout), 600))
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=effective_timeout)
        output = result.stdout
        if result.stderr:
            output += "\nSTDERR: " + result.stderr
        if result.returncode != 0 and not output.strip():
            output = f"Command exited with code {result.returncode}"
        if len(output) > 50000:
            output = output[:50000] + "\n\n[... output truncated at 50000 chars ...]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {effective_timeout} seconds"
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str, offset: int = 0, limit: int = 0, line_numbers: bool = False) -> str:
    """Read and return the contents of a file.
    
    Supports line-level pagination for large files:
    - offset: 1-based starting line number (0 = from beginning)
    - limit: max lines to return (0 = all lines, subject to char cap)
    - line_numbers: prepend "L{n}: " to each line (useful for edit_file)
    
    Output includes a header with file stats when using pagination or line_numbers.
    """
    MAX_READ_CHARS = 50000
    try:
        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        
        # Check if binary
        _, ext = os.path.splitext(path)
        if ext.lower() in BINARY_EXTS:
            size = os.path.getsize(path)
            return f"[Binary file: {path} ({size} bytes)]"
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
        
        total_lines = len(all_lines)
        total_chars = sum(len(l) for l in all_lines)
        
        # Apply offset/limit (offset is 1-based for user convenience)
        if offset > 0:
            start = max(0, offset - 1)  # convert to 0-based
        else:
            start = 0
        
        if limit > 0:
            end = min(start + limit, total_lines)
        else:
            end = total_lines
        
        selected_lines = all_lines[start:end]
        
        # Build output
        is_paginated = (offset > 0 or limit > 0)
        header = ""
        
        if is_paginated or line_numbers:
            showing_start = start + 1
            showing_end = start + len(selected_lines)
            header = f"[{path}: {total_lines} lines total, showing L{showing_start}-L{showing_end}]\n"
        
        if line_numbers:
            output_parts = []
            for i, line in enumerate(selected_lines):
                line_num = start + i + 1
                # Strip trailing newline for cleaner format, then add it back
                line_text = line.rstrip('\n')
                output_parts.append(f"L{line_num}: {line_text}")
            content = '\n'.join(output_parts)
        else:
            content = ''.join(selected_lines)
        
        result = header + content
        
        # Truncation safety net (still respect MAX_READ_CHARS)
        if len(result) > MAX_READ_CHARS:
            head = MAX_READ_CHARS - 5000
            result = (
                result[:head]
                + f"\n\n[... truncated at {MAX_READ_CHARS} chars, "
                + f"total file: {total_chars} chars, {total_lines} lines. "
                + f"Use offset/limit for pagination ...]\n\n"
                + result[-5000:]
            )
        
        return result
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed.
    
    For surgical edits (replacing specific text), prefer edit_file instead —
    it saves tokens by not requiring the full file content.
    """
    try:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        lines = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
        return f"Successfully wrote {len(content)} chars ({lines} lines) to {path}"
    except Exception as e:
        return f"Error: {e}"


def list_directory(path: str = ".", depth: int = 1, limit: int = 200, offset: int = 0) -> str:
    """List the contents of a directory with tree-style output.
    
    Enhanced features:
    - depth: recursion depth (1 = immediate children, 2+ = nested, default 1)
    - limit: max entries to return (default 200, prevents context explosion)
    - offset: skip first N entries for pagination (0-based, default 0)
    - Shows file sizes, symlink markers, and directory suffixes
    - Automatically skips .git, node_modules, __pycache__, etc.
    
    If no path is provided, lists the current directory.
    """
    try:
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return f"Error: Not a directory: {path}"
        
        entries = []
        
        def _walk(dir_path, current_depth, prefix=""):
            """Recursively collect entries with tree-style prefixes."""
            if current_depth > depth:
                return
            
            try:
                items = sorted(os.listdir(dir_path))
            except PermissionError:
                entries.append(f"{prefix}[permission denied]")
                return
            
            # Separate dirs and files
            dirs_list = []
            files_list = []
            for item in items:
                # Skip hidden files/dirs at depth > 1 (keep them at root level)
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    if item in SKIP_DIRS:
                        continue
                    dirs_list.append(item)
                else:
                    files_list.append(item)
            
            # Process directories first, then files
            all_items = [(d, True) for d in dirs_list] + [(f, False) for f in files_list]
            
            for i, (item, is_dir) in enumerate(all_items):
                item_path = os.path.join(dir_path, item)
                is_last = (i == len(all_items) - 1)
                
                # Tree connector
                if current_depth == 1 and not prefix:
                    connector = ""
                    child_prefix = "  "
                else:
                    connector = "└── " if is_last else "├── "
                    child_prefix = prefix + ("    " if is_last else "│   ")
                
                if is_dir:
                    suffix = "/"
                    if os.path.islink(item_path):
                        suffix = "@ → " + os.path.realpath(item_path)
                    entries.append(f"{prefix}{connector}{item}{suffix}")
                    
                    # Recurse into subdirectory
                    if current_depth < depth:
                        _walk(item_path, current_depth + 1, child_prefix)
                else:
                    # File with size
                    try:
                        size = os.path.getsize(item_path)
                        size_str = _format_size(size)
                    except OSError:
                        size_str = "?"
                    
                    link_marker = "@ " if os.path.islink(item_path) else ""
                    entries.append(f"{prefix}{connector}{link_marker}{item}  ({size_str})")
        
        _walk(path, 1)
        
        # Apply offset/limit pagination
        total_entries = len(entries)
        paginated = entries[offset:offset + limit]
        
        # Build header
        rel_path = os.path.relpath(path, os.getcwd())
        if rel_path == '.':
            rel_path = os.path.basename(path) or path
        
        header = f"📁 {rel_path}/ ({total_entries} entries"
        if offset > 0:
            header += f", showing from #{offset + 1}"
        if total_entries > offset + limit:
            header += f", {total_entries - offset - limit} more"
        header += ")\n"
        
        if not paginated:
            return header + "  (empty directory)"
        
        result = header + '\n'.join(paginated)
        if total_entries > offset + limit:
            result += f"\n\n[... {total_entries - offset - limit} more entries. Use offset={offset + limit} to see next page]"
        
        return result
    except Exception as e:
        return f"Error: {e}"


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f}MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.1f}GB"


# ─── Hot-reload tool (simplified from former evolve_self) ─────────────────────

def evolve_self(module_name: str, feedback: str) -> str:
    """Hot-reload a jyagent module after external edits.

    This is a lightweight utility: validate the module's current on-disk source
    with AST checks, then hot-reload it into the running process.
    Actual code changes should be made via edit_file / write_file in conversation,
    NOT by asking Claude to regenerate an entire module.

    Args:
        module_name: Module to reload (e.g. 'planner', 'tools', 'agent').
        feedback: Reason for the reload (logged for reference).
    """
    import importlib

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(pkg_dir, f"{module_name}.py")

    if not os.path.exists(module_path):
        return f"Error: Module '{module_name}' not found at {module_path}"

    # 1. Syntax check before reload
    try:
        with open(module_path, "r") as f:
            source = f.read()
        import ast
        ast.parse(source)
    except SyntaxError as e:
        return f"Syntax error in {module_name}.py line {e.lineno}: {e.msg} — not reloading."
    except Exception as e:
        return f"Validation error: {e}"

    # 2. Hot-reload
    full_name = f"jyagent.{module_name}"
    try:
        if full_name in sys.modules:
            importlib.reload(sys.modules[full_name])
        else:
            importlib.import_module(full_name)
        return f"✅ Module '{module_name}' reloaded successfully. (reason: {feedback})"
    except Exception as e:
        return f"Hot-reload failed: {e}"


# ─── Dynamic tool creation ───────────────────────────────────────────────────

def add_tool(name: str, code: str, description: str, parameters: str) -> str:
    """Create and register a new tool at runtime. Automatically persists as a tool_*.py file for future sessions."""
    try:
        try:
            params_dict = json.loads(parameters)
        except json.JSONDecodeError:
            return "Error: parameters must be valid JSON"
        
        # Syntax check
        import ast
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error at line {e.lineno}: {e.msg}"
        
        # Execute the code to get the function
        import types
        module = types.ModuleType(f"tool_{name}")
        exec(compile(code, f"<tool_{name}>", "exec"), module.__dict__)
        fn = getattr(module, name, None)
        if fn is None or not callable(fn):
            return f"Error: Code must define a callable function named '{name}'"
        
        cleaned_params = _strip_unsupported_schema_keys(params_dict)
        
        # Only mark params without defaults as required
        required_params = []
        try:
            sig = inspect.signature(fn)
            for pname, param in sig.parameters.items():
                if param.default is inspect.Parameter.empty:
                    required_params.append(pname)
        except (ValueError, TypeError):
            required_params = list(cleaned_params.keys())
        
        schema = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": cleaned_params,
                "required": required_params
            }
        }
        
        get_registry().register(name, fn, schema)
        
        # Persist as a tool_*.py file so it auto-loads on restart
        try:
            tool_dir = os.path.dirname(os.path.abspath(__file__))
            tool_path = os.path.join(tool_dir, f"tool_{name}.py")
            if not os.path.exists(tool_path):
                tool_code = code + f"\n\nTOOL_SCHEMA = {json.dumps(schema, indent=2)}\n"
                with open(tool_path, 'w') as f:
                    f.write(tool_code)
        except Exception:
            pass  # Registration succeeded even if file persistence fails
        
        return f"Successfully created and registered tool '{name}'"
    except Exception as e:
        return f"Error: {e}"


# ─── Memory management tool (LLM-accessible) ─────────────────────────────────

def manage_memory(action: str, text: str = "", category: str = "") -> str:
    """Manage the agent's self-use memory system. Actions: 'remember' (save a learning/fact), 'forget' (remove memories by keyword), 'show' (display all memories), 'topic' (manage topic files: list/read/write/delete), 'goal' (add/complete a goal), 'note' (add a working note). This tool lets you proactively remember things about the user for future sessions."""
    from .self_memory import (
        remember, forget, show_memory,
        append_memory_md,
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
        
        # Backwards compatibility
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


# ─── Skills management tool (LLM-accessible) ─────────────────────────────────

def manage_skills(action: str, name: str = "", description: str = "",
                  instructions: str = "", resource_path: str = "") -> str:
    """Manage Agent Skills (agentskills.io). Actions: 'list' (show all skills), 'activate'/'deactivate' (control which skills are loaded into context), 'info' (show skill details), 'create' (create new skill), 'delete' (remove skill), 'resources' (list skill files), 'read' (read a skill resource file), 'reload' (re-scan skills directory)."""
    from .skills import get_skill_manager
    
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
            
            result = mgr.create_skill(name, description, instructions)
            return result  # create_skill already returns a user-friendly message
        
        elif action == "delete":
            if not name:
                return "Error: 'name' parameter required"
            result = mgr.delete_skill(name)
            return result  # delete_skill already returns a user-friendly message
        
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


# ============================================================
# Register core tools
# ============================================================

_registry = get_registry()

CORE_TOOLS = [
    # --- run_shell ---
    {
        "name": "run_shell",
        "description": "Execute a shell command and return the output. Use the timeout parameter for long-running commands like installs, downloads, or compilations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (1-600). Defaults to 60. Use 300 for installs/downloads."
                }
            },
            "required": ["command"]
        }
    },
    # --- read_file ---
    {
        "name": "read_file",
        "description": "Read and return the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based starting line number. Use with limit for pagination of large files. (default: 0 = from beginning)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of lines to return. (default: 0 = all lines, subject to 50K char cap)"
                },
                "line_numbers": {
                    "type": "boolean",
                    "description": "Prepend 'L{n}: ' to each line. Useful before edit_file to see exact line numbers. (default: false)"
                }
            },
            "required": ["path"]
        }
    },
    # --- write_file ---
    {
        "name": "write_file",
        "description": "Write content to a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    # --- list_directory ---
    {
        "name": "list_directory",
        "description": "List the contents of a directory. If no path is provided, lists the current directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (defaults to current directory if omitted)"
                },
                "depth": {
                    "type": "integer",
                    "description": "Recursion depth (1 = immediate children, 2+ = nested). Default 1."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return (default 200). Prevents context explosion on large dirs."
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip first N entries for pagination (0-based, default 0)."
                }
            },
            "required": []
        }
    },
    # --- evolve_self ---
    {
        "name": "evolve_self",
        "description": "Validate and hot-reload a jyagent module after edits. Use edit_file to make changes first, then call this to reload.",
        "input_schema": {
            "type": "object",
            "properties": {
                "module_name": {
                    "type": "string",
                    "description": "Name of the module to reload (e.g. 'planner', 'tools', 'agent')"
                },
                "feedback": {
                    "type": "string",
                    "description": "Reason for the reload (logged for reference)"
                }
            },
            "required": ["module_name", "feedback"]
        }
    },
    # --- add_tool ---
    {
        "name": "add_tool",
        "description": "Create and register a new tool at runtime. Automatically persists as a tool_*.py file for future sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the new tool"
                },
                "code": {
                    "type": "string",
                    "description": "Python code for the tool function"
                },
                "description": {
                    "type": "string",
                    "description": "Description of what the tool does"
                },
                "parameters": {
                    "type": "string",
                    "description": "JSON string describing the tool's parameters"
                }
            },
            "required": ["name", "code", "description", "parameters"]
        }
    },
    # --- manage_memory ---
    {
        "name": "manage_memory",
        "description": "Manage the agent's self-use memory system. Actions: 'remember' (save a learning/fact), 'forget' (remove memories by keyword), 'show' (display all memories), 'topic' (manage topic files: list/read/write/delete), 'goal' (add/complete a goal), 'note' (add a working note). This tool lets you proactively remember things about the user for future sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["remember", "forget", "show", "topic", "goal", "note"],
                    "description": "Action: remember, forget, show, topic, goal, note"
                },
                "text": {
                    "type": "string",
                    "description": "Main text content (what to remember, keyword to forget, topic command like 'read:chrome-mcp', etc.)"
                },
                "category": {
                    "type": "string",
                    "description": "Category for learnings: correction, preference, gotcha, tip, workflow, user_stated, goal, note",
                }
            },
            "required": ["action"]
        }
    },
    # --- manage_skills ---
    {
        "name": "manage_skills",
        "description": "Manage Agent Skills (agentskills.io). Actions: 'list' (show all skills), 'activate'/'deactivate' (control which skills are loaded into context), 'info' (show skill details), 'create' (create new skill), 'delete' (remove skill), 'resources' (list skill files), 'read' (read a skill resource file), 'reload' (re-scan skills directory).",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "activate", "deactivate", "info", "create", "delete", "resources", "read", "reload"],
                    "description": "Action: list, activate, deactivate, info, create, delete, resources, read, reload"
                },
                "name": {
                    "type": "string",
                    "description": "Skill name (for activate, deactivate, info, create, delete, resources, read)"
                },
                "description": {
                    "type": "string",
                    "description": "Skill description (for create action)"
                },
                "instructions": {
                    "type": "string",
                    "description": "Skill instructions/body content (for create action)"
                },
                "resource_path": {
                    "type": "string",
                    "description": "Relative path to a resource file within the skill (for read action)"
                }
            },
            "required": ["action"]
        }
    },
]

for tool_def in CORE_TOOLS:
    _registry.register(tool_def["name"], globals()[tool_def["name"]], tool_def)


# ============================================================
# Auto-discover tool_*.py files
# ============================================================

def _auto_discover_tools():
    """Auto-discover and register tools from tool_*.py files.
    
    Each tool_*.py file can export tools in two ways:
    1. TOOL_SCHEMA (dict) + a function with the same name as schema["name"]
    2. TOOL_SCHEMAS (list of dicts) + corresponding functions for each schema
    
    If neither is present, falls back to inferring schema from function signature.
    """
    tool_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(tool_dir, "tool_*.py")
    
    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        module_name = filename[:-3]  # strip .py
        
        try:
            # Use package-qualified name so relative imports (from .xxx) work
            qualified_name = f"jyagent.{module_name}"
            spec = importlib.util.spec_from_file_location(qualified_name, filepath,
                submodule_search_locations=[])
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            # Ensure parent package is in sys.modules
            import sys
            if "jyagent" not in sys.modules:
                import jyagent as _pkg
                sys.modules["jyagent"] = _pkg
            spec.loader.exec_module(module)
            sys.modules[qualified_name] = module
            
            # Collect tools to register from this file
            tools_to_register = []
            
            # Check for TOOL_SCHEMAS (plural) first — multi-tool files
            schemas = getattr(module, 'TOOL_SCHEMAS', None)
            if schemas and isinstance(schemas, list):
                for schema in schemas:
                    tool_name = schema.get("name", "")
                    fn = getattr(module, tool_name, None)
                    if fn and callable(fn):
                        tools_to_register.append((tool_name, fn, schema))
            
            # Fall back to TOOL_SCHEMA (singular) — single-tool files
            elif hasattr(module, 'TOOL_SCHEMA'):
                schema = module.TOOL_SCHEMA
                tool_name = schema.get("name", module_name.replace("tool_", ""))
                fn = getattr(module, tool_name, None)
                if fn and callable(fn):
                    tools_to_register.append((tool_name, fn, schema))
            
            # Fall back to auto-inference from function signature
            else:
                tool_name = module_name.replace("tool_", "")
                fn = getattr(module, tool_name, None)
                if fn and callable(fn):
                    sig = inspect.signature(fn)
                    properties = {}
                    required = []
                    for pname, param in sig.parameters.items():
                        prop = {"type": "string", "description": f"Parameter: {pname}"}
                        if param.annotation != inspect.Parameter.empty:
                            type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
                            prop["type"] = type_map.get(param.annotation, "string")
                        properties[pname] = prop
                        if param.default is inspect.Parameter.empty:
                            required.append(pname)
                    
                    schema = {
                        "name": tool_name,
                        "description": fn.__doc__ or f"Auto-discovered tool: {tool_name}",
                        "input_schema": {
                            "type": "object",
                            "properties": _strip_unsupported_schema_keys(properties),
                            "required": required
                        }
                    }
                    tools_to_register.append((tool_name, fn, schema))
            
            # Register all tools from this file
            for tname, fn, schema in tools_to_register:
                _registry.register(tname, fn, schema)
                
        except Exception:
            pass  # Silently skip broken tool files


# Run auto-discovery at import time
_auto_discover_tools()
