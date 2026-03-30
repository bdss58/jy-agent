"""
Code Validator — AST-based checks before evolution rewrites are applied.
Reads required symbols from manifest.json when available, falls back to defaults.
"""

import ast
import json
import os
from typing import Dict, List, Tuple

# Default expected top-level names (fallback when manifest.json does not exist)
DEFAULT_REQUIRED_SYMBOLS = {
    "tools.py": ["TOOL_SCHEMAS", "TOOL_FUNCTIONS", "run_shell", "read_file", "write_file", "evolve_self", "add_tool", "set_client"],
    "memory.py": ["ConversationMemory", "PersistentMemory"],
    "planner.py": ["plan_next_action"],
    "agent.py": ["run"],
    "registry.py": ["ToolRegistry", "get_registry"],
    "evolution_strategy.py": ["build_evolution_prompt", "build_evaluation_prompt", "parse_evaluation_result"],
}

# Imports that should never appear in generated code
BLOCKED_IMPORTS = {"ctypes"}

# Project-level modules that are allowed to import
ALLOWED_IMPORTS = {"evolver", "validator"}

# Source directory is the package directory itself
GENERATED_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(GENERATED_DIR, "manifest.json")


def _get_top_level_names(tree: ast.Module) -> set:
    """Extract all top-level names defined in an AST."""
    names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def get_top_level_names_from_source(code: str) -> list:
    """Parse source code and return sorted list of top-level names."""
    tree = ast.parse(code)
    return sorted(_get_top_level_names(tree))


def _get_imports(tree: ast.Module) -> set:
    """Extract all imported module names from an AST."""
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.level:  # skip relative imports
                imports.add(node.module.split(".")[0])
    return imports


def _load_manifest() -> Dict[str, List[str]]:
    """
    Load required symbols from manifest.json.
    Returns {filename: [required_symbols]} dict.
    Falls back to DEFAULT_REQUIRED_SYMBOLS if manifest does not exist.
    """
    if not os.path.exists(MANIFEST_PATH):
        return DEFAULT_REQUIRED_SYMBOLS

    try:
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
        result = {}
        for module_name, meta in manifest.get("modules", {}).items():
            filename = f"{module_name}.py"
            result[filename] = meta.get("exports", [])
        return result
    except (json.JSONDecodeError, OSError):
        return DEFAULT_REQUIRED_SYMBOLS


def validate_modules(code_dict: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Validate a dict of {filename: source_code} for the generated agent.

    Returns (success, errors) where success is True if all checks pass.
    """
    errors = []
    required_symbols = _load_manifest()

    # Check that all expected files are present
    for filename in required_symbols:
        if filename not in code_dict:
            errors.append(f"Missing expected file: {filename}")

    for filename, code in code_dict.items():
        # 1. Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"{filename}: Syntax error at line {e.lineno}: {e.msg}")
            continue  # can't do further checks without a valid AST

        # 2. Required symbols check
        if filename in required_symbols:
            defined = _get_top_level_names(tree)
            for symbol in required_symbols[filename]:
                if symbol not in defined:
                    errors.append(f"{filename}: Missing required symbol '{symbol}'")

        # 3. Dangerous import check
        imports = _get_imports(tree)
        blocked = imports & BLOCKED_IMPORTS
        if blocked:
            errors.append(
                f"{filename}: Blocked import(s): {', '.join(sorted(blocked))}"
            )

    return (len(errors) == 0, errors)


def validate_single_module(filename: str, code: str) -> Tuple[bool, List[str]]:
    """
    Validate a single module's source code.
    Checks syntax, required symbols (from manifest), and blocked imports.
    Unknown modules (not in manifest) get syntax + blocked-import checks only.
    """
    errors = []

    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"{filename}: Syntax error at line {e.lineno}: {e.msg}")
        return (False, errors)

    # 2. Required symbols check (from manifest or defaults)
    required_symbols = _load_manifest()
    if filename in required_symbols:
        defined = _get_top_level_names(tree)
        for symbol in required_symbols[filename]:
            if symbol not in defined:
                errors.append(f"{filename}: Missing required symbol '{symbol}'")

    # 3. Dangerous import check
    imports = _get_imports(tree)
    blocked = imports & BLOCKED_IMPORTS
    if blocked:
        errors.append(
            f"{filename}: Blocked import(s): {', '.join(sorted(blocked))}"
        )

    return (len(errors) == 0, errors)
