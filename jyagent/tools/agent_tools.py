# Agent self-modification tools: evolve_self, add_tool

import ast
import importlib
import inspect
import json
import os
import sys
import types
from typing import Any

from ..registry import get_registry
from .utils import strip_unsupported_schema_keys


def evolve_self(module_name: str, feedback: str) -> str:
    """Hot-reload a jyagent module after external edits.

    Validates the module's current on-disk source with AST checks,
    then hot-reloads it into the running process.
    """
    try:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        module_path = os.path.join(pkg_dir, f"{module_name}.py")

        if not os.path.exists(module_path):
            # Try sub-package
            module_path = os.path.join(pkg_dir, module_name.replace(".", "/") + ".py")
            if not os.path.exists(module_path):
                return f"Error: Module file not found: {module_name}.py"

        # AST validation
        with open(module_path, "r") as f:
            source = f.read()
        try:
            ast.parse(source)
        except SyntaxError as e:
            return f"Syntax error at line {e.lineno}: {e.msg}"

        # Hot-reload
        qualified = f"jyagent.{module_name}"
        if qualified in sys.modules:
            importlib.reload(sys.modules[qualified])
            return f"✅ Module '{module_name}' validated and hot-reloaded. ({feedback})"
        else:
            return f"⚠️ Module '{module_name}' validated but not yet imported (no reload needed). ({feedback})"

    except Exception as e:
        return f"Hot-reload failed: {e}"


def add_tool(name: str, code: str, description: str, parameters: str) -> str:
    """Create and register a new tool at runtime. Persists as a tool_*.py file."""
    try:
        try:
            params_dict = json.loads(parameters)
        except json.JSONDecodeError:
            return "Error: parameters must be valid JSON"

        # Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error at line {e.lineno}: {e.msg}"

        # Execute the code to get the function
        module = types.ModuleType(f"tool_{name}")
        exec(compile(code, f"<tool_{name}>", "exec"), module.__dict__)
        fn = getattr(module, name, None)
        if fn is None or not callable(fn):
            return f"Error: Code must define a callable function named '{name}'"

        cleaned_params = strip_unsupported_schema_keys(params_dict)

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
            tool_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
