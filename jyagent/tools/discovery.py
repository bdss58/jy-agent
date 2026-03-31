# Auto-discovery engine — finds and registers tool_*.py files at import time.

import glob
import importlib
import importlib.util
import inspect
import os
import sys

from ..registry import get_registry
from .utils import strip_unsupported_schema_keys


def auto_discover_tools() -> None:
    """Auto-discover and register tools from tool_*.py files.

    Each tool_*.py file can export tools in two ways:
    1. TOOL_SCHEMA (dict) + a function with the same name as schema["name"]
    2. TOOL_SCHEMAS (list of dicts) + corresponding functions for each schema

    If neither is present, falls back to inferring schema from function signature.
    """
    _registry = get_registry()

    # tool_*.py files live in the parent package dir (jyagent/), not tools/
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(pkg_dir, "tool_*.py")

    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        module_name = filename[:-3]  # strip .py

        try:
            # Use package-qualified name so relative imports (from .xxx) work
            qualified_name = f"jyagent.{module_name}"
            spec = importlib.util.spec_from_file_location(
                qualified_name, filepath, submodule_search_locations=[])
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            # Ensure parent package is in sys.modules
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
                            "properties": strip_unsupported_schema_keys(properties),
                            "required": required
                        }
                    }
                    tools_to_register.append((tool_name, fn, schema))

            # Register all tools from this file
            for tname, fn, schema in tools_to_register:
                _registry.register(tname, fn, schema)

        except Exception as e:
            # Item 1: Never silently swallow discovery errors
            print(f"⚠️  Failed to load {filename}: {e}", file=sys.stderr)
