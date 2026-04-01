# Tests for tool registration and discovery

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.registry import get_registry
import jyagent.tools  # noqa: F401 — ensure tools are registered for these tests


class TestRegistry:
    def test_core_tools_registered(self):
        reg = get_registry()
        tools = reg.list_tools()
        # All tools should be registered
        expected = [
            'run_shell', 'read_file', 'write_file', 'list_directory',
            'edit_file', 'glob_files', 'grep_files',
            'manage_memory', 'manage_skills',
            'web_fetch', 'mcp',
        ]
        for name in expected:
            assert name in tools, f"Tool '{name}' not registered"

    def test_get_function(self):
        reg = get_registry()
        fn = reg.get_function('run_shell')
        assert fn is not None
        assert callable(fn)

    def test_get_schemas(self):
        reg = get_registry()
        schemas = reg.get_schemas()
        assert len(schemas) > 0
        # Each schema should have name and input_schema
        for schema in schemas:
            assert 'name' in schema
            assert 'input_schema' in schema

    def test_schema_has_description(self):
        reg = get_registry()
        schemas = reg.get_schemas()
        for schema in schemas:
            assert 'description' in schema, f"Schema for '{schema['name']}' missing description"
            assert len(schema['description']) > 0

    def test_register_unregister(self):
        reg = get_registry()
        # Register a test tool
        reg.register("_test_tool", lambda: "test", {
            "name": "_test_tool",
            "description": "test",
            "input_schema": {"type": "object", "properties": {}}
        })
        assert "_test_tool" in reg.list_tools()
        
        # Unregister
        assert reg.unregister("_test_tool") is True
        assert "_test_tool" not in reg.list_tools()
        assert reg.unregister("_test_tool") is False  # already gone
