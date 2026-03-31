# Tool schemas — JSON schema definitions for all core tools.
# Separated from implementations so schemas are easy to review and edit.

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
                    "description": "Action: remember, forget, show, topic, goal, note",
                    "enum": ["remember", "forget", "show", "topic", "goal", "note"]
                },
                "text": {
                    "type": "string",
                    "description": "Main text content (what to remember, keyword to forget, topic command like 'read:chrome-mcp', etc.)"
                },
                "category": {
                    "type": "string",
                    "description": "Category for learnings: correction, preference, gotcha, tip, workflow, user_stated, goal, note"
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
                    "description": "Action: list, activate, deactivate, info, create, delete, resources, read, reload",
                    "enum": ["list", "activate", "deactivate", "info", "create", "delete", "resources", "read", "reload"]
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
