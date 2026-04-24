# Tool schemas — JSON schema definitions for all core tools.
# Separated from implementations so schemas are easy to review and edit.

CORE_TOOLS = [
    # --- run_shell ---
    {
        "name": "run_shell",
        "description": "Execute a shell command and return the output. Defaults to 60 seconds when timeout is omitted. Use timeout=600 for long-running commands such as agent CLIs (`claude -p`, `codex exec`, `codex review`), installs, builds, and test runs. If a 600-second command still times out, retry with a narrower scope instead of repeating the same broad command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 600,
                    "description": "Timeout in seconds (1-600). Defaults to 60 when omitted. Use timeout=600 for agent CLIs, installs, builds, test runs, and other long-running commands. If 600 seconds is still not enough, narrow the task and retry."
                }
            },
            "required": ["command"]
        }
    },
    # --- read_file ---
    {
        "name": "read_file",
        "description": "Read and return the contents of a file. Always read a file before editing it with edit_file. Use line_numbers=True to see exact content before making edits. Use offset/limit for large files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "1-based starting line number. Use with limit for pagination of large files. (default: 0 = from beginning)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 0,
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
        "description": "Write content to a file. Creates parent directories if needed. For modifying existing files, prefer edit_file — it saves tokens by not requiring full file content. Use write_file only for new files or complete rewrites.",
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
        "description": "List directory contents with tree-style output. Use this to understand project structure. For finding specific files by pattern, prefer glob_files instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (defaults to current directory if omitted)"
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Recursion depth (1 = immediate children, 2+ = nested). Default 1."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Max entries to return (default 200). Prevents context explosion on large dirs."
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Skip first N entries for pagination (0-based, default 0)."
                }
            },
            "required": []
        }
    },
    # --- manage_memory ---
    {
        "name": "manage_memory",
        "description": "Manage the agent's three-tier self-use memory system. TIERS: (1) MEMORY.md — always-loaded index, hard cap 200 lines / 25 KB; only durable, data-independent rules / facts that prevent future mistakes. (2) topics/<name>.md — curated on-demand knowledge files for extended detail (architecture, library quirks, ongoing projects). (3) journal/YYYY-MM.md — append-only chronological session notes; NEVER auto-loaded; the right home for 'what I did today' style entries. Actions: 'remember' (append durable rule to MEMORY.md — use sparingly), 'forget' (remove memories by keyword), 'show' (display all memories + size warnings), 'topic' (manage curated topic files: list/read/write/delete), 'goal' (add/complete a goal), 'note' (DEPRECATED alias → routes to journal), 'journal' (append a dated session note), 'consolidate' (read-only analysis: dedup candidates, oversized lines, dated notes that belong in journal).",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: remember | forget | show | topic | goal | note | journal | consolidate",
                    "enum": ["remember", "forget", "show", "topic", "goal", "note", "journal", "consolidate"]
                },
                "text": {
                    "type": "string",
                    "description": "Main text content. REQUIRED for: remember, forget, topic, goal, note, journal. NOT needed for: show, consolidate. For 'remember': the durable rule/fact (1 line, imperative). For 'journal'/'note': the chronological note (multi-line OK). For 'forget': keyword to match. For 'topic': command 'list' | 'read:<name>' | 'write:<name>|<content>' | 'delete:<name>'. For 'goal': goal text or 'done:<text>'."
                },
                "category": {
                    "type": "string",
                    "description": "Category tag. For 'remember': correction | preference | gotcha | tip | workflow | user_stated | goal. For 'journal'/'note': any short tag (default 'note'), e.g. session | debug | ship | research."
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
                    "description": "Skill name. REQUIRED for: activate, deactivate, info, create, delete, resources, read. Not needed for: list, reload."
                },
                "description": {
                    "type": "string",
                    "description": "Skill description. REQUIRED for 'create' action."
                },
                "instructions": {
                    "type": "string",
                    "description": "Skill instructions/body content. REQUIRED for 'create' action."
                },
                "resource_path": {
                    "type": "string",
                    "description": "Relative path to a resource file within the skill. REQUIRED for 'read' action."
                }
            },
            "required": ["action"]
        }
    },
    # --- edit_file ---
    {
        "name": "edit_file",
        "description": "Edit a file by replacing old_text with new_text (exact match), inserting at a line, appending, or creating. "
                       "More token-efficient than write_file for surgical edits. "
                       "IMPORTANT: Always use read_file with line_numbers=True first to see exact content and indentation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to edit"
                },
                "new_text": {
                    "type": "string",
                    "description": "The new text to insert (replacement, insertion, or file content)"
                },
                "old_text": {
                    "type": "string",
                    "description": "The existing text to find and replace (exact match). Leave empty for append/insert/create modes."
                },
                "operation": {
                    "type": "string",
                    "description": "Explicit edit operation. If omitted, mode is inferred from old_text/insert_at_line values.",
                    "enum": ["replace", "insert", "append", "create"]
                },
                "insert_at_line": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Insert new_text before this line number (1-based). 0=disabled. Use read_file with line_numbers=True to find line numbers."
                },
                "create_if_missing": {
                    "type": "boolean",
                    "description": "If true and file doesn't exist, create it with new_text (default false)"
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, show what would change without writing (default false)"
                }
            },
            "required": ["path", "new_text"]
        }
    },
    # --- glob_files ---
    {
        "name": "glob_files",
        "description": "Find files matching a glob pattern recursively. Use when you need to discover files by name or extension. "
                       "For searching file contents, use grep_files instead. "
                       "Skips binary files and common ignore patterns (.git, node_modules, etc).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '*.py', '**/*.ts', 'src/**/*.js')"
                },
                "path": {
                    "type": "string",
                    "description": "Root directory to search from (default: current dir)"
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of results (default 200)"
                }
            },
            "required": ["pattern"]
        }
    },
    # --- grep_files ---
    {
        "name": "grep_files",
        "description": "Search for text or regex in files. Returns matches with file paths and line numbers. "
                       "Use output_mode='files_only' when you only need to know which files match. "
                       "More efficient than run_shell('grep ...') for code search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text or regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Root directory or file to search (default: current dir)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Only search files matching this glob (e.g., '*.py', '*.js')"
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of matching lines (default 50)"
                },
                "context_lines": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of context lines before/after each match (default 0)"
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default false)"
                },
                "output_mode": {
                    "type": "string",
                    "description": "Output format: 'content' (match lines, default), 'files_only' (file paths only), 'count' (match counts per file)",
                    "enum": ["content", "files_only", "count"]
                }
            },
            "required": ["pattern"]
        }
    },
    # --- run_background ---
    {
        "name": "run_background",
        "description": "Start a long-running shell command in the background and return immediately. Use this instead of run_shell when a command may exceed 600 seconds (e.g., slow agent CLIs like `codex exec`, large builds, long test suites). Returns {pid, output_file, status:'started'}. Output (stdout+stderr merged) is streamed to output_file — use check_background(pid, tail=N) to poll progress; do NOT `cat` the file via run_shell. Commands run with stdin redirected from /dev/null by default (noninteractive). A global cap of 8 concurrent background jobs is enforced.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run in the background"
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Hard deadline in seconds. If >0, the job is auto-killed when it elapses and status becomes 'timed_out'. 0 (default) = no deadline. Use this for any untrusted long-running command — do not rely on remembering to kill."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command. Empty (default) = inherit. Prefer this over `cd X && cmd` shell tricks."
                },
                "stdin_null": {
                    "type": "boolean",
                    "description": "Redirect stdin from /dev/null so the command cannot hang on an interactive prompt. Default: true. Set false only if the child must inherit the parent's stdin."
                }
            },
            "required": ["command"]
        }
    },
    # --- check_background ---
    {
        "name": "check_background",
        "description": "Check status and read output of a background process started by run_background. Returns {pid, status, exit_code, elapsed_seconds, command, output_file, output}. Status values: 'running' | 'succeeded' (exit 0) | 'failed' (exit != 0) | 'killed' (caller signaled it) | 'timed_out' (auto-killed at deadline). Always inspect exit_code — 'succeeded' ≠ 'the task did what you wanted'. Use tail=N (20–50) while polling to avoid flooding context; use tail=0 only for final collection. action='wait' blocks up to wait_timeout_seconds for the job to finish — prefer this over tight polling loops (each poll costs a model turn). action='kill' SIGTERMs the process group (no-op if already exited; status reflects the real outcome, not the request).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pid": {
                    "type": "integer",
                    "description": "Process ID returned by run_background"
                },
                "tail": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Return only the last N lines of output (bounded backward scan). 0 = last ~50 KB of the file (default). Use 20–50 while polling a running process."
                },
                "action": {
                    "type": "string",
                    "enum": ["status", "wait", "kill"],
                    "description": "Action to take: 'status' (default) returns current progress; 'wait' blocks up to wait_timeout_seconds for completion then returns; 'kill' terminates the process group (SIGTERM, then SIGKILL after 5s)."
                },
                "wait_timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Only used with action='wait'. Max seconds to block waiting for the job to finish. Default 60, hard cap 300."
                }
            },
            "required": ["pid"]
        }
    },
]
