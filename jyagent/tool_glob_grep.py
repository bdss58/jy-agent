"""
glob_files / grep_files — Specialized search tools for code navigation.

More token-efficient than `run_shell("find ...")` or `run_shell("grep ...")`:
- Structured output with line numbers
- Automatic binary file skipping
- Result count limits to prevent context explosion
- Gitignore-aware (skips .git, node_modules, __pycache__, etc.)
- glob_files shows file sizes and modification times
"""

import os
import re
import fnmatch
import time

try:
    from .config import SKIP_DIRS, BINARY_EXTS
    from .tools.core import _format_size
except ImportError:
    # Fallback for when loaded via auto-discovery with different package context
    from jyagent.config import SKIP_DIRS, BINARY_EXTS
    from jyagent.tools.core import _format_size


def _should_skip_dir(dirname: str) -> bool:
    """Check if a directory should be skipped."""
    return dirname in SKIP_DIRS or dirname.startswith('.')


def _is_binary(filepath: str) -> bool:
    """Check if a file is likely binary based on extension."""
    _, ext = os.path.splitext(filepath)
    return ext.lower() in BINARY_EXTS


def _format_mtime(mtime: float) -> str:
    """Format modification time as relative or absolute."""
    now = time.time()
    delta = now - mtime
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    elif delta < 86400:
        return f"{int(delta / 3600)}h ago"
    elif delta < 86400 * 30:
        return f"{int(delta / 86400)}d ago"
    else:
        return time.strftime("%Y-%m-%d", time.localtime(mtime))


def glob_files(pattern: str, path: str = ".", max_results: int = 200) -> str:
    """Find files matching a glob pattern. Shows file size and modification time.

    Searches recursively from `path`. Supports standard glob patterns:
    - "*.py" — all Python files
    - "**/*.test.js" — all test.js files in any subdirectory
    - "src/**/*.ts" — TypeScript files under src/
    - "config.*" — any file named config with any extension

    Args:
        pattern: Glob pattern to match (e.g., "*.py", "**/*.ts")
        path: Root directory to search from (default: current dir)
        max_results: Maximum number of results (default 200)
    """
    try:
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return f"Error: Directory not found: {path}"

        matches = []  # list of (rel_path, size, mtime)
        is_recursive = '**' in pattern

        for root, dirs, files in os.walk(path):
            # Prune skipped directories
            dirs[:] = [d for d in dirs if not _should_skip_dir(d)]

            rel_root = os.path.relpath(root, path)
            if rel_root == '.':
                rel_root = ''

            for filename in sorted(files):
                if _is_binary(filename):
                    continue

                rel_path = os.path.join(rel_root, filename) if rel_root else filename
                matched = False

                if is_recursive:
                    if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(filename, pattern.replace('**/', '')):
                        matched = True
                else:
                    if fnmatch.fnmatch(filename, pattern):
                        matched = True

                if matched:
                    full_path = os.path.join(root, filename)
                    try:
                        stat = os.stat(full_path)
                        matches.append((rel_path, stat.st_size, stat.st_mtime))
                    except OSError:
                        matches.append((rel_path, 0, 0))

                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        if not matches:
            return f"No files matching '{pattern}' found in {path}"

        # Format output with size and mtime
        lines = []
        for rel_path, size, mtime in matches:
            size_str = _format_size(size)
            mtime_str = _format_mtime(mtime) if mtime > 0 else "?"
            lines.append(f"  {rel_path}  ({size_str}, {mtime_str})")

        result = f"Found {len(matches)} files matching '{pattern}':\n"
        result += '\n'.join(lines)
        if len(matches) >= max_results:
            result += f"\n  ... (truncated at {max_results} results)"
        return result

    except Exception as e:
        return f"Error: {e}"


def grep_files(
    pattern: str,
    path: str = ".",
    file_pattern: str = "",
    max_results: int = 50,
    context_lines: int = 0,
    ignore_case: bool = False,
) -> str:
    """Search for a text/regex pattern in files.

    More efficient than `run_shell("grep ...")` — skips binary files,
    respects common ignore patterns, and formats output for LLM consumption.

    Args:
        pattern: Text or regex pattern to search for
        path: Root directory or single file to search (default: current dir)
        file_pattern: Only search files matching this glob (e.g., "*.py", "*.js")
        max_results: Maximum number of matching lines (default 50)
        context_lines: Number of context lines before/after each match (default 0)
        ignore_case: Case-insensitive search (default false)
    """
    try:
        path = os.path.abspath(path)
        flags = re.IGNORECASE if ignore_case else 0

        try:
            regex = re.compile(pattern, flags)
        except re.error:
            # Fall back to literal search if regex is invalid
            regex = re.compile(re.escape(pattern), flags)

        results = []
        files_searched = 0
        files_with_matches = 0

        if os.path.isfile(path):
            file_results = _search_file(path, regex, max_results, context_lines)
            if file_results:
                results.extend(file_results)
                files_with_matches = 1
            files_searched = 1
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not _should_skip_dir(d)]

                for filename in sorted(files):
                    if _is_binary(filename):
                        continue
                    if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                        continue

                    filepath = os.path.join(root, filename)
                    file_results = _search_file(
                        filepath, regex,
                        max_results - len(results),
                        context_lines,
                        show_path=os.path.relpath(filepath, path)
                    )
                    files_searched += 1
                    if file_results:
                        results.extend(file_results)
                        files_with_matches += 1

                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break
        else:
            return f"Error: Path not found: {path}"

        if not results:
            extra = f" in {file_pattern} files" if file_pattern else ""
            return f"No matches for '{pattern}'{extra} ({files_searched} files searched)"

        header = (
            f"Found {len(results)} matches in {files_with_matches} files "
            f"({files_searched} searched):\n"
        )
        body = '\n'.join(results)
        if len(results) >= max_results:
            body += f"\n... (truncated at {max_results} results)"
        return header + body

    except Exception as e:
        return f"Error: {e}"


def _search_file(filepath, regex, max_results, context_lines, show_path=None):
    """Search a single file for regex matches. Returns list of formatted result lines."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return []

    display_path = show_path or filepath
    results = []
    matched_lines = set()

    # First pass: find all matching line indices
    match_indices = []
    for i, line in enumerate(lines):
        if regex.search(line):
            match_indices.append(i)
            if len(match_indices) >= max_results:
                break

    if not match_indices:
        return []

    # Second pass: collect matches with context
    for idx in match_indices:
        ctx_start = max(0, idx - context_lines)
        ctx_end = min(len(lines), idx + context_lines + 1)

        for i in range(ctx_start, ctx_end):
            if i in matched_lines:
                continue
            matched_lines.add(i)

            line_content = lines[i].rstrip('\n\r')
            # Truncate very long lines, showing the match area
            if len(line_content) > 200:
                match = regex.search(line_content)
                if match and i == idx:
                    # Center around the match
                    start = max(0, match.start() - 80)
                    end = min(len(line_content), match.end() + 80)
                    line_content = ('...' if start > 0 else '') + line_content[start:end] + ('...' if end < len(line_content) else '')
                else:
                    line_content = line_content[:200] + '...'

            prefix = ">" if i == idx else " "
            results.append(f"  {prefix} {display_path}:L{i + 1}: {line_content}")

        # Add separator between non-adjacent match groups
        if context_lines > 0 and idx != match_indices[-1]:
            next_idx = match_indices[match_indices.index(idx) + 1]
            if next_idx - idx > 2 * context_lines + 1:
                results.append("  ---")

    return results


# ─── Tool schemas ─────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "glob_files",
        "description": "Find files matching a glob pattern recursively. Shows file sizes and modification times. Skips binary files and common ignore patterns (.git, node_modules, etc). Use when you need to discover files in a project.",
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
                    "description": "Maximum number of results (default 200)"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "grep_files",
        "description": "Search for a text or regex pattern in files. Returns matches with file paths and line numbers. Skips binary files and common ignore patterns. More efficient than run_shell('grep ...') for code search.",
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
                    "description": "Maximum number of matching lines (default 50)"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines before/after each match (default 0)"
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default false)"
                }
            },
            "required": ["pattern"]
        }
    },
]

TOOL_SCHEMA = TOOL_SCHEMAS[0]
