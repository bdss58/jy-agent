# Search tools: glob_files, grep_files
#
# More token-efficient than `run_shell("find ...")` or `run_shell("grep ...")`:
# - Structured output with line numbers
# - Automatic binary file skipping
# - Result count limits to prevent context explosion
# - Skips .git, node_modules, __pycache__, *.egg-info, etc.
# - glob_files shows file sizes and modification times

import os
import re
import glob as _glob
import fnmatch
import time

from ..toolresult import ToolResult
from .core import _format_size, is_binary_ext, should_skip_dir


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


def glob_files(pattern: str, path: str = ".", max_results: int = 200) -> ToolResult:
    """Find files matching a glob pattern. Shows file size and modification time.

    Searches recursively from `path`. Supports standard glob patterns:
    - "*.py" — all Python files in root dir
    - "**/*.py" — all Python files in any subdirectory
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
            return ToolResult(f"Error: Directory not found: {path}", is_error=True)

        matches = []  # list of (rel_path, size, mtime)
        is_recursive = '**' in pattern

        for rel_path in _glob.iglob(pattern, root_dir=path, recursive=is_recursive):
            # Skip binary files
            if is_binary_ext(rel_path):
                continue

            # Skip entries under directories that should be skipped
            parts = rel_path.replace(os.sep, '/').split('/')
            if any(should_skip_dir(p) for p in parts[:-1]):
                continue

            full_path = os.path.join(path, rel_path)

            # Only include files, not directories
            if not os.path.isfile(full_path):
                continue

            try:
                stat = os.stat(full_path)
                matches.append((rel_path, stat.st_size, stat.st_mtime))
            except OSError:
                matches.append((rel_path, 0, 0))

            if len(matches) >= max_results:
                break

        if not matches:
            return ToolResult(f"No files matching '{pattern}' found in {path}")

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
        return ToolResult(result)

    except Exception as e:
        return ToolResult(f"Error: {e}", is_error=True)


def grep_files(
    pattern: str,
    path: str = ".",
    file_pattern: str = "",
    max_results: int = 50,
    context_lines: int = 0,
    ignore_case: bool = False,
    output_mode: str = "content",
) -> ToolResult:
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
        output_mode: "content" (match lines), "files_only" (file paths), "count" (per-file counts)
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
        file_match_counts = []  # for count mode: (rel_path, count)

        if os.path.isfile(path):
            if output_mode == "content":
                file_results = _search_file(path, regex, max_results, context_lines)
                if file_results:
                    results.extend(file_results)
                    files_with_matches = 1
            else:
                count = _count_matches(path, regex)
                if count > 0:
                    files_with_matches = 1
                    file_match_counts.append((os.path.basename(path), count))
            files_searched = 1
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not should_skip_dir(d)]

                for filename in sorted(files):
                    if is_binary_ext(filename):
                        continue
                    if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                        continue

                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, path)
                    files_searched += 1

                    if output_mode == "content":
                        file_results = _search_file(
                            filepath, regex,
                            max_results - len(results),
                            context_lines,
                            show_path=rel_path
                        )
                        if file_results:
                            results.extend(file_results)
                            files_with_matches += 1
                        if len(results) >= max_results:
                            break
                    else:
                        count = _count_matches(filepath, regex)
                        if count > 0:
                            files_with_matches += 1
                            file_match_counts.append((rel_path, count))
                        if files_with_matches >= max_results:
                            break
                if output_mode == "content" and len(results) >= max_results:
                    break
                if output_mode != "content" and files_with_matches >= max_results:
                    break
        else:
            return ToolResult(f"Error: Path not found: {path}", is_error=True)

        # --- Format output based on mode ---
        if output_mode == "files_only":
            if not file_match_counts:
                fp_note = f" (filtered to {file_pattern})" if file_pattern else ""
                return ToolResult(f"No matches for '{pattern}' in {files_searched} files searched{fp_note}")
            header = f"Found matches in {files_with_matches} files ({files_searched} searched):\n"
            body = '\n'.join(f"  {rel}" for rel, _ in file_match_counts)
            return ToolResult(header + body)

        if output_mode == "count":
            if not file_match_counts:
                fp_note = f" (filtered to {file_pattern})" if file_pattern else ""
                return ToolResult(f"No matches for '{pattern}' in {files_searched} files searched{fp_note}")
            total = sum(c for _, c in file_match_counts)
            header = f"Found {total} matches in {files_with_matches} files ({files_searched} searched):\n"
            body = '\n'.join(f"  {rel}: {c}" for rel, c in file_match_counts)
            return ToolResult(header + body)

        # content mode (default)
        if not results:
            fp_note = f" (filtered to {file_pattern})" if file_pattern else ""
            return ToolResult(f"No matches for '{pattern}' in {files_searched} files searched{fp_note}")

        header = f"Found {len(results)} matches in {files_with_matches} files ({files_searched} searched):\n"
        body = '\n'.join(results)
        if len(results) >= max_results:
            body += f"\n  ... (truncated at {max_results} results)"

        return ToolResult(header + body)

    except Exception as e:
        return ToolResult(f"Error: {e}", is_error=True)


def _count_matches(filepath: str, regex) -> int:
    """Count regex matches in a file without storing content."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return sum(1 for line in f if regex.search(line))
    except (PermissionError, OSError):
        return 0


def _search_file(
    filepath: str,
    regex,
    max_results: int,
    context_lines: int,
    show_path: str = None,
) -> list:
    """Search a single file for regex matches. Returns formatted match lines.

    When context_lines == 0, streams line-by-line to avoid loading the
    entire file into memory.
    """
    if show_path is None:
        show_path = filepath

    try:
        f = open(filepath, 'r', encoding='utf-8', errors='replace')
    except (PermissionError, OSError):
        return []

    results = []

    # Fast path: no context needed — stream line-by-line
    if context_lines == 0:
        try:
            for i, line in enumerate(f):
                if regex.search(line):
                    results.append(
                        f"    {show_path}:L{i + 1}: {line.rstrip()}"
                    )
                    if len(results) >= max_results:
                        break
        finally:
            f.close()
        return results

    # Context mode: need random access, load all lines
    try:
        lines = f.readlines()
    finally:
        f.close()

    shown_context = set()  # Track context line numbers already shown

    for i, line in enumerate(lines):
        if regex.search(line):
            ctx_start = max(0, i - context_lines)
            ctx_end = min(len(lines), i + context_lines + 1)

            # Add separator between non-contiguous context blocks
            if results and ctx_start > 0:
                last_shown = max(shown_context) if shown_context else -1
                if ctx_start > last_shown + 1:
                    results.append("  ---")

            for j in range(ctx_start, ctx_end):
                if j not in shown_context:
                    shown_context.add(j)
                    prefix = "  >" if j == i else "   "
                    results.append(
                        f"{prefix} {show_path}:L{j + 1}: {lines[j].rstrip()}"
                    )

            if len(results) >= max_results:
                break

    return results
