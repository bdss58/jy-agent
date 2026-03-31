# Core file/shell tools: run_shell, read_file, write_file, list_directory

import os
import subprocess

from ..config import SKIP_DIRS, BINARY_EXTS


def run_shell(command: str, timeout: int = 60) -> str:
    """Execute a shell command and return the output."""
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
    """Read and return the contents of a file with optional pagination.
    
    - offset: 1-based starting line number (0 = from beginning)
    - limit: max lines to return (0 = all lines, subject to char cap)
    - line_numbers: prepend "L{n}: " to each line
    """
    MAX_READ_CHARS = 50000
    try:
        if not os.path.exists(path):
            return f"Error: File not found: {path}"

        _, ext = os.path.splitext(path)
        if ext.lower() in BINARY_EXTS:
            size = os.path.getsize(path)
            return f"[Binary file: {path} ({size} bytes)]"

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        total_chars = sum(len(l) for l in all_lines)

        if offset > 0:
            start = max(0, offset - 1)
        else:
            start = 0

        if limit > 0:
            end = min(start + limit, total_lines)
        else:
            end = total_lines

        selected_lines = all_lines[start:end]

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
                line_text = line.rstrip('\n')
                output_parts.append(f"L{line_num}: {line_text}")
            content = '\n'.join(output_parts)
        else:
            content = ''.join(selected_lines)

        result = header + content

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
    
    - depth: recursion depth (1 = immediate children, 2+ = nested, default 1)
    - limit: max entries to return (default 200, prevents context explosion)
    - offset: skip first N entries for pagination (0-based, default 0)
    - Shows file sizes, symlink markers, and directory suffixes
    - Automatically skips .git, node_modules, __pycache__, etc.
    """
    try:
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return f"Error: Not a directory: {path}"

        entries = []

        def _walk(dir_path, current_depth, prefix=""):
            if current_depth > depth:
                return
            try:
                items = sorted(os.listdir(dir_path))
            except PermissionError:
                entries.append(f"{prefix}[permission denied]")
                return

            dirs_list = []
            files_list = []
            for item in items:
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    if item in SKIP_DIRS:
                        continue
                    dirs_list.append(item)
                else:
                    files_list.append(item)

            all_items = [(d, True) for d in dirs_list] + [(f, False) for f in files_list]

            for i, (item, is_dir) in enumerate(all_items):
                item_path = os.path.join(dir_path, item)
                is_last = (i == len(all_items) - 1)

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
                    if current_depth < depth:
                        _walk(item_path, current_depth + 1, child_prefix)
                else:
                    try:
                        size = os.path.getsize(item_path)
                        size_str = _format_size(size)
                    except OSError:
                        size_str = "?"
                    link_marker = "@ " if os.path.islink(item_path) else ""
                    entries.append(f"{prefix}{connector}{link_marker}{item}  ({size_str})")

        _walk(path, 1)

        total_entries = len(entries)
        paginated = entries[offset:offset + limit]

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
