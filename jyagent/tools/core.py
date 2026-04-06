# Core file/shell tools and shared path/file helpers.

import os
import difflib
import fnmatch
import subprocess
import tempfile

from ..config import SKIP_DIRS, BINARY_EXTS
from ..toolresult import ToolResult

_SKIP_EXACT: set[str] = set()
_SKIP_PATTERNS: list[str] = []
for _entry in SKIP_DIRS:
    if any(char in _entry for char in ("*", "?", "[")):
        _SKIP_PATTERNS.append(_entry)
    else:
        _SKIP_EXACT.add(_entry)


def resolve_path(path: str, root: str | None = None) -> str:
    """Resolve a path to absolute, expanding ~ and relative segments."""
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(root or os.getcwd(), path)
    return os.path.abspath(path)


def atomic_write(path: str, content: str, encoding: str = "utf-8") -> None:
    """Write content to a temp file, then atomically replace the target."""
    dirname = os.path.dirname(path) or "."
    os.makedirs(dirname, exist_ok=True)

    fd = None
    tmp_path = None
    try:
        fd_int, tmp_path = tempfile.mkstemp(dir=dirname, prefix=".tmp_", suffix=".write")
        fd = os.fdopen(fd_int, "w", encoding=encoding)
        fd.write(content)
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        fd = None
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if fd is not None:
            try:
                fd.close()
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def should_skip_dir(dirname: str) -> bool:
    """Return True when traversal should skip this directory name."""
    if dirname.startswith("."):
        return True
    if dirname in _SKIP_EXACT:
        return True
    return any(fnmatch.fnmatch(dirname, pattern) for pattern in _SKIP_PATTERNS)


def is_binary_ext(path: str) -> bool:
    """Return True if the path has a configured binary extension."""
    _, ext = os.path.splitext(path)
    return ext.lower() in BINARY_EXTS


def run_shell(command: str, timeout: int = 60) -> ToolResult:
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
        return ToolResult(output, is_error=(result.returncode != 0))
    except subprocess.TimeoutExpired:
        return ToolResult(f"Error: Command timed out after {effective_timeout} seconds", is_error=True)
    except Exception as e:
        return ToolResult(f"Error: {e}", is_error=True)


def read_file(path: str, offset: int = 0, limit: int = 0, line_numbers: bool = False) -> ToolResult:
    """Read and return the contents of a file with optional pagination.

    - offset: 1-based starting line number (0 = from beginning)
    - limit: max lines to return (0 = all lines, subject to char cap)
    - line_numbers: prepend "L{n}: " to each line
    """
    MAX_READ_CHARS = 50000
    try:
        path = resolve_path(path)
        if not os.path.exists(path):
            return ToolResult(f"Error: File not found: {path}", is_error=True)

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

        return ToolResult(result)
    except Exception as e:
        return ToolResult(f"Error: {e}", is_error=True)


def write_file(path: str, content: str) -> ToolResult:
    """Write content to a file. Creates parent directories if needed.

    For surgical edits (replacing specific text), prefer edit_file instead —
    it saves tokens by not requiring the full file content.
    """
    try:
        path = resolve_path(path)
        # Capture old file stats for overwrite summary
        old_exists = os.path.exists(path)
        old_chars = 0
        old_lines = 0
        if old_exists:
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    old_content = f.read()
                old_chars = len(old_content)
                old_lines = old_content.count('\n') + (1 if old_content and not old_content.endswith('\n') else 0)
            except Exception:
                pass  # proceed with write even if read fails

        atomic_write(path, content)

        new_lines = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
        new_chars = len(content)

        if old_exists and old_chars > 0:
            line_delta = new_lines - old_lines
            char_delta = new_chars - old_chars
            delta_str = f"{'+' if line_delta >= 0 else ''}{line_delta} lines, {'+' if char_delta >= 0 else ''}{char_delta} chars"
            return ToolResult(f"Overwrote {path}: {old_lines}→{new_lines} lines, {old_chars}→{new_chars} chars ({delta_str})")
        else:
            return ToolResult(f"Created {path}: {new_chars} chars ({new_lines} lines)")
    except Exception as e:
        return ToolResult(f"Error: {e}", is_error=True)


def list_directory(path: str = ".", depth: int = 1, limit: int = 200, offset: int = 0) -> ToolResult:
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
            return ToolResult(f"Error: Not a directory: {path}", is_error=True)

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
                    if should_skip_dir(item):
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
                    if current_depth < depth and not os.path.islink(item_path):
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
            return ToolResult(header + "  (empty directory)")

        result = header + '\n'.join(paginated)
        if total_entries > offset + limit:
            result += f"\n\n[... {total_entries - offset - limit} more entries. Use offset={offset + limit} to see next page]"

        return ToolResult(result)
    except Exception as e:
        return ToolResult(f"Error: {e}", is_error=True)


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


# ─── edit_file ────────────────────────────────────────────────────────────────

def edit_file(
    path: str,
    new_text: str,
    old_text: str = "",
    operation: str = "",
    insert_at_line: int = 0,
    create_if_missing: bool = False,
    dry_run: bool = False,
) -> ToolResult:
    """Edit a file by replacing old_text with new_text, or insert at a line number.

    Modes (explicit via *operation*, or inferred from parameters):
    - replace: provide both old_text and new_text to replace a specific block
    - insert: set insert_at_line=N to insert new_text before line N (1-based)
    - append: provide only new_text to append to file
    - create: create a new file with new_text as content

    Args:
        path: File path to edit
        new_text: The new text to insert (replacement, insertion, or file content)
        old_text: The existing text to find and replace (exact match required).
                  Leave empty for append/insert/create modes.
        operation: Explicit edit operation. If omitted, mode is inferred.
        insert_at_line: Insert new_text before this line number (1-based). 0 = disabled.
        create_if_missing: If True and file doesn't exist, create it with new_text
        dry_run: If True, show what would change without actually writing
    """
    path = resolve_path(path)

    # --- Validate explicit operation ---
    if operation:
        _valid_ops = ("replace", "insert", "append", "create")
        if operation not in _valid_ops:
            return ToolResult(f"Error: Unknown operation '{operation}'. Must be one of {_valid_ops}", is_error=True)
        if operation == "replace" and not old_text:
            return ToolResult("Error: operation='replace' requires old_text", is_error=True)
        if operation == "insert" and insert_at_line <= 0:
            return ToolResult("Error: operation='insert' requires insert_at_line > 0", is_error=True)
        # Map explicit operation to existing parameter semantics
        if operation == "create":
            create_if_missing = True
        if operation == "append":
            old_text = ""
            insert_at_line = 0

    # --- Create mode ---
    if not os.path.exists(path):
        if create_if_missing:
            if dry_run:
                return ToolResult(f"[DRY RUN] Would create {path} ({len(new_text)} chars)")
            atomic_write(path, new_text)
            lines = new_text.count('\n') + (1 if new_text and not new_text.endswith('\n') else 0)
            return ToolResult(f"Created {path} ({lines} lines, {len(new_text)} chars)")
        else:
            return ToolResult(f"Error: File not found: {path} (set create_if_missing=True to create)", is_error=True)

    # --- Binary guard ---
    if is_binary_ext(path):
        return ToolResult(f"Error: Cannot edit binary file: {path}", is_error=True)

    # --- Read existing file ---
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return ToolResult(f"Error: {path} contains non-UTF-8 bytes. Cannot safely edit. Use read_file to inspect.", is_error=True)
    except Exception as e:
        return ToolResult(f"Error reading {path}: {e}", is_error=True)

    # --- Insert at line mode ---
    if insert_at_line > 0:
        all_lines = content.split('\n')
        # insert_at_line is 1-based; inserting before that line
        idx = max(0, min(insert_at_line - 1, len(all_lines)))

        # Ensure new_text ends with newline for clean insertion
        insert_text = new_text if new_text.endswith('\n') else new_text + '\n'
        insert_lines = insert_text.split('\n')
        # split produces trailing empty element for trailing \n
        if insert_lines and insert_lines[-1] == '':
            insert_lines = insert_lines[:-1]
        n_inserted = len(insert_lines)

        if dry_run:
            return ToolResult(
                f"[DRY RUN] Would insert {n_inserted} lines before L{insert_at_line} in {path}\n"
                f"  File currently has {len(all_lines)} lines"
            )

        new_lines = all_lines[:idx] + insert_lines + all_lines[idx:]
        new_content = '\n'.join(new_lines)
        atomic_write(path, new_content)

        # Show context
        ctx_start = max(0, idx - 1)
        ctx_end = min(len(new_lines), idx + n_inserted + 1)
        context = '\n'.join(
            f"  {'>' if idx <= j < idx + n_inserted else ' '} "
            f"L{j + 1}: {new_lines[j]}"
            for j in range(ctx_start, ctx_end)
        )

        return ToolResult(
            f"Inserted {n_inserted} lines before L{insert_at_line} in {path} "
            f"(now {len(new_lines)} lines)\n{context}"
        )

    # --- Append mode ---
    if old_text == "":
        if dry_run:
            return ToolResult(f"[DRY RUN] Would append {len(new_text)} chars to {path}")
        new_content = content + new_text
        atomic_write(path, new_content)
        added_lines = new_text.count('\n') + (1 if new_text and not new_text.endswith('\n') else 0)
        total_lines = new_content.count('\n') + 1
        return ToolResult(f"Appended to {path} (+{added_lines} lines, total {total_lines} lines)")

    # --- Replace mode (str_replace) ---
    count = content.count(old_text)
    if count == 0:
        return _diagnose_no_match(path, content, old_text)

    if count > 1:
        return _diagnose_multi_match(path, content, old_text, count)

    # Exactly one match — perform the replacement
    new_content = content.replace(old_text, new_text, 1)

    # Calculate edit stats
    old_line_count = old_text.count('\n') + 1
    new_line_count = new_text.count('\n') + 1
    delta = new_line_count - old_line_count

    # Find the line number of the edit
    edit_pos = content.index(old_text)
    edit_line = content[:edit_pos].count('\n') + 1

    if dry_run:
        return ToolResult(
            f"[DRY RUN] Would edit {path} at L{edit_line}:\n"
            f"  - Remove {old_line_count} lines\n"
            f"  + Add {new_line_count} lines\n"
            f"  Net: {'+' if delta > 0 else ''}{delta} lines"
        )

    atomic_write(path, new_content)

    total_lines = new_content.count('\n') + 1
    delta_str = f"+{delta}" if delta > 0 else str(delta)

    # Show context around the edit
    result_lines = new_content.split('\n')
    ctx_start = max(0, edit_line - 2)
    ctx_end = min(len(result_lines), edit_line + new_line_count + 1)
    context = '\n'.join(
        f"  {'>' if edit_line <= (j + 1) <= edit_line + new_line_count - 1 else ' '} "
        f"L{j + 1}: {result_lines[j]}"
        for j in range(ctx_start, ctx_end)
    )

    return ToolResult(
        f"Edited {path}: replaced {old_line_count} lines with {new_line_count} lines "
        f"at L{edit_line} ({delta_str}), total {total_lines} lines\n{context}"
    )


def _diagnose_no_match(path: str, content: str, old_text: str) -> ToolResult:
    """Provide helpful diagnostics when old_text is not found, including fuzzy matching."""
    old_stripped = old_text.strip()

    # Check for whitespace/indentation mismatch
    if old_stripped and old_stripped in content:
        all_lines = content.split('\n')
        for i, line in enumerate(all_lines):
            if old_stripped[:40] in line:
                start = max(0, i - 1)
                end = min(len(all_lines), i + 4)
                snippet = '\n'.join(
                    f"  L{start + 1 + j}: {all_lines[start + j]}"
                    for j in range(end - start)
                )
                return ToolResult(
                    f"Error: Exact match not found in {path}.\n"
                    f"Content exists but with different whitespace/indentation.\n"
                    f"Nearby lines:\n{snippet}\n\n"
                    f"Hint: Copy the exact text including indentation. "
                    f"Use read_file with line_numbers=True to see exact content.",
                    is_error=True
                )

    # Fuzzy matching: find the most similar block in the file
    old_lines = old_text.split('\n')
    n_lines = len(old_lines)
    all_lines = content.split('\n')

    if n_lines <= len(all_lines) and n_lines > 0:
        best_ratio = 0.0
        best_start = 0
        for i in range(len(all_lines) - n_lines + 1):
            candidate = '\n'.join(all_lines[i:i + n_lines])
            ratio = difflib.SequenceMatcher(None, old_text, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        if best_ratio >= 0.6:
            snippet_lines = all_lines[best_start:best_start + n_lines]
            snippet = '\n'.join(
                f"  L{best_start + 1 + j}: {snippet_lines[j]}"
                for j in range(len(snippet_lines))
            )
            return ToolResult(
                f"Error: Exact match not found in {path}.\n"
                f"Most similar block ({best_ratio:.0%} match) at L{best_start + 1}:\n{snippet}\n\n"
                f"Hint: Use read_file with line_numbers=True to copy the exact text.",
                is_error=True
            )

    # Check for partial first-line match
    first_line_candidates = old_text.strip().split('\n')
    if len(first_line_candidates) > 1:
        first_line = first_line_candidates[0].strip()
        if first_line and first_line in content:
            return ToolResult(
                f"Error: Exact match not found in {path}.\n"
                f"First line found ('{first_line[:60]}') but full block doesn't match.\n"
                f"Hint: Use read_file with line_numbers=True to check exact content.",
                is_error=True
            )

    return ToolResult(
        f"Error: old_text not found in {path}.\n"
        f"Searched for ({len(old_text)} chars): {repr(old_text[:100])}{'...' if len(old_text) > 100 else ''}\n"
        f"Hint: Use read_file with line_numbers=True to verify the current content.",
        is_error=True
    )


def _diagnose_multi_match(path: str, content: str, old_text: str, count: int) -> ToolResult:
    """Provide helpful diagnostics when old_text appears multiple times."""
    positions = []
    search_from = 0
    for _ in range(count):
        idx = content.index(old_text, search_from)
        line_num = content[:idx].count('\n') + 1
        positions.append(line_num)
        search_from = idx + 1

    return ToolResult(
        f"Error: old_text appears {count} times in {path} (at lines {positions}).\n"
        f"Include more surrounding context in old_text to make it unique.\n"
        f"Hint: Add a few lines before/after the target to disambiguate.",
        is_error=True
    )
