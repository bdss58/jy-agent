# Core file/shell tools and shared path/file helpers.
#
# TODO: split this module into ``core`` and
# ``background`` (the run_background / check_background / _bg_* helpers
# stack accounts for ~half the file and could live in ``tools/background.py``
# alongside their schemas).  Deferred because it's a large move; pick this
# up when next touching either group.

import os
import json
import time
import threading
import difflib
import fnmatch
import subprocess
import tempfile

from ..config import SKIP_DIRS, BINARY_EXTS
from ..runtime.tools.result import ToolResult


def _track_file(path: str) -> None:
    """Record file access for post-compaction re-injection (best-effort)."""
    try:
        from ..memory.compaction import record_file_access
        record_file_access(path)
    except Exception:
        pass  # never let tracking break tool execution

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

    # Preserve original file permissions (mkstemp defaults to 0o600)
    import stat as _stat
    original_mode = None
    try:
        original_mode = os.stat(path).st_mode
    except FileNotFoundError:
        pass

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
        if original_mode is not None:
            os.chmod(tmp_path, _stat.S_IMODE(original_mode))
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


def run_shell(
    command: str,
    timeout: int = 60,
    _cancel_event: "threading.Event | None" = None,
) -> ToolResult:
    """Execute a shell command and return the output.

    Cooperative cancellation
    ------------------------
    When the runtime injects ``_cancel_event`` (i.e. the engine has a
    cancellation event wired up), we switch from the blocking
    ``subprocess.run`` to a ``Popen``-and-poll loop so a Ctrl-C / cancel
    signal teardowns the child process cleanly instead of leaking it as
    a daemon-thread side effect.  Tools that don't opt into the kwarg
    keep using the legacy blocking path.
    """
    try:
        effective_timeout = max(1, min(int(timeout), 600))
    except (TypeError, ValueError):
        effective_timeout = 60

    # Legacy path — no cancel event wired up.  Preserved verbatim so the
    # blocking semantics on offline / scripted callers are unchanged.
    if _cancel_event is None:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=effective_timeout,
            )
            output = result.stdout
            if result.stderr:
                output += "\nSTDERR: " + result.stderr
            if result.returncode != 0 and not output.strip():
                output = f"Command exited with code {result.returncode}"
            if len(output) > 50000:
                output = output[:50000] + "\n\n[... output truncated at 50000 chars ...]"
            return ToolResult(output, is_error=(result.returncode != 0))
        except subprocess.TimeoutExpired:
            return ToolResult(
                f"Error: Command timed out after {effective_timeout} seconds",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(f"Error: {e}", is_error=True)

    # Cooperative-cancel path: spawn the child in its own process group,
    # poll for completion / cancel / timeout, terminate gracefully on
    # cancel.  Process-group teardown ensures we kill the whole shell-
    # spawned tree (most ``run_shell`` invocations look like
    # ``foo && bar``), not just ``/bin/sh``.
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # POSIX: detach into new process group
        )
    except Exception as e:
        return ToolResult(f"Error: {e}", is_error=True)

    deadline = time.monotonic() + effective_timeout
    poll_interval = 0.1
    cancelled = False
    timed_out = False
    while True:
        if proc.poll() is not None:
            break
        if _cancel_event.is_set():
            cancelled = True
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(poll_interval)

    if cancelled or timed_out:
        # Graceful shutdown: SIGTERM the whole process group, give it a
        # short grace period, then SIGKILL anything still alive.
        try:
            os.killpg(os.getpgid(proc.pid), 15)  # SIGTERM
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), 9)  # SIGKILL
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    try:
        stdout, stderr = proc.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        stdout, stderr = "", ""

    if cancelled:
        return ToolResult(
            "Cancelled: shell command aborted on cancel signal "
            "(child process group terminated).",
            is_error=True,
        )
    if timed_out:
        return ToolResult(
            f"Error: Command timed out after {effective_timeout} seconds",
            is_error=True,
        )

    output = stdout or ""
    if stderr:
        output += "\nSTDERR: " + stderr
    rc = proc.returncode if proc.returncode is not None else -1
    if rc != 0 and not output.strip():
        output = f"Command exited with code {rc}"
    if len(output) > 50000:
        output = output[:50000] + "\n\n[... output truncated at 50000 chars ...]"
    return ToolResult(output, is_error=(rc != 0))


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

        _track_file(path)  # record for post-compaction re-injection
        _, ext = os.path.splitext(path)
        if ext.lower() in BINARY_EXTS:
            size = os.path.getsize(path)
            return ToolResult(f"[Binary file: {path} ({size} bytes)]")

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
        _track_file(path)  # record for post-compaction re-injection
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
    _track_file(path)  # record for post-compaction re-injection

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

    # --- Explicit create on existing file → overwrite (not append) ---
    if operation == "create":
        if dry_run:
            return ToolResult(f"[DRY RUN] Would overwrite {path} ({len(new_text)} chars)")
        atomic_write(path, new_text)
        lines = new_text.count('\n') + (1 if new_text and not new_text.endswith('\n') else 0)
        return ToolResult(f"Overwrote {path} ({lines} lines, {len(new_text)} chars)")

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

        return ToolResult(
            f"Inserted {n_inserted} lines before L{insert_at_line} in {path} "
            f"(now {len(new_lines)} lines)"
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
    if old_text == new_text:
        return ToolResult(f"No change needed: old_text and new_text are identical in {path}")

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

    return ToolResult(
        f"Edited {path}: replaced {old_line_count} lines with {new_line_count} lines "
        f"at L{edit_line} ({delta_str}), total {total_lines} lines"
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

    if n_lines <= len(all_lines) and n_lines > 0 and len(all_lines) <= 5000:
        best_ratio = 0.0
        best_start = 0
        for i in range(len(all_lines) - n_lines + 1):
            candidate = '\n'.join(all_lines[i:i + n_lines])
            sm = difflib.SequenceMatcher(None, old_text, candidate)
            if sm.quick_ratio() < best_ratio:
                continue
            ratio = sm.ratio()
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


# ─── Background process management ──────────────────────────────────────────

import atexit
import threading as _bg_threading

# {pid: {"command", "output_file", "file_handle", "process", "started_at"}}
_background_processes: dict[int, dict] = {}
_bg_lock = _bg_threading.Lock()

# Completed-entry TTL: keep for 10 minutes so callers can re-read, then evict.
_BG_COMPLETED_TTL = 600  # seconds


def _bg_cleanup_completed() -> None:
    """Housekeeping sweep — runs opportunistically from run_background().

    Does two things:
    1. Reap processes that exited without ever being polled — close their
       write fd and stamp completed_at so the TTL clock starts.
    2. Evict completed entries whose TTL has elapsed; delete their temp file.
    """
    now = time.time()
    to_remove: list[int] = []
    with _bg_lock:
        for pid, info in _background_processes.items():
            proc = info.get("process")
            # (0) Enforce deadlines even if no one polled this job.
            _enforce_deadline(info)
            # (1) Plug leak: proc exited but was never polled → close fh now
            if (
                proc is not None
                and proc.poll() is not None
                and info.get("completed_at") is None
            ):
                info["completed_at"] = now
                fh = info.get("file_handle")
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
            # (2) TTL eviction
            if info.get("completed_at") and now - info["completed_at"] > _BG_COMPLETED_TTL:
                to_remove.append(pid)
        for pid in to_remove:
            info = _background_processes.pop(pid, None)
            if info:
                _bg_close_and_cleanup(info)


def _bg_close_and_cleanup(info: dict) -> None:
    """Close file handle and remove temp file (best-effort)."""
    try:
        info["file_handle"].close()
    except Exception:
        pass
    try:
        os.unlink(info["output_file"])
    except Exception:
        pass


def _bg_atexit_cleanup() -> None:
    """Terminate all tracked background processes on interpreter exit."""
    with _bg_lock:
        for pid, info in list(_background_processes.items()):
            proc = info.get("process")
            if proc and proc.poll() is None:
                try:
                    os.killpg(proc.pid, 15)  # SIGTERM to process group
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        proc.kill()
                    except Exception:
                        pass
            _bg_close_and_cleanup(info)
        _background_processes.clear()


atexit.register(_bg_atexit_cleanup)


# ── Concurrency and deadline policy ────────────────────────────────────
_BG_MAX_CONCURRENT = 8          # cap simultaneous live background jobs
_BG_WAIT_MAX_SECONDS = 300      # hard cap on action="wait" blocking time


def _count_live_jobs() -> int:
    """Return how many tracked jobs are still running. Caller holds _bg_lock."""
    n = 0
    for info in _background_processes.values():
        proc = info.get("process")
        if proc is not None and proc.poll() is None:
            n += 1
    return n


def _enforce_deadline(info: dict) -> bool:
    """If the job has a deadline and it has passed, kill the process group
    and mark the entry as timed_out. Returns True if we just killed it."""
    deadline = info.get("deadline")
    if not deadline:
        return False
    proc = info.get("process")
    if proc is None or proc.poll() is not None:
        return False
    if time.time() < deadline:
        return False
    # Past deadline and still running — terminate the group.
    try:
        os.killpg(proc.pid, 15)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    info["timed_out"] = True
    return True


def run_background(
    command: str,
    timeout_seconds: int = 0,
    cwd: str = "",
    stdin_null: bool = True,
) -> ToolResult:
    """Start a long-running command in the background.

    Returns immediately with the PID and output file path.
    Use check_background(pid) to poll status and read output.

    timeout_seconds: if > 0, the job is auto-killed once the deadline passes
                     (enforced on every check_background call and via atexit).
                     Status will then read 'timed_out'.
    cwd: optional working directory for the command. Empty → inherit.
    stdin_null: default True — redirects stdin from /dev/null so commands
                that accidentally prompt do not hang forever. Set False only
                if the child must inherit the parent's stdin.
    """
    # Reject if we're already at the concurrency cap.
    with _bg_lock:
        live = _count_live_jobs()
    if live >= _BG_MAX_CONCURRENT:
        return ToolResult(
            json.dumps({
                "status": "rejected",
                "reason": "concurrency_cap",
                "message": (
                    f"Already {live} live background jobs (cap={_BG_MAX_CONCURRENT}). "
                    f"Kill or wait for existing jobs before starting a new one."
                ),
                "live_jobs": live,
                "cap": _BG_MAX_CONCURRENT,
            }),
            is_error=True,
        )

    # Validate cwd early — Popen's error is less actionable.
    popen_cwd = None
    if cwd:
        if not os.path.isdir(cwd):
            return ToolResult(
                f"Error starting background process: cwd does not exist: {cwd}",
                is_error=True,
            )
        popen_cwd = cwd

    fh = None
    output_file = None
    stdin_fh = None
    try:
        fd_int, output_file = tempfile.mkstemp(
            prefix="jyagent_bg_", suffix=".out", dir="/tmp",
        )
        fh = os.fdopen(fd_int, "w")
        if stdin_null:
            stdin_arg = subprocess.DEVNULL
        else:
            stdin_arg = None  # inherit
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=stdin_arg,
            cwd=popen_cwd,
            start_new_session=True,  # detach from parent signal group
        )
        started_at = time.time()
        deadline = started_at + timeout_seconds if timeout_seconds > 0 else None
        with _bg_lock:
            _background_processes[proc.pid] = {
                "command": command,
                "cwd": popen_cwd or "",
                "output_file": output_file,
                "file_handle": fh,
                "process": proc,
                "started_at": started_at,
                "deadline": deadline,
                "completed_at": None,
                "killed": False,
                "timed_out": False,
            }
        # Opportunistically evict old completed entries
        _bg_cleanup_completed()

        return ToolResult(json.dumps({
            "pid": proc.pid,
            "output_file": output_file,
            "status": "started",
        }))
    except Exception as e:
        # Clean up on failure — avoid leaking the temp file and fd
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
        if output_file is not None:
            try:
                os.unlink(output_file)
            except Exception:
                pass
        return ToolResult(f"Error starting background process: {e}", is_error=True)


# Hard cap on how many bytes any single read returns to the model.
# The file on disk is unbounded; this only caps what we pull into memory.
_BG_OUTPUT_MAX_BYTES = 50_000
# When scanning backwards to find N newlines, stop after this many bytes
# regardless of newline count. Prevents single-line / no-newline pathology.
_BG_TAIL_SCAN_BUDGET = 256 * 1024  # 256 KB


def _read_tail_efficient(filepath: str, n: int) -> str:
    """Read the last N lines from a file efficiently using seek-from-end.

    Hardened against pathological input (no newlines, huge single line):
    - Bounds the backward scan to `_BG_TAIL_SCAN_BUDGET` bytes.
    - Always caps the decoded string at `_BG_OUTPUT_MAX_BYTES` chars.
    - Uses utf-8 with replacement (does not depend on locale).
    """
    try:
        with open(filepath, "rb") as f:
            f.seek(0, 2)  # end of file
            size = f.tell()
            if size == 0:
                return ""
            # How far back we are willing to read.
            scan_cap = min(size, _BG_TAIL_SCAN_BUDGET)
            chunk_size = 8192
            lines_found = 0
            pos = size
            scanned = 0
            while pos > 0 and scanned < scan_cap and lines_found <= n:
                read_size = min(chunk_size, pos, scan_cap - scanned)
                pos -= read_size
                scanned += read_size
                f.seek(pos)
                chunk = f.read(read_size)
                lines_found += chunk.count(b"\n")
            f.seek(pos)
            raw = f.read(size - pos)
            data = raw.decode("utf-8", errors="replace")
            # If we found newlines, slice by line; otherwise return the last
            # scan_cap bytes as-is (single-line / no-newline case).
            if lines_found > 0:
                out = "\n".join(data.splitlines()[-n:])
            else:
                out = data
            if len(out) > _BG_OUTPUT_MAX_BYTES:
                out = "[... truncated to last %d chars ...]\n" % _BG_OUTPUT_MAX_BYTES + out[-_BG_OUTPUT_MAX_BYTES:]
            return out
    except Exception as e:
        return f"(error reading tail: {e})"


def _read_tail_bytes(filepath: str, max_bytes: int) -> str:
    """Return the last `max_bytes` of a file as utf-8 (errors=replace).

    Used for the `tail=0` path so we never load a multi-GB log into memory.
    """
    try:
        with open(filepath, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return ""
            read_size = min(size, max_bytes)
            f.seek(size - read_size)
            data = f.read(read_size).decode("utf-8", errors="replace")
            if read_size < size:
                data = "[... truncated to last %d chars ...]\n" % max_bytes + data
            return data
    except Exception as e:
        return f"(error reading output: {e})"


def _classify_exit(returncode: int | None, killed: bool, timed_out: bool) -> str:
    """Derive a terminal status from observed process state.

    Precedence: running > timed_out > killed > exit_code.

    - running  : still alive (returncode is None)
    - timed_out: we killed it because it hit its deadline
    - killed   : caller explicitly killed it via action="kill"
    - succeeded: exited with 0
    - failed   : exited with non-zero
    """
    if returncode is None:
        return "running"
    if timed_out:
        return "timed_out"
    if killed:
        return "killed"
    return "succeeded" if returncode == 0 else "failed"


def check_background(
    pid: int,
    tail: int = 0,
    action: str = "status",
    wait_timeout_seconds: int = 60,
) -> ToolResult:
    """Check on a background process started by run_background.

    action="status" (default): return current status + output.
    action="wait":  block up to wait_timeout_seconds for the job to finish,
                    then return status + output. Hard cap: 300 s.
                    Saves a model turn compared to tight polling.
    action="kill":  send SIGTERM (then SIGKILL after 5 s) to the process group.

    tail=0: return the last ~50 KB of output (seek-from-end, bounded).
    tail=N: return only the last N lines, with a bounded backward scan.
    """
    with _bg_lock:
        info = _background_processes.get(pid)
    if info is None:
        # Check if PID exists in OS but wasn't started by us
        try:
            os.kill(pid, 0)
            return ToolResult(json.dumps({
                "pid": pid,
                "status": "unknown",
                "message": "PID exists but was not started by run_background",
            }), is_error=True)
        except ProcessLookupError:
            return ToolResult(json.dumps({
                "pid": pid,
                "status": "not_found",
                "message": "No such process. It may have already been checked after completion.",
            }), is_error=True)

    proc: subprocess.Popen = info["process"]

    # Enforce deadline before anything else — if the job overran its
    # timeout_seconds, we terminate it and mark timed_out=True.
    _enforce_deadline(info)

    # Track whether *this invocation* actually signaled the process. This
    # distinguishes "we killed it" from "it had already exited" so the
    # returned status reflects observed outcome rather than the request.
    signaled = False

    if action == "wait":
        # Block up to the cap; no-op if already exited.
        if proc.poll() is None:
            wait_s = max(1, min(int(wait_timeout_seconds), _BG_WAIT_MAX_SECONDS))
            try:
                proc.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                pass
            # Deadline may have elapsed while we were blocked.
            _enforce_deadline(info)
    elif action == "kill":
        if proc.poll() is None:  # still running — actually signal
            signaled = True
            try:
                os.killpg(proc.pid, 15)  # SIGTERM to process group
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, 9)  # SIGKILL to process group
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        # else: process already terminated on its own — do NOT claim "killed"

    returncode = proc.poll()
    is_running = returncode is None
    elapsed = time.time() - info["started_at"]

    # Preserve "killed" sticky across subsequent polls (once killed, stays killed).
    if signaled:
        info["killed"] = True
    killed_flag = bool(info.get("killed"))
    timed_out_flag = bool(info.get("timed_out"))

    # Read output (bounded on both tail>0 and tail=0 paths)
    output_file = info["output_file"]
    try:
        if tail > 0:
            output = _read_tail_efficient(output_file, tail)
        else:
            output = _read_tail_bytes(output_file, _BG_OUTPUT_MAX_BYTES)
    except FileNotFoundError:
        output = "(output file not yet created)"

    result = {
        "pid": pid,
        "status": _classify_exit(returncode, killed_flag, timed_out_flag),
        "exit_code": returncode,
        "elapsed_seconds": round(elapsed, 1),
        "command": info["command"],
        "output_file": output_file,
        "output": output,
    }
    if info.get("deadline") is not None:
        result["deadline_seconds_remaining"] = round(
            max(0.0, info["deadline"] - time.time()), 1
        )

    # Mark completed exactly once (done under lock; close handle idempotently).
    if not is_running:
        with _bg_lock:
            if info.get("completed_at") is None:
                info["completed_at"] = time.time()
                fh = info.get("file_handle")
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass

    return ToolResult(json.dumps(result, ensure_ascii=False))
