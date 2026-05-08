# Core file/shell tools and shared path/file helpers.
#
# TODO: split this module into ``core`` and
# ``background`` (the run_background / check_background / _bg_* helpers
# stack accounts for ~half the file and could live in ``tools/background.py``
# alongside their schemas).  Deferred because it's a large move; pick this
# up when next touching either group.

import os
import sys
import time
import threading
import difflib
import fnmatch
import subprocess
import tempfile

from ..config import SKIP_DIRS, BINARY_EXTS
from ..runtime.tools.result import ToolResult
from ..utils.files import atomic_write


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


# ``atomic_write`` is imported from ``..utils.files`` for use in
# ``write_file``/``edit_file`` below.  Callers outside this module should
# import it from ``jyagent.utils.files`` directly — not from here.


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


# --- Bounded-output drain for run_shell -----------------------------------
#
# Why this exists: the previous implementation used
# ``subprocess.run(capture_output=True, text=True, ...)`` (or
# ``proc.communicate()``), both of which buffer the *entire* stdout+stderr
# in RAM before truncating to 50 000 chars. A runaway command (recursive
# find, ``kubectl logs -f``, infinite log loop, accidental ``yes`` pipe…)
# could push the parent agent into multi-GB territory and trigger
# macOS jetsam / Linux OOM-killer (we observed ~74 GiB resident → SIGKILL
# in the wild — see journal 2026-04-30).
#
# Three layers of defense, in order:
#   1. Bounded head + tail byte buffers per stream (head=8 KB, tail=128 KB).
#      Memory usage is O(head + tail) regardless of how much the child writes.
#   2. **Spill-to-disk** the moment tail is about to lose data (total >
#      tail_max).  Tail still contains the full output at that instant; we
#      dump it to /tmp and tee every subsequent chunk.  This makes the full
#      output recoverable by the agent even when the inline view is
#      head+tail-truncated.  (Same recovery model as Claude Code's
#      tool-results spill files; we keep head+tail inline preview rather
#      than head-only.)
#   3. Hard SIGKILL of the child's process group at ``hard_kill_bytes`` per
#      stream as a last-resort circuit breaker.  Spill captures whatever
#      was written before the kill.

def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Parse an int env var with explicit warnings on bad values.

    Claude Code's `BASH_MAX_OUTPUT_LENGTH` regression in v2.1.2 silently
    ignored the env var; this helper logs to stderr instead so misconfig
    is at least visible at startup.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        v = int(raw.strip())
    except ValueError:
        print(
            f"[jyagent] env {name}={raw!r} is not an int; using default {default}",
            file=sys.stderr,
        )
        return default
    if v < minimum:
        print(
            f"[jyagent] env {name}={v} below minimum {minimum}; using default {default}",
            file=sys.stderr,
        )
        return default
    return v


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    print(
        f"[jyagent] env {name}={raw!r} not a bool; using default {default}",
        file=sys.stderr,
    )
    return default


# All knobs below are ENV-CONFIGURABLE so users can tune memory budget /
# spill behaviour without patching source.  Names are stable; never silently
# ignore a set value (warn loudly if malformed — see _env_int).
_RUN_SHELL_HEAD_BYTES = _env_int(
    "JYAGENT_RUN_SHELL_HEAD_BYTES", 8 * 1024, minimum=0,
)  # in-memory head buffer (banners, command echo, early progress)
_RUN_SHELL_TAIL_BYTES = _env_int(
    "JYAGENT_RUN_SHELL_TAIL_BYTES", 128 * 1024, minimum=0,
)  # in-memory tail buffer (errors, final result, stack traces)
_RUN_SHELL_HARD_KILL_BYTES = _env_int(
    "JYAGENT_RUN_SHELL_HARD_KILL_BYTES", 32 * 1024 * 1024, minimum=1024,
)  # per-stream hard cap → SIGKILL pgroup
_RUN_SHELL_OUTPUT_CHAR_CAP = _env_int(
    "JYAGENT_RUN_SHELL_OUTPUT_CHAR_CAP", 50_000, minimum=100,
)  # user-visible char cap on the returned ToolResult
_RUN_SHELL_SPILL_ENABLED = _env_bool(
    "JYAGENT_RUN_SHELL_SPILL", True,
)  # tee large outputs to /tmp for agent-side recovery
_RUN_SHELL_SPILL_DIR = os.environ.get(
    "JYAGENT_RUN_SHELL_SPILL_DIR", "/tmp",
).strip() or "/tmp"


class _BoundedStreamReader(threading.Thread):
    """Drain a binary pipe into a bounded head+tail buffer with optional disk spill.

    Total RAM held per instance ≈ ``head_bytes + tail_bytes``. When the
    child's output for this stream is about to exceed ``tail_bytes``
    (i.e. the tail buffer would lose its first bytes), and ``spill_dir``
    is provided, we lazily open a spill file in ``spill_dir`` and dump
    the current tail (which still holds the full output up to that
    point) before continuing to tee every chunk.  When the child
    exceeds ``hard_kill_bytes``, ``on_overflow`` is invoked (typically
    to SIGKILL the whole process group) and further bytes are still
    consumed from the pipe (so the child's ``write()`` calls don't
    block) but discarded from the in-memory buffer.  Spill continues
    writing until the pipe closes, so the file captures everything the
    child managed to emit before the kill.
    """

    _CHUNK = 64 * 1024

    def __init__(
        self,
        stream,
        head_bytes: int,
        tail_bytes: int,
        hard_kill_bytes: int,
        on_overflow,
        spill_path_factory=None,
    ):
        super().__init__(daemon=True)
        self._stream = stream
        self._head_max = head_bytes
        self._tail_max = tail_bytes
        self._hard_kill = hard_kill_bytes
        self._on_overflow = on_overflow
        # Spill: a callable returning a (path, file_handle) tuple, called
        # lazily the first time we need to spill.  None disables spill.
        # The factory pattern lets callers control where the file lives
        # (and lets tests inject a mock without touching /tmp).
        self._spill_path_factory = spill_path_factory
        self._head = bytearray()
        self._tail = bytearray()
        self._total = 0
        self._overflow_fired = False
        self._spill_path: str | None = None
        self._spill_fh = None  # open binary file handle once spill activates

    def _activate_spill(self) -> None:
        """Lazy-open the spill file and dump the tail buffer.

        Called when the next chunk would push total past tail_max — the
        last moment at which the tail buffer still holds the full output
        (bytes [0, total)).  After this, every subsequent chunk is teed
        to the spill file in addition to head/tail bookkeeping.
        """
        if self._spill_path_factory is None or self._spill_fh is not None:
            return
        try:
            path, fh = self._spill_path_factory()
        except Exception:
            # Spill is best-effort; if /tmp is unwritable (e.g. read-only
            # rootfs in a container) we silently disable and continue with
            # the in-memory bounded buffer.
            self._spill_path_factory = None
            return
        self._spill_path = path
        self._spill_fh = fh
        # Tail currently holds bytes [0, self._total) since we haven't
        # exceeded tail_max yet.
        try:
            self._spill_fh.write(bytes(self._tail))
        except Exception:
            self._close_spill()

    def _close_spill(self) -> None:
        if self._spill_fh is not None:
            try:
                self._spill_fh.flush()
                self._spill_fh.close()
            except Exception:
                pass
            self._spill_fh = None

    def run(self) -> None:  # noqa: D401 — Thread.run override
        try:
            while True:
                chunk = self._stream.read(self._CHUNK)
                if not chunk:
                    break

                # Spill activation: if adding this chunk would cause the
                # tail buffer to overflow (i.e. lose its early bytes), open
                # the spill file NOW while tail still has bytes [0, total).
                if (
                    self._spill_fh is None
                    and self._spill_path_factory is not None
                    and self._total + len(chunk) > self._tail_max
                ):
                    self._activate_spill()

                self._total += len(chunk)

                # Head: fill up to head_max, then stop appending.
                if len(self._head) < self._head_max:
                    take = self._head_max - len(self._head)
                    self._head.extend(chunk[:take])

                # Tail: keep last tail_max bytes via concat + slice.
                # (For the chunk sizes we use, this is cheap; bytearray
                # slicing copies but stays bounded by tail_max + chunk.)
                self._tail.extend(chunk)
                if len(self._tail) > self._tail_max:
                    del self._tail[: len(self._tail) - self._tail_max]

                # Tee chunk to spill (if spill activated).
                if self._spill_fh is not None:
                    try:
                        self._spill_fh.write(chunk)
                    except Exception:
                        # Disk full / fd error — close and continue with
                        # in-memory buffer only.
                        self._close_spill()

                if (
                    not self._overflow_fired
                    and self._total >= self._hard_kill
                ):
                    self._overflow_fired = True
                    try:
                        self._on_overflow()
                    except Exception:
                        pass
                    # Keep draining so the child can exit cleanly on
                    # SIGKILL; ``read`` will return b'' shortly.  Spill
                    # also keeps writing — captures everything the child
                    # emitted up to the kill.
        except Exception:
            # Pipe closed / decode error / EBADF — drain ends, caller
            # will collect whatever we have.
            pass
        finally:
            self._close_spill()

    def collect(self) -> tuple[str, bool]:
        """Decode the bounded buffer to str. Returns (text, truncated).

        Three regimes, depending on how much the child wrote:

        1. ``total <= tail_max``: the tail buffer never overflowed, so it
           holds the entire output (and head is a prefix of it). Return
           tail. No data loss.
        2. ``tail_max < total <= head_max + tail_max``: tail dropped its
           early bytes but head still has them. The two buffers overlap
           in the middle and together cover the full output — dedupe by
           taking the prefix of head that tail no longer has, then all
           of tail. No data loss, no elision marker.
        3. ``total > head_max + tail_max``: real gap between head and
           tail. Return ``head + elision_marker + tail``.
        """
        if self._total <= self._tail_max:
            data = bytes(self._tail)
        elif self._total <= self._head_max + self._tail_max:
            # head_keep = bytes that head has but tail dropped.
            head_keep = self._total - self._tail_max
            data = bytes(self._head[:head_keep]) + bytes(self._tail)
        else:
            elided = self._total - self._head_max - len(self._tail)
            data = (
                bytes(self._head)
                + f"\n\n[... {elided} bytes elided ...]\n\n".encode("utf-8")
                + bytes(self._tail)
            )
        text = data.decode("utf-8", errors="replace")
        return text, self._overflow_fired

    @property
    def spill_path(self) -> str | None:
        """Path to the on-disk spill file, or None if spill never activated.

        Reliable to read after the reader thread has joined.
        """
        return self._spill_path


def _killpg_quiet(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _make_spill_path_factory(label: str):
    """Return a factory that lazy-creates a spill tempfile for `label` (out|err).

    Returns ``None`` if spill is disabled via env, so the reader skips
    spill entirely and stays in-memory only.  The factory itself returns
    a (path, file_handle) tuple, opened in append-binary mode.
    """
    if not _RUN_SHELL_SPILL_ENABLED:
        return None

    def factory():
        import tempfile
        # mkstemp avoids the python tempfile auto-delete; we want the
        # file to persist so the agent can read it back.  Caller is
        # responsible for cleanup (or relying on /tmp's reaper).
        fd, path = tempfile.mkstemp(
            prefix=f"jyagent_runshell_{label}_",
            suffix=".out",
            dir=_RUN_SHELL_SPILL_DIR,
        )
        return path, os.fdopen(fd, "ab")

    return factory


def _drain_proc_bounded(proc: "subprocess.Popen") -> tuple[
    "_BoundedStreamReader", "_BoundedStreamReader"
]:
    """Spawn drain threads for proc.stdout and proc.stderr."""
    overflow_kill = lambda: _killpg_quiet(proc.pid, 9)  # SIGKILL on overflow
    out_reader = _BoundedStreamReader(
        proc.stdout,
        _RUN_SHELL_HEAD_BYTES,
        _RUN_SHELL_TAIL_BYTES,
        _RUN_SHELL_HARD_KILL_BYTES,
        overflow_kill,
        spill_path_factory=_make_spill_path_factory("out"),
    )
    err_reader = _BoundedStreamReader(
        proc.stderr,
        _RUN_SHELL_HEAD_BYTES,
        _RUN_SHELL_TAIL_BYTES,
        _RUN_SHELL_HARD_KILL_BYTES,
        overflow_kill,
        spill_path_factory=_make_spill_path_factory("err"),
    )
    out_reader.start()
    err_reader.start()
    return out_reader, err_reader


def _spill_note(out_reader, err_reader) -> str:
    """Format spill-path lines for cancel/timeout error messages."""
    lines = []
    if out_reader.spill_path:
        lines.append(f"\n[stdout spilled to {out_reader.spill_path}]")
    if err_reader.spill_path:
        lines.append(f"\n[stderr spilled to {err_reader.spill_path}]")
    return "".join(lines)


def _format_run_shell_output(
    stdout_text: str,
    stderr_text: str,
    returncode: int,
    overflowed: bool,
    stdout_spill_path: str | None = None,
    stderr_spill_path: str | None = None,
) -> str:
    output = stdout_text
    if stderr_text:
        if output and not output.endswith("\n"):
            output += "\n"
        output += "STDERR: " + stderr_text
    if returncode != 0 and not output.strip() and not overflowed:
        output = f"Command exited with code {returncode}"
    # Apply the user-visible char cap *before* appending the overflow
    # marker. Otherwise a runaway child fills the cap with garbage and
    # the marker gets sliced off — silent regression of the very signal
    # we care about.
    if len(output) > _RUN_SHELL_OUTPUT_CHAR_CAP:
        output = (
            output[:_RUN_SHELL_OUTPUT_CHAR_CAP]
            + f"\n\n[... output truncated at {_RUN_SHELL_OUTPUT_CHAR_CAP} chars ...]"
        )
    # Spill-to-disk markers: tell the agent where to find the full output
    # so it can `read_file` / `run_shell head -c …` into the recovered
    # file.  Same recovery model as Claude Code's tool-results spill, but
    # the inline preview above is head+tail rather than head-only.
    spill_lines = []
    if stdout_spill_path:
        spill_lines.append(
            f"[full stdout spilled to {stdout_spill_path} — "
            "use read_file / run_shell tail/grep to inspect]"
        )
    if stderr_spill_path:
        spill_lines.append(
            f"[full stderr spilled to {stderr_spill_path} — "
            "use read_file / run_shell tail/grep to inspect]"
        )
    if spill_lines:
        if output and not output.endswith("\n"):
            output += "\n"
        output += "\n" + "\n".join(spill_lines)
    if overflowed:
        if output and not output.endswith("\n"):
            output += "\n"
        output += (
            f"\n[!! output exceeded {_RUN_SHELL_HARD_KILL_BYTES // (1024*1024)} MB "
            "on a single stream — child process group SIGKILLed to protect "
            "the agent. Use run_background for high-volume commands.]"
        )
    return output


def run_shell(
    command: str,
    timeout: int = 60,
    _cancel_event: "threading.Event | None" = None,
) -> ToolResult:
    """Execute a shell command and return the output.

    Memory safety
    -------------
    Output is drained through bounded head+tail byte buffers
    (``_BoundedStreamReader``) so a child emitting GBs of output cannot
    push this process into the OOM-killer. If either stdout or stderr
    exceeds ``_RUN_SHELL_HARD_KILL_BYTES`` we SIGKILL the child's
    process group and tag the output with an overflow marker.

    Cooperative cancellation
    ------------------------
    When the runtime injects ``_cancel_event`` (i.e. the engine has a
    cancellation event wired up), the poll loop checks it on every tick
    and tears down the child's process group on cancel. Without the
    event we just block on ``proc.wait`` with a timeout.
    """
    try:
        effective_timeout = max(1, min(int(timeout), 600))
    except (TypeError, ValueError):
        effective_timeout = 60

    # Always Popen + drain. ``start_new_session=True`` puts the child in
    # its own process group so ``killpg`` cleans up the whole tree
    # (most run_shell invocations look like ``foo && bar``, not a single
    # exec).
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            # NOTE: text=False (default) — we decode at collect() time so
            # the bounded buffer holds bytes, not Python str objects.
        )
    except Exception as e:
        return ToolResult(f"Error: {e}", is_error=True)

    out_reader, err_reader = _drain_proc_bounded(proc)

    deadline = time.monotonic() + effective_timeout
    poll_interval = 0.1
    cancelled = False
    timed_out = False
    while True:
        if proc.poll() is not None:
            break
        if _cancel_event is not None and _cancel_event.is_set():
            cancelled = True
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(poll_interval)

    if cancelled or timed_out:
        # SIGTERM, brief grace period, then SIGKILL the whole pgroup.
        _killpg_quiet(proc.pid, 15)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _killpg_quiet(proc.pid, 9)
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    # Make sure the child is reaped and drain threads finish — they exit
    # as soon as the pipes close, which happens at child exit.
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _killpg_quiet(proc.pid, 9)
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    out_reader.join(timeout=2)
    err_reader.join(timeout=2)

    if cancelled:
        spill_note = _spill_note(out_reader, err_reader)
        return ToolResult(
            "Cancelled: shell command aborted on cancel signal "
            "(child process group terminated)." + spill_note,
            is_error=True,
        )
    if timed_out:
        spill_note = _spill_note(out_reader, err_reader)
        return ToolResult(
            f"Error: Command timed out after {effective_timeout} seconds"
            + spill_note,
            is_error=True,
        )

    stdout_text, stdout_overflow = out_reader.collect()
    stderr_text, stderr_overflow = err_reader.collect()
    overflowed = stdout_overflow or stderr_overflow
    rc = proc.returncode if proc.returncode is not None else -1
    output = _format_run_shell_output(
        stdout_text, stderr_text, rc, overflowed,
        stdout_spill_path=out_reader.spill_path,
        stderr_spill_path=err_reader.spill_path,
    )
    return ToolResult(output, is_error=(rc != 0 or overflowed))


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
#
# Lives in ``jyagent/tools/background.py`` (extracted 2026-05-06 so this
# module can focus on shell + file/edit primitives).  Tool registration
# pulls run_background/check_background directly from there in
# tools/__init__.py.
