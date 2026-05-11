"""run_shell: bounded-output shell execution with spill-to-disk recovery.

Why this lives in its own module: ``run_shell`` carries ~470 LOC of
streaming-drain machinery (_BoundedStreamReader, spill-tempfile factory,
process-group kill helpers, output formatter) that has nothing to do with
filesystem read/write/edit primitives.  Splitting them keeps each file
focused on one job.

History: extracted from the former monolithic ``tools/core.py`` (2026-05).
Long-running background jobs live in ``tools/background.py``.
"""
import os
import sys
import time
import threading
import subprocess
import tempfile

from ..runtime.tools.result import ToolResult



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
            stdin=subprocess.DEVNULL,  # prevent interactive commands from hanging
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

