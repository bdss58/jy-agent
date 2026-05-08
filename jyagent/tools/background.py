"""Background-process tools: ``run_background`` / ``check_background``.

Extracted from ``tools/core.py`` (2026-05-06) so the ~1.5 KLOC core module
no longer mixes shell/file/edit primitives with long-running process
management.  This module owns:

  * the in-process registry of live & recently-completed background jobs
    (``_background_processes`` + ``_bg_lock``)
  * lifecycle: spawn, deadline enforcement, atexit cleanup, completed-TTL eviction
  * tail-efficient output reading (``_read_tail_efficient`` / ``_read_tail_bytes``)
  * the ``run_background`` and ``check_background`` tool entrypoints

Self-contained: depends only on stdlib + ``ToolResult``.  Tool registration
happens in ``tools/__init__.py`` which imports ``run_background`` /
``check_background`` directly from this module.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import tempfile
import threading
import time

from ..runtime.tools.result import ToolResult



# {pid: {"command", "output_file", "file_handle", "process", "started_at"}}
_background_processes: dict[int, dict] = {}
_bg_lock = threading.Lock()

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
