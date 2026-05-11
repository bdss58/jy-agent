"""POSIX process-tree utilities for MCP subprocess lifecycle management.

When ``MCPClient.disconnect()`` runs, the SDK's ``stdio_client`` cleanup
should already terminate the immediate MCP server subprocess. But the MCP
server itself may have spawned children that ``setsid()`` into their own
process groups — Chrome via ``chrome-devtools-mcp`` is the canonical case.
Killing the npm process group does NOT kill Chrome.

These helpers walk the descendant process tree, collect every distinct
process group id, and apply ``SIGTERM`` (then ``SIGKILL`` after a brief
grace period) to every group. Last-resort cleanup; safe to call after
graceful shutdown to backstop leaked children.

Pure functions — no class state. Tested through ``MCPClient.disconnect``;
isolated here so the lifecycle logic can grow independently of the SDK
wrapper. Extracted from ``client.py`` 2026-05-12 as step 2 of the mcp/
cleanup.
"""

from __future__ import annotations

import os
import signal
import subprocess as sp
import time


__all__ = ["force_kill_process_tree"]


def _get_descendants(root_pid: int) -> set[int]:
    """Recursively find every descendant PID of ``root_pid`` via ``pgrep -P``.

    Returns the empty set if ``pgrep`` is unavailable or ``root_pid`` has no
    children. Each ``pgrep`` invocation has a 3-second timeout so a hung
    descendant cannot block cleanup forever.
    """
    descendants: set[int] = set()
    to_visit = [root_pid]
    while to_visit:
        parent = to_visit.pop()
        try:
            result = sp.run(
                ["pgrep", "-P", str(parent)],
                capture_output=True, text=True, timeout=3,
            )
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    child_pid = int(line.strip())
                    if child_pid not in descendants:
                        descendants.add(child_pid)
                        to_visit.append(child_pid)
        except Exception:
            pass
    return descendants


def _kill_pgid(pgid: int, sig: int) -> None:
    """Send ``sig`` to process group ``pgid``, swallowing errors.

    A killed PID may map to a defunct group by the time we look it up; we
    treat ProcessLookupError / PermissionError / OSError as "already gone".
    """
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _is_alive(pid: int) -> bool:
    """Return True if ``pid`` is still a running process.

    ``os.kill(pid, 0)`` is the canonical "is this process alive" probe — it
    sends signal 0 (no-op) and raises ``ProcessLookupError`` for dead
    processes. ``OSError`` (including ``EPERM``) means the process exists
    but we can't signal it, which still counts as alive for cleanup
    purposes.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def force_kill_process_tree(
    root_pid: int,
    *,
    grace_seconds: float = 2.0,
    poll_interval: float = 0.1,
) -> None:
    """SIGTERM-then-SIGKILL every process group in the tree rooted at ``root_pid``.

    Strategy
    --------
    1. Walk the descendant tree (``pgrep -P`` recursively) and collect every
       distinct process group id, including ``root_pid``'s own group.
    2. SIGTERM every collected group.
    3. Poll for up to ``grace_seconds`` (every ``poll_interval`` seconds);
       return early if all PIDs are gone.
    4. If anything survives, SIGKILL every collected group.

    Why kill by group, not by PID? Children that ``setsid()`` (Chrome,
    headless browsers, daemonized helpers) live in their own group; the
    parent's group does NOT include them. We have to enumerate the tree to
    discover the groups, then signal each group.

    Pure side effects — no return value. Errors are swallowed because this
    is best-effort cleanup; the alternative is leaving zombies behind.
    """
    try:
        all_descendants = _get_descendants(root_pid)
        all_pids = {root_pid} | all_descendants
        pgids: set[int] = set()
        for p in all_pids:
            try:
                pgids.add(os.getpgid(p))
            except (ProcessLookupError, OSError):
                pass

        for pgid in pgids:
            _kill_pgid(pgid, signal.SIGTERM)

        deadline_steps = int(grace_seconds / poll_interval) if poll_interval > 0 else 0
        for _ in range(max(deadline_steps, 1)):
            if not any(_is_alive(p) for p in all_pids):
                return
            time.sleep(poll_interval)

        for pgid in pgids:
            _kill_pgid(pgid, signal.SIGKILL)
    except Exception:
        pass
