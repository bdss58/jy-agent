"""Sub-agent terminal status display.

Lives in ``jyagent.ui`` (not ``jyagent.tools``) because it is pure
terminal-rendering: ANSI escape codes, animation thread, line clearing.
The sub-agent tool implementation imports the tracker singleton and
calls ``add`` / ``update_progress`` / ``remove`` — it does NOT know
about ANSI codes.

History: extracted from ``tools/subagent.py`` (2026-05) to remove the
UI-leak smell (codex review hypothesis H2).
"""
import sys
import threading
import time

from .output import _STDOUT_LOCK

# ─── ANSI palette ────────────────────────────────────────────────────────────

COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[1;36m"
COLOR_GREEN = "\033[0;32m"
COLOR_RED = "\033[1;31m"
COLOR_YELLOW = "\033[1;33m"


class _SubagentTracker:
    """Global consolidated spinner for all active sub-agents.

    Thread-safe.  Automatically starts/stops the animation thread when
    the first/last sub-agent registers/deregisters.

    Single subagent:  "🤖 Sub-agent: task preview (45s)"
    Multiple:         "🤖 3 sub-agents running (45s)"
    With steps:       "🤖 Sub-agent: task preview (3/30 steps, 45s)"
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self._lock = threading.Lock()
        self._agents: dict[int, dict] = {}  # id -> {task, t0, step, max_steps}
        self._next_id = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, task: str, max_steps: int = 0) -> int:
        """Register a new sub-agent.  Returns a unique agent id."""
        preview = task[:60] + "..." if len(task) > 60 else task
        with self._lock:
            agent_id = self._next_id
            self._next_id += 1
            self._agents[agent_id] = {
                "task": preview,
                "t0": time.time(),
                "step": 0,
                "max_steps": max_steps,
            }
            if len(self._agents) == 1:
                self._start_animation()
        return agent_id

    def remove(self, agent_id: int) -> None:
        """Deregister a completed sub-agent."""
        with self._lock:
            self._agents.pop(agent_id, None)
            if not self._agents:
                self._stop_animation()

    def update_progress(self, agent_id: int, step: int, max_steps: int = 0) -> None:
        """Update the step progress for an active sub-agent."""
        with self._lock:
            info = self._agents.get(agent_id)
            if info is not None:
                info["step"] = step
                if max_steps:
                    info["max_steps"] = max_steps

    # ── Animation lifecycle ──────────────────────────────────────────────

    def _start_animation(self):
        """Start the animation thread.  Caller must hold self._lock."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def _stop_animation(self):
        """Stop the animation thread.  Caller must hold self._lock."""
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None:
            # Release the lock during join to avoid deadlock
            self._lock.release()
            try:
                t.join(timeout=1)
            finally:
                self._lock.acquire()

    def _animate(self):
        idx = 0
        while not self._stop.is_set():
            with self._lock:
                agents = dict(self._agents)

            if not agents:
                break

            frame = self._FRAMES[idx % len(self._FRAMES)]
            now = time.time()
            count = len(agents)

            if count == 1:
                info = next(iter(agents.values()))
                elapsed = now - info["t0"]
                step = info["step"]
                max_steps = info["max_steps"]
                if step and max_steps:
                    time_info = f"({step}/{max_steps} steps, {elapsed:.0f}s)"
                elif step:
                    time_info = f"(step {step}, {elapsed:.0f}s)"
                else:
                    time_info = f"({elapsed:.0f}s)"
                line = f"\r{COLOR_DIM}  {frame} 🤖 Sub-agent: {info['task']} {time_info}{COLOR_RESET}"
            else:
                # Use the oldest t0 for elapsed
                oldest_t0 = min(a["t0"] for a in agents.values())
                elapsed = now - oldest_t0
                line = f"\r{COLOR_DIM}  {frame} 🤖 {count} sub-agents running ({elapsed:.0f}s){COLOR_RESET}"

            # Serialize on the package-shared lock so this spinner cannot
            # interleave with the planner spinner / reasoning preview /
            # streaming text in :mod:`jyagent.ui.terminal`.
            with _STDOUT_LOCK:
                sys.stdout.write(line)
                sys.stdout.flush()
            idx += 1
            self._stop.wait(0.1)

        # Clear the spinner line (still under the shared lock so a
        # concurrent writer doesn't paint into the half-cleared row).
        with _STDOUT_LOCK:
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()

# Global singleton — sub-agents register/deregister via this.
_subagent_tracker = _SubagentTracker()
