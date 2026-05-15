"""Shared thread-helper mixin for ``AgentLoop`` and ``LLMRunner``.

Both classes need three identical helpers — ``_is_cancelled`` /
``_cancellable_sleep`` / ``_fire`` — that consult their
``_cancel_event`` (``threading.Event | None``) and ``_callbacks``
(``LoopCallbacks``) instance attributes.  Before the helper extraction
the methods were duplicated verbatim across both classes (~30 lines of
cut-and-paste).

The mixin assumes both attributes are named ``_cancel_event`` and
``_callbacks`` on the instance — both consumers were aligned on those
names in the 2026-05 simplification pass.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Any


class LoopThreadHelper:
    """Mixin providing ``_is_cancelled`` / ``_cancellable_sleep`` /
    ``_fire`` / ``_fire_with_return``.

    Requires the consumer to expose two instance attributes:
      * ``_cancel_event`` — ``threading.Event | None``
      * ``_callbacks`` — ``LoopCallbacks`` (or ``None`` for silent runs)
    """

    _cancel_event: Any  # threading.Event | None — declared by subclass
    _callbacks: Any      # LoopCallbacks | None — declared by subclass

    def _is_cancelled(self) -> bool:
        ev = self._cancel_event
        return ev is not None and ev.is_set()

    def _cancellable_sleep(self, seconds: float) -> bool:
        """Sleep that returns early if cancellation is signalled.

        Returns True if cancelled during the wait, False otherwise.
        """
        ev = self._cancel_event
        if ev is None:
            time.sleep(seconds)
            return False
        return ev.wait(seconds)

    def _fire(self, name: str, *args: Any) -> None:
        """Invoke a named callback, swallowing exceptions.

        Callbacks are presentation-layer hooks (UI updates, log lines) and
        must never abort the engine loop.
        """
        cbs = self._callbacks
        if cbs is None:
            return
        cb = getattr(cbs, name, None)
        if cb is None:
            return
        try:
            cb(*args)
        except Exception:
            print(
                f"[warning] callback {name!r} raised:",
                traceback.format_exc(),
                file=sys.stderr,
            )

    def _fire_with_return(self, name: str, *args: Any, default: Any = None) -> Any:
        """Like ``_fire`` but returns the callback's return value.

        Used by gate-style callbacks (e.g. ``on_tool_pre_execute``).
        Returns ``default`` if the callback is missing or raises.
        """
        cbs = self._callbacks
        if cbs is None:
            return default
        cb = getattr(cbs, name, None)
        if cb is None:
            return default
        try:
            return cb(*args)
        except Exception:
            print(
                f"[warning] callback {name!r} raised:",
                traceback.format_exc(),
                file=sys.stderr,
            )
            return default


__all__ = ["LoopThreadHelper"]
