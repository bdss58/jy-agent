"""Shared thread-helper mixin.

``AgentLoop`` (engine.py) and ``LLMRunner`` (llm_runner.py) both need three
identical helpers — ``_is_cancelled``, ``_cancellable_sleep``, ``_fire`` —
to consult their (different-named) ``cancel_event`` / ``callbacks``
instance attributes.  Before this mixin the methods were duplicated
verbatim across both classes (~30 lines of cut-and-paste).

Design: ``LoopThreadHelper`` reads instance attributes via class-level
string overrides.  Subclasses set the two attributes to the names they
already use, avoiding any rename / breakage to the existing public-ish
attribute surface (``AgentLoop._cancel_event`` / ``AgentLoop._callbacks``
vs. ``LLMRunner.cancel_event`` / ``LLMRunner.callbacks``).

We use single-underscore-prefixed class attribute names
(``_helper_cancel_event_attr``, ``_helper_callbacks_attr``) so Python's
double-underscore name mangling doesn't kick in — that would otherwise
rewrite ``self.__callbacks_attr__`` to ``self._AgentLoop__callbacks_attr__``
inside AgentLoop and break the lookup.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Any


class LoopThreadHelper:
    """Mixin: ``_is_cancelled`` / ``_cancellable_sleep`` / ``_fire``.

    Subclasses override the two class attributes below if their cancel
    event / callbacks instance attributes use different names.  Defaults
    match ``AgentLoop`` (the heavier consumer); ``LLMRunner`` overrides
    to its un-prefixed names.
    """

    # Name of the instance attribute holding the cancel ``threading.Event``
    # (or ``None`` when cooperative cancellation isn't wired up).
    _helper_cancel_event_attr: str = "_cancel_event"
    # Name of the instance attribute holding the ``LoopCallbacks`` dataclass.
    _helper_callbacks_attr: str = "_callbacks"

    def _is_cancelled(self) -> bool:
        """Check if external cancellation has been requested."""
        ev = getattr(self, self._helper_cancel_event_attr, None)
        return ev is not None and ev.is_set()

    def _cancellable_sleep(self, seconds: float) -> bool:
        """Sleep that returns early if cancellation is signalled.

        Returns True if cancelled during the wait, False otherwise.  When
        no cancel_event is attached, falls back to a plain blocking
        ``time.sleep``.
        """
        ev = getattr(self, self._helper_cancel_event_attr, None)
        if ev is None:
            time.sleep(seconds)
            return False
        # Event.wait returns True when set, False on timeout.
        return ev.wait(seconds)

    def _fire(self, name: str, *args: Any) -> None:
        """Invoke a named callback, swallowing exceptions.

        Callbacks are presentation-layer hooks (UI updates, log lines) and
        must never abort the engine loop.  Any raised exception is logged
        to stderr and otherwise dropped — same contract both AgentLoop
        and LLMRunner used before the mixin extraction.
        """
        cbs = getattr(self, self._helper_callbacks_attr, None)
        if cbs is None:
            return
        cb = getattr(cbs, name, None)
        if cb is not None:
            try:
                cb(*args)
            except Exception:
                # Callbacks are for presentation — never abort the engine loop.
                print(
                    f"[warning] callback {name!r} raised:",
                    traceback.format_exc(),
                    file=sys.stderr,
                )


__all__ = ["LoopThreadHelper"]
