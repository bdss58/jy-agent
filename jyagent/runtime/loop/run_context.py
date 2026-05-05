"""``RunContext`` — the structural contract ``step.py`` requires from its driver.

Background
----------
Per-step orchestration lives in ``step.py``; outer lifecycle (run lock,
terminal handlers, KeyboardInterrupt wrapping) lives in ``engine.py``
on the ``AgentLoop`` class.  Historically ``step.py`` typed its
``loop`` parameter as ``AgentLoop`` (under ``TYPE_CHECKING``) and
reached into ``loop._private`` attributes / methods directly — a hard
runtime-coupling masquerading as a clean module split.

This module makes that contract explicit.  ``RunContext`` is a
``runtime_checkable`` ``Protocol`` that lists exactly the surface
``step.py`` needs:

* configuration & callbacks (``_config``, ``_callbacks``)
* dispatch wiring (``_runtime_owner``, ``_model_spec``, ``_tool_source``,
  ``_executor``, ``_cancel_event``)
* cross-turn mutable state (``_partial_side_effects``, ``_run_id``,
  ``_todos``)
* helper methods (``_fire``, ``_fire_with_return``, ``_is_cancelled``,
  ``_call_llm_with_retry``, ``_write_checkpoint``)

``AgentLoop`` already exposes every member listed below — there is no
inheritance relationship; the Protocol is satisfied **structurally**.
Tests, replay drivers, and alternative engines that want to invoke
``run_step`` directly only need to provide the same surface.

Why keep the underscore names?
------------------------------
The members are still ``_``-prefixed because they are AgentLoop's
implementation surface — public callers of ``AgentLoop`` should never
touch them.  ``RunContext`` exists to formalize the **internal**
runtime contract between two sibling modules of the same package, not
to publish a stable external API.  If the day comes to expose a
public driver-level API, that's a *different* refactor (rename + add
public wrappers).  For now, surfacing the underscored names in the
Protocol is the most honest minimum-change description of the actual
coupling.

Lifecycle invariants
--------------------
Members declared in ``RunContext`` must remain readable for the
duration of a ``run()`` call.  The mutable state members
(``_partial_side_effects``, ``_run_id``, ``_todos``) are deliberately
exposed as attributes — ``step.py`` mutates them through that
contract.  Engine subclasses that swap a fresh deque/list mid-run
would silently break ``run_step``; don't.
"""

from __future__ import annotations

import collections
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Type-only imports — RunContext describes shapes, doesn't construct them.
    from .callbacks import LoopCallbacks
    from .config import LoopConfig
    from .llm_client import LLMClient
    from .llm_types import LLMOptions, ModelSpec, ToolCallRequest


# Tool source closure — same alias engine.py uses.  Kept local so RunContext
# stays self-contained.
ToolSource = Callable[[], tuple[list[dict], dict[str, Callable]]]


@runtime_checkable
class RunContext(Protocol):
    """Structural contract for objects that drive ``run_step``.

    ``AgentLoop`` satisfies this Protocol by virtue of declaring every
    member below; no inheritance required.  Use ``isinstance(x, RunContext)``
    only as a debugging aid — Protocols with non-method members are
    expensive to runtime-check (Python iterates every annotation).  Type
    hints are the primary use site.
    """

    # ── Configuration & callbacks ─────────────────────────────────────────
    _config: "LoopConfig"
    _callbacks: "LoopCallbacks"

    # ── LLM + tool dispatch wiring ────────────────────────────────────────
    _runtime_owner: "LLMClient"
    _model_spec: "ModelSpec | None"
    _tool_source: "ToolSource | None"
    _executor: ThreadPoolExecutor
    _cancel_event: "threading.Event | None"

    # ── Cross-turn mutable state ──────────────────────────────────────────
    # These live on the driver (not in RunState) because their lifecycle is
    # cross-turn — successive .run() calls on the same AgentLoop instance
    # see the previous turn's todos and reset partial_side_effects on entry.
    _partial_side_effects: "collections.deque[str]"
    _run_id: str
    _todos: list

    # ── Helper methods ────────────────────────────────────────────────────
    def _fire(self, event_name: str, *args: Any) -> None:
        """Dispatch a LoopCallbacks event; swallows callback exceptions."""
        ...

    def _fire_with_return(self, event_name: str, *args: Any) -> Any:
        """Dispatch a LoopCallbacks event and return the callback's value."""
        ...

    def _is_cancelled(self) -> bool:
        """True iff the cooperative cancel event has been set."""
        ...

    def _call_llm_with_retry(
        self,
        context: dict,
        options: "LLMOptions",
        step: int,
    ) -> "tuple[str, list[ToolCallRequest], str, dict]":
        """Call the LLM with transient-error retry + truncation recovery.

        Returns ``(step_text, tool_call_blocks, stop_reason, final_message)``.
        Subclass-overridable on AgentLoop — kept on the driver so test
        fakes and tier-swapping subagents preserve the contract.
        """
        ...

    def _write_checkpoint(
        self,
        *,
        step: "int | str",
        messages: list,
        total_input_tokens: int,
        total_output_tokens: int,
        tool_calls_count: int,
        status: str,
        total_cache_creation_tokens: int = 0,
        total_cache_read_tokens: int = 0,
        api_calls: int = 0,
        error: "str | None" = None,
    ) -> None:
        """Persist a LoopCheckpoint when checkpointing is enabled.

        No-op when ``_config.checkpoint_dir`` is None.  Errors logged via
        ``on_warning`` and swallowed — checkpointing must never break a run.

        The cache-token / api_calls fields default to 0 and are kw-only so
        ``RunContext`` callers may omit them; when present, they are
        recorded in ``LoopCheckpoint`` for resume-time stats reconciliation.
        """
        ...


__all__ = ["RunContext", "ToolSource"]
