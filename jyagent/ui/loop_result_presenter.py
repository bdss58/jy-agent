"""Map a ``LoopResult`` to user-facing output + persisted-history fields.

Extracted from ``jyagent.agent`` during the LIGHT-CLEANUP follow-up
(see journal 2026-05-11).  The main run-loop used to carry a 47-line
if/elif chain dispatching on ``result.status`` to (a) print a status
banner and (b) compute the ``(response, final_text, planner_messages)``
triple that gets persisted into ConversationMemory.

Both concerns are presentation-layer (banner styling lives in the CLI
console; ``response`` is the text we show + remember), so the whole
block lives here and ``agent.run()`` just calls
``present_loop_result(...)`` and consumes a small dataclass.

Status semantics preserved exactly; covered by existing run-loop tests.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .output import console
from .terminal import interrupted_msg

if TYPE_CHECKING:
    from ..runtime.loop.config import LoopConfig, LoopResult
    from ..llm.types import Message


@dataclass
class PresentedResult:
    """The triple ``agent.run()`` needs after a planner turn finishes."""
    response: str
    final_text: str
    planner_messages: "list[Message]"


def present_loop_result(
    result: "LoopResult",
    config: "LoopConfig",
    streaming_ui,
) -> PresentedResult:
    """Print the appropriate status banner and return the persistence triple.

    Side effects (intentional — this IS the UI layer):
      * ``"completed"`` → flush trailing newline on the streaming UI.
      * ``"max_steps"`` / ``"cost_limit"`` / ``"dedup_break"`` → print a
        bold-yellow warning under the assistant text.
      * ``"interrupted"`` → print the standard interrupted notice.
      * ``"error"`` → print the error in bold red.

    Returns ``PresentedResult(response, final_text, planner_messages)``.
    The caller persists ``response`` (assistant turn text) and renders
    ``final_text`` (just the model's narrative — no tool I/O) with
    Markdown formatting.
    """
    status = result.status

    if status == "completed":
        streaming_ui.flush_trailing_newline()
        return PresentedResult(
            response=result.text,
            final_text=result.final_text,
            planner_messages=result.messages,
        )

    if status == "max_steps":
        msg = (
            f"\n\n⚠️ Reached maximum reasoning steps ({config.max_steps}). "
            "My response may be incomplete."
        )
        sys.stdout.flush()
        console.print(f"[bold yellow]{msg}[/bold yellow]")
        response = (
            result.text
            or "I've reached my maximum reasoning steps. Please try rephrasing your request."
        )
        return PresentedResult(
            response=response,
            final_text=result.final_text,
            planner_messages=result.messages,
        )

    if status == "interrupted":
        interrupted_msg()
        return PresentedResult(
            response=result.text,
            final_text="",
            planner_messages=result.messages,
        )

    if status == "error":
        sys.stdout.flush()
        console.print(f"\n[Error: {result.error}]", style="bold red", markup=False)
        if result.text:
            response = result.text + f"\n\n[Error: {result.error}]"
        else:
            response = f"Error during planning: {result.error}"
        return PresentedResult(
            response=response,
            final_text="",
            planner_messages=result.messages,
        )

    if status == "cost_limit":
        cost_msg = f"\n\n⚠️ {result.error}"
        sys.stdout.flush()
        console.print(cost_msg, style="bold yellow", markup=False)
        response = result.text + cost_msg if result.text else cost_msg
        return PresentedResult(
            response=response,
            final_text=result.final_text,
            planner_messages=result.messages,
        )

    if status == "dedup_break":
        dedup_msg = "\n\n⚠️ Loop detected — stopped to prevent infinite loop."
        sys.stdout.flush()
        console.print(dedup_msg, style="bold yellow", markup=False)
        response = result.text + dedup_msg if result.text else dedup_msg
        return PresentedResult(
            response=response,
            final_text=result.final_text,
            planner_messages=result.messages,
        )

    # Fallback — should not happen if all LoopResult statuses are listed above.
    return PresentedResult(
        response=result.text or "Unknown error",
        final_text="",
        planner_messages=result.messages,
    )


__all__ = ["PresentedResult", "present_loop_result"]
