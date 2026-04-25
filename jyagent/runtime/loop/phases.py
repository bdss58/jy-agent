# jyagent/phases.py — Phase-aware tool_choice shaping for the agent loop.
#
# Production agent harnesses (Cursor Agent, Aider architect mode, the
# Deep Research orchestrators surveyed on 2026-04-18) all vary their
# tool-choice policy across logical phases of a long rollout:
#
#     plan      — encourage the model to draft a task plan before acting
#                 (in jy-agent this often means calling write_todos first)
#     act       — normal tool-use, no override
#     verify    — near the end of the allowed budget, prefer wrapping up
#     finalize  — the very last step, force no tools so the model
#                 produces a natural-language answer
#
# jy-agent already does the "finalize" hack inside the max_steps fallback
# (forcing tool_choice=none on one extra call).  This module generalises
# it: a user-supplied ``PhasePolicy`` maps (step, max_steps, tool_calls)
# to a ``PhaseDirective`` that can override tool_choice for that step.
#
# Unlike the TODO scratchpad and reflection step, phase shaping does NOT
# mutate the message history — it only tweaks request options.  That
# keeps Anthropic's prefix cache fully intact (tool_choice is not part
# of the cached prefix).

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

# Phase names — kept as plain strings for ease of serialization/logging.
# (A Literal type alias is offered for call-site typing.)
PhaseName = Literal["plan", "act", "verify", "finalize"]


@dataclass(frozen=True)
class PhaseDirective:
    """The engine applies ``tool_choice`` to the next LLM call.

    ``None`` means "no override for this step"; the engine uses its
    default (no tool_choice sent, model decides).  Use ``{"type": "none"}``
    to force no tool use, ``{"type": "auto"}`` to let the model choose,
    ``{"type": "any"}`` to require a tool call, or
    ``{"type": "tool", "name": "..."}`` to pin a specific tool.
    """

    phase: str
    tool_choice: Optional[dict] = None


# Policy signature: (step, max_steps, tool_calls_count) -> PhaseDirective | None.
# Return None to fall through to engine defaults for this step.
PhasePolicy = Callable[[int, int, int], "Optional[PhaseDirective]"]


def default_phase_policy(
    *,
    plan_on_first_step: bool = False,
    verify_before_last: bool = True,
    finalize_on_last: bool = True,
) -> PhasePolicy:
    """A conservative reference policy.

    Defaults are intentionally narrow so adopting ``default_phase_policy()``
    doesn't silently change behaviour for most runs:

      * ``plan_on_first_step=False``: no hint on step 0.  Set True to
        nudge the model toward ``write_todos`` first (needs ``todos_enabled``).
      * ``verify_before_last=True``: on ``step == max_steps - 2``, reminds
        the model it has two steps left (no tool_choice override, only a
        phase label for observability).
      * ``finalize_on_last=True``: on ``step == max_steps - 1``, forces
        ``tool_choice={"type": "none"}`` so the terminal call must
        synthesise rather than start a new tool chain that can't complete.
    """

    def policy(step: int, max_steps: int, tool_calls_count: int) -> Optional[PhaseDirective]:
        # Finalize phase — the very last allowed step.  Force no tools.
        if finalize_on_last and step == max_steps - 1:
            return PhaseDirective(phase="finalize", tool_choice={"type": "none"})

        # Verify phase — one step before the last.  Observability only.
        if verify_before_last and step == max_steps - 2:
            return PhaseDirective(phase="verify", tool_choice=None)

        # Plan phase — very first step.  Let the model know.
        if plan_on_first_step and step == 0:
            return PhaseDirective(phase="plan", tool_choice=None)

        return None

    return policy


__all__ = ["PhaseDirective", "PhaseName", "PhasePolicy", "default_phase_policy"]
