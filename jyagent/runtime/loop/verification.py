"""Pre-completion verification gate for the agentic loop engine.

When the model stops calling tools and is about to return a final answer,
this module decides whether to inject a self-check prompt so the model can
catch obvious issues before completing.

Disabled by default. Opt-in via AGENT_VERIFICATION_ENABLED=1.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Imported lazily to avoid a circular import: verification.py is
    # imported from runtime.loop modules, and ToolBatch lives under
    # runtime.tools — pulling it eagerly would couple the verification
    # gate to the tools package at module load.
    from ..tools.registry import ToolBatch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VERIFICATION_ENABLED: bool = os.environ.get(
    "AGENT_VERIFICATION_ENABLED", ""
).lower() in ("1", "true", "yes")

# Sentinel content prefix used to detect an already-injected verification
# prompt so we never inject twice in the same turn.
_VERIFICATION_MARKER = "[VERIFICATION]"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def should_verify(
    messages: list[dict[str, Any]],
    tool_calls_count: int,
    *,
    since_index: int = 0,
    batch: "ToolBatch",
) -> bool:
    """Return True if a verification prompt should be injected.

    Criteria – **all** must be true:
    1. ``VERIFICATION_ENABLED`` is set.
    2. At least one tool call was made this turn.
    3. The turn involved tools with the registry's ``mutating=True`` flag
       (run_shell, edit_file, write_file, mcp, dispatch_agent,
       run_background, plus any dynamic MCP tool that registers as
       mutating).  Classification is done via ``batch.is_mutating(name)``
       so a freshly-registered MCP mutator surfaces the gate without a
       static keyword change here.
    4. A verification prompt has not already been injected this turn.

    ``since_index`` (default 0, i.e. scan everything) bounds the mutation
    scan to messages appended during *this* turn — pass the value of
    ``len(messages)`` captured at the start of ``_run_impl``.  Without
    this, a replayed historical mutation in prior turns would re-arm the
    verification gate on a non-mutating new turn.

    ``batch`` is the immutable per-step ``ToolBatch`` snapshot the engine
    already builds for dispatch — re-using it keeps the mutating-set
    classification consistent with the dispatch loop's other ``mutating``
    consumers (timeout warning, partial-side-effects accumulator).
    """
    if not VERIFICATION_ENABLED:
        return False

    if tool_calls_count < 1:
        return False

    if _already_injected(messages):
        return False

    if not _has_mutation(messages, since_index=since_index, batch=batch):
        return False

    return True


def build_verification_prompt(messages: list[dict[str, Any]]) -> str:
    """Build a concise self-check prompt to inject as a user message.

    The returned string starts with ``_VERIFICATION_MARKER`` so that
    ``should_verify`` can detect it on subsequent calls.
    """
    return (
        f"{_VERIFICATION_MARKER} Before you finish, review what you just did:\n"
        "\n"
        "1. **Syntax check** — re-read every code block you wrote or edited. "
        "Are there typos, unclosed brackets, bad indentation, or import errors?\n"
        "2. **Correctness** — do the changes actually address the original request? "
        "Did you miss any requirements or edge cases mentioned by the user?\n"
        "3. **Tests** — should any existing tests be run to confirm nothing broke? "
        "If so, run them now.\n"
        "4. **Consistency** — are all modified files in a valid state? No leftover "
        "debug prints, no placeholder TODOs that should have been filled in?\n"
        "\n"
        "If everything looks correct, provide your final response. "
        "If you find issues, fix them now using the available tools."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _already_injected(messages: list[dict[str, Any]]) -> bool:
    """Check whether a verification prompt was already injected this turn."""
    # Walk backwards — the marker would be near the tail of the conversation.
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        # content may be a string or a list of content blocks
        if isinstance(content, str):
            if content.startswith(_VERIFICATION_MARKER):
                return True
        elif isinstance(content, list):
            for block in content:
                text = block.get("text", "") if isinstance(block, dict) else ""
                if text.startswith(_VERIFICATION_MARKER):
                    return True
        # Only inspect the most recent user message — if it isn't the marker
        # we haven't injected yet (or the model has replied since).
        break
    return False


def _has_mutation(
    messages: list[dict[str, Any]],
    *,
    since_index: int = 0,
    batch: "ToolBatch",
) -> bool:
    """Return True if any tool-result message references a mutating tool.

    Bounded to ``messages[since_index:]`` so a replayed mutation from a
    prior turn (still present in the persisted history) does not re-arm
    the verification gate on a new, non-mutating turn.

    Mutating-classification is done via ``batch.is_mutating(name)`` —
    the per-step ToolBatch snapshot built by the engine.  This is the
    same source of truth the dispatch loop uses for timeout-warning and
    partial-side-effects accumulation, so a tool that's flagged
    mutating in one place is flagged mutating everywhere.  Unknown names
    (e.g. tools unregistered between the call and the verification scan)
    default to False — the historical default for the gate.
    """
    for msg in messages[since_index:]:
        # Anthropic format: role "user" with tool_result content blocks
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and batch.is_mutating(block.get("tool_name", ""))
                    ):
                        return True

        # OpenAI / normalized format: role "tool" with a name field
        if msg.get("role") == "tool":
            if batch.is_mutating(msg.get("name", "")):
                return True

        # Also support an explicit "tool_name" key used by some adapters
        if batch.is_mutating(msg.get("tool_name", "")):
            return True

    return False
