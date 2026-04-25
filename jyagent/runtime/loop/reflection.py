# jyagent/reflection.py — Mid-loop reflection / critic step.
#
# Injects a short "check-in" prompt after meaningful work boundaries so the
# model re-grounds itself on the original task before continuing.  This
# mitigates drift on long-horizon rollouts (a consistent failure mode for
# agent loops per the OpenAI Deep Research and Anthropic multi-agent posts
# we surveyed on 2026-04-18).
#
# Triggers (both configurable via LoopConfig):
#   reflect_every_n_tool_calls: N > 0 → inject after every N completed tool
#                               calls (cumulative, not per-batch).
#   reflect_after_subagent: True → inject after any batch that contained
#                           a `dispatch_agent` call.
#
# The prompt is injected as a regular user message so it lives in the
# normal conversation history.  Compaction may eventually clear it (Tier 1
# ephemeral), which is fine — its job is to nudge the next LLM call.

from __future__ import annotations

# Sentinel prefix used to detect already-injected reflection prompts so we
# never chain two back-to-back.
REFLECTION_MARKER = "<reflection-prompt>"

SUBAGENT_TOOL_NAMES = {"dispatch_agent"}


def build_reflection_prompt(reason: str, tool_calls_this_run: int) -> str:
    """Return the prompt text to inject.  Kept short to minimize token cost."""
    header_by_reason = {
        "every_n": (
            f"Progress check after {tool_calls_this_run} tool call(s)."
        ),
        "after_subagent": "Sub-agent has returned.  Integrate its output.",
    }
    header = header_by_reason.get(reason, "Progress check.")
    return (
        f"{REFLECTION_MARKER}\n"
        f"{header}\n"
        "\n"
        "Before running more tools, answer briefly (2–4 sentences total):\n"
        "  1. What concrete progress has been made toward the original task?\n"
        "  2. What is the *minimum* remaining work to finish?\n"
        "  3. Are you still on the most efficient path, or should you "
        "re-plan?\n"
        "\n"
        "If the answer to (3) is 're-plan', update your task plan (using "
        "`write_todos` if the todos tool is enabled) before proceeding.  "
        "Otherwise continue with the next necessary action.\n"
        "</reflection-prompt>"
    )


def should_reflect(
    *,
    reflect_every_n: int,
    reflect_after_subagent: bool,
    tool_calls_total: int,
    tool_calls_at_last_reflection: int,
    batch_tool_names: list[str],
    messages: list,
) -> tuple[bool, str]:
    """Decide whether to inject a reflection prompt after the current batch.

    Returns ``(inject, reason)``.  ``reason`` is one of:
      * "every_n"        — the cadence trigger fired
      * "after_subagent" — the batch contained a sub-agent dispatch
      * ""               — no trigger fired

    We deliberately don't inject when the last user message is already a
    reflection prompt (prevents back-to-back reflections when both triggers
    fire in the same batch or across batches).
    """
    # Guard against duplicate injection.
    if messages and isinstance(messages[-1], dict):
        content = messages[-1].get("content", "")
        if messages[-1].get("role") == "user":
            # String content
            if isinstance(content, str) and content.startswith(REFLECTION_MARKER):
                return False, ""
            # List-of-blocks content — inspect text blocks.
            if isinstance(content, list):
                for b in content:
                    if (
                        isinstance(b, dict)
                        and b.get("type") == "text"
                        and isinstance(b.get("text", ""), str)
                        and b["text"].startswith(REFLECTION_MARKER)
                    ):
                        return False, ""

    # (1) sub-agent trigger — priority because it has richer context.
    if reflect_after_subagent:
        if any(name in SUBAGENT_TOOL_NAMES for name in batch_tool_names):
            return True, "after_subagent"

    # (2) cadence trigger.
    if reflect_every_n and reflect_every_n > 0:
        delta = tool_calls_total - tool_calls_at_last_reflection
        if delta >= reflect_every_n:
            return True, "every_n"

    return False, ""


__all__ = [
    "REFLECTION_MARKER",
    "build_reflection_prompt",
    "should_reflect",
]
