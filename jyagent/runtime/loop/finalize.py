"""Terminal-path helpers for the agent loop.

Extracted from ``engine.py`` so that ``step.py`` (and any future
sub-module that needs to terminate a run) can import these without
reaching back into ``engine.py``.  Before this split, ``step.py``
imported helpers from ``.engine`` — that closed a circular conceptual
dependency (engine owns step, step reaches back into engine for
terminal helpers).

Three helpers live here:

  * ``is_truncated`` — pure predicate over ``(stop_reason, tool_calls)``;
    classifies a provider response as a mid-tool-call truncation that
    should trigger token-budget rescaling and a retry.

  * ``strip_dangling_verification`` — idempotent in-place cleaner for
    the trailing ``[VERIFICATION]`` user message injected by
    ``verification.py``.  Run on every terminal path so an early exit
    does not leak an unanswered self-check prompt into persisted
    sessions.

  * ``finalize_run`` — the canonical ``_run_impl`` exit path.  Funnels
    every terminal ``LoopResult`` construction through one place:
    strips dangling verification, finishes + flushes the trace span
    (best-effort), and returns a populated ``LoopResult``.

These used to carry ``_``-prefixed names in ``engine.py`` while they
were module-private.  Both the extraction and the subsequent rename
landed in one branch — no alias bridge, no dual-name surface.  Engine
and step both import the public names directly.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from .config import LoopResult
from .llm_types import ToolCallRequest

if TYPE_CHECKING:
    from ...llm.types import Message

_logger = logging.getLogger(__name__)


def is_truncated(stop_reason: str, tool_calls: list[ToolCallRequest]) -> bool:
    """Detect if a response was truncated while emitting tool calls."""
    return stop_reason == "length" and bool(tool_calls)


# ── Prose-shaped tool call detector (Bug A — see fix plan 2026-05) ──────────
#
# The agent loop only continues when the assistant response carries a
# structured tool_use block (or stop_reason == "tool_use").  When the model
# instead WRITES what looks like a tool call into its assistant text — e.g.
# ``[Tool call: run_shell]{"cmd": "ls"}`` or a fenced ```tool_use``` block —
# there are no tool_call_blocks and the loop terminates the turn cleanly.
# From the user's perspective the agent "suddenly stopped mid-task".
#
# This predicate is a high-precision detector for that specific class of
# malformed-attempt prose.  The patterns are deliberately narrow:
#
#   1. ``[Tool call: NAME]`` or ``[Tool: NAME]`` at the start of a line.
#      This is the literal Claude-Code transcript-render syntax that the
#      model occasionally falls back to when it's narrating a tool use
#      instead of invoking one.
#   2. A fenced code block whose info-string is ``tool_use`` / ``tool_call``.
#      Same intent, different rendering.
#
# Anchoring matters.  Discussing tool calls in prose ("I'll run the
# `run_shell` tool to ...") must NOT trigger; only line-leading invocation-
# shaped strings do.  A false positive here means re-prompting a turn that
# was actually fine — annoying but not silently broken.  A false negative
# means the original Bug A reappears — which is what we already had.
#
# Returning True does NOT terminate or alter state; the caller decides the
# remediation (currently: inject a corrective user message and continue the
# loop).  Keeping this as a pure predicate keeps it trivially testable.
_PROSE_TOOL_CALL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Line-leading "[Tool call: NAME]" / "[Tool: NAME]" / "[tool_use: NAME]".
    # Tolerates surrounding brackets, optional space, common case variants.
    re.compile(r"(?im)^\s*\[\s*tool(?:[ _-]?call|[ _-]?use)?\s*:\s*[A-Za-z_][\w-]*\s*\]"),
    # Fenced code block tagged tool_use / tool_call (info string after ```).
    re.compile(r"(?im)^\s*```\s*tool[_-]?(?:use|call)\b"),
)


def looks_like_prose_tool_call(text: str) -> bool:
    """Detect when assistant prose appears to be a malformed tool invocation.

    Returns True iff ``text`` contains a high-confidence pattern that the
    model TRIED to emit a tool call but did so as prose instead of a real
    structured tool_use block.  Used by ``run_step`` to decide whether to
    inject a corrective re-prompt rather than terminating the turn.

    Conservative by design — see module-level comment for the patterns
    matched and the false-positive / false-negative trade-off.  Any future
    pattern additions belong in ``_PROSE_TOOL_CALL_PATTERNS``.
    """
    if not text:
        return False
    return any(p.search(text) for p in _PROSE_TOOL_CALL_PATTERNS)


# Sentinel marker (parallel to ``_VERIFICATION_MARKER`` in verification.py)
# that prefixes the corrective user message injected when a prose tool call
# is detected.  Used to (a) detect repeats / break out if the model keeps
# doing it and (b) strip a dangling unanswered correction at terminal
# cleanup time.  Public so ``step.py`` can compose the message and the
# strip helper can recognize it.
PROSE_TOOL_CALL_MARKER = "[MALFORMED_TOOL_CALL]"


def build_prose_tool_call_correction() -> str:
    """Build the corrective user message injected when Bug A fires.

    The message tells the model exactly what went wrong and what to do
    next.  Kept terse — verbose prompts here just consume tokens that the
    next attempt needs.  Returns a string starting with
    ``PROSE_TOOL_CALL_MARKER`` so the gate can detect repeated firings.
    """
    return (
        f"{PROSE_TOOL_CALL_MARKER} Your previous message contained text that "
        "looks like a tool call (e.g. `[Tool call: ...]` or a ```tool_use``` "
        "block) but no actual structured tool invocation was emitted, so the "
        "tool did NOT run.  If you intended to call a tool, invoke it via "
        "the real function-calling channel now.  If you intended plain prose, "
        "continue without the pseudo-syntax."
    )


# ── Empty-assistant-turn detector (Bug B — see session fe07d3bc, 2026-05-14) ─
#
# Failure mode: the provider returned a structurally-valid assistant
# response that nevertheless carried NO visible output — e.g. only an
# empty extended-thinking block:
#
#     {"role": "assistant",
#      "content": [{"type": "thinking", "thinking": "", "signature": ""}],
#      "stop_reason": "stop",
#      "usage": {"output_tokens": 63, ...}}
#
# ``extract_text`` (which sums only ``type == "text"`` blocks) yields
# ``step_text == ""``, ``tool_call_blocks == []``, and the no-tool branch
# of ``run_step`` accepts it as a clean terminal completion.  From the
# user's perspective the agent silently halted mid-task right after a
# tool result — the exact "sudden stop while prompting" symptom from
# session ``fe07d3bc-e5dc-49c3-aa7e-5190c2b98b72`` (1014 events, last
# assistant turn at seq=1014 was an empty thinking block).
#
# Detector: assistant content has zero non-empty text blocks AND zero
# tool_use blocks.  Thinking blocks (even non-empty ones) do NOT count as
# visible output because the user never sees them.  We deliberately do
# NOT key off ``stop_reason`` — Anthropic, OpenAI, and Gemini all use
# slightly different terminal strings (``stop`` / ``end_turn`` /
# ``stop_sequence``), and the symptom is the absence of output, not the
# reason string.
#
# Returning True is purely advisory — caller decides remediation.  Like
# the prose-tool-call gate the recovery is a one-shot "continue" prompt,
# bounded by a per-run cap so a model that genuinely has nothing to say
# isn't pestered forever.
# Content-block types that the user NEVER sees in the rendered transcript
# and that therefore do NOT count as "visible output" for the empty-turn
# detector.  This list is the conservative core of the gate (Codex
# review, 2026-05-14): if a future provider adds a NEW visible block type
# (e.g. ``image``, ``citation``, ``ui_card``), we MUST NOT treat a turn
# carrying only that block as empty — falsely retrying would erase real
# output and pester the model.  Hence the rule is:
#
#   * every block whose ``type`` is in this set → invisible.
#   * every block whose ``type`` is NOT in this set → visible (don't fire).
#
# Keep the set tight.  Add a type here ONLY after confirming end users
# truly never see it (e.g. it is stripped by the renderer before display).
_INVISIBLE_BLOCK_TYPES: frozenset[str] = frozenset({
    "thinking",
    "redacted_thinking",
})


def looks_like_empty_turn(
    step_text: str,
    tool_call_blocks: list,
    final_message: "Message | dict | None",
) -> bool:
    """Detect an assistant turn that produced no visible output AND no tool call.

    Returns True iff:

      * ``tool_call_blocks`` is empty (no structured tool invocation), AND
      * ``step_text`` (concatenated visible text blocks) is empty or
        whitespace-only, AND
      * the assistant ``final_message`` carries no non-empty ``text``
        content block AND no content block whose ``type`` is OUTSIDE the
        invisible-types allowlist.  I.e. every present block must be a
        known-invisible kind (``thinking`` / ``redacted_thinking``) or an
        empty ``text``.

    The allowlist contract (Codex review, 2026-05-14) is the important
    half: a future visible block type (e.g. ``image``, ``citation``)
    must NOT register as empty, because retrying would erase real
    output.  Unknown block types fall through to "visible" by default,
    keeping the detector safely on the no-false-fire side.

    Cheap — short-circuits as soon as any visible content is seen.
    """
    if tool_call_blocks:
        return False
    if step_text and step_text.strip():
        return False
    if not isinstance(final_message, dict):
        # No structured message to inspect (e.g. provider returned a bare
        # string).  ``step_text`` was already empty → still an empty turn.
        return True
    content = final_message.get("content")
    if not isinstance(content, list):
        # Same reasoning as above: no structured content list means we
        # only have the (already-empty) ``step_text`` to go on.
        return True
    for block in content:
        if not isinstance(block, dict):
            # A non-dict block is an unknown shape — treat as visible to
            # stay on the safe side (don't fire).
            return False
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if isinstance(text, str) and text.strip():
                return False
            # Empty text block → invisible, keep scanning.
            continue
        if btype in _INVISIBLE_BLOCK_TYPES:
            continue
        # Unknown / non-text / non-thinking block type → treat as
        # visible.  This is the Codex-review guard against silently
        # discarding future provider features.
        return False
    return True


# Sentinel marker for the corrective user message injected when an empty
# assistant turn is detected.  Public so ``step.py`` can compose the
# message and ``strip_dangling_verification`` can recognize an unanswered
# correction at terminal cleanup time.
EMPTY_TURN_MARKER = "[EMPTY_TURN]"


def build_empty_turn_correction() -> str:
    """Build the corrective user message injected when Bug B fires.

    Terse on purpose — verbose prompts here just steal tokens from the
    retry.  Starts with ``EMPTY_TURN_MARKER`` so the gate can detect
    repeats and ``strip_dangling_verification`` can drop a dangling one.
    """
    return (
        f"{EMPTY_TURN_MARKER} Your previous response was empty — no visible "
        "text and no tool call.  Please continue from where you left off: "
        "either invoke the next tool, or summarize the result so the user "
        "can see it."
    )


def strip_dangling_verification(messages: "list[Message]") -> None:
    """Remove a trailing unanswered recovery-marker user message in-place.

    Historically scoped to ``[VERIFICATION]`` — hence the name, which is
    retained for call-site stability.  The function now strips any of
    three recovery-marker families, all of which share the same failure
    mode (a corrective user prompt was injected and the loop terminated
    before the model replied) and the same fix (drop the dangling tail):

      * ``[VERIFICATION]`` — verification gate self-check prompt.
      * ``[MALFORMED_TOOL_CALL]`` — Bug A: prose-shaped tool-call
        correction.  See ``looks_like_prose_tool_call``.
      * ``[EMPTY_TURN]`` — Bug B: empty-assistant-turn correction.
        See ``looks_like_empty_turn``.

    If any future recovery gate adds a marker, register it in the
    ``startswith`` block below AND add a regression test.  Leaving a
    dangling correction in the persisted tail poisons the next turn.

    Idempotent: safe to call on every terminal path regardless of whether
    a correction was actually injected.  This is why the canonical exit
    helper ``finalize_run`` calls it unconditionally — gating on a
    per-marker ``*_injected`` flag is a micro-optimization that
    historically led to bugs (cleanup forgotten on new exit paths).
    """
    if not messages:
        return
    tail = messages[-1]
    if not isinstance(tail, dict):
        return
    if tail.get("role") != "user":
        return
    tail_content = tail.get("content", "")
    if isinstance(tail_content, str) and (
        tail_content.startswith("[VERIFICATION]")
        or tail_content.startswith(PROSE_TOOL_CALL_MARKER)
        or tail_content.startswith(EMPTY_TURN_MARKER)
    ):
        messages.pop()


def finalize_run(
    *,
    status: str,
    text: str,
    final_text: str,
    messages: "list[Message]",
    steps: int,
    total_input_tokens: int,
    total_output_tokens: int,
    tool_calls_count: int,
    total_cache_creation_tokens: int = 0,
    total_cache_read_tokens: int = 0,
    api_calls: int = 0,
    error: str | None = None,
    trace: Any = None,
    trace_status: str | None = None,
    trace_total_steps: int | None = None,
    trace_total_cost_usd: float | None = None,
) -> LoopResult:
    """Centralized exit path for ``_run_impl``.

    Every ``return LoopResult(...)`` in the loop must funnel through here so
    that:

      1. Dangling ``[VERIFICATION]`` user messages are *always* stripped
         (idempotent — see ``strip_dangling_verification``).  Historically
         this was open-coded at every exit, and three exit paths
         (``cost_limit``, repeated truncation, cooperative cancellation)
         were missed, leaking unanswered prompts into persisted sessions.

      2. Trace finish + flush happens uniformly, eliminating exit paths
         that emitted a ``LoopResult`` but never closed the trace span.

    The ``trace_*`` overrides exist for cases where the trace status string
    or step count differs from the ``LoopResult`` (currently only
    ``max_steps`` uses ``trace_total_steps=cfg.max_steps`` while reporting
    ``steps=cfg.max_steps`` — both happen to match, but the override keeps
    the seam explicit for future use).

    Keyword-only by design: every field is named at the call site so that
    a careless ``LoopResult(*args)`` style cannot regress the contract.
    """
    strip_dangling_verification(messages)
    if trace is not None:
        finish_kwargs: dict = {
            "status": trace_status or status,
            "total_steps": trace_total_steps if trace_total_steps is not None else steps,
        }
        if trace_total_cost_usd is not None:
            finish_kwargs["total_cost_usd"] = trace_total_cost_usd
        # Tracing must never fail-close a successful run.  Disk-full /
        # read-only fs / permission errors here used to bubble up and
        # discard the entire LoopResult.  Log + swallow so observability
        # stays non-fatal.
        try:
            trace.finish(**finish_kwargs)
            trace.flush()
        except Exception as trace_err:  # noqa: BLE001 — observability is best-effort
            _logger.warning(
                "trace finalize failed (non-fatal): %s: %s",
                type(trace_err).__name__,
                trace_err,
            )
    return LoopResult(
        status=status,
        text=text,
        final_text=final_text,
        messages=messages,
        steps=steps,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        tool_calls_count=tool_calls_count,
        total_cache_creation_tokens=total_cache_creation_tokens,
        total_cache_read_tokens=total_cache_read_tokens,
        api_calls=api_calls,
        error=error,
    )


__all__ = [
    "build_empty_turn_correction",
    "build_prose_tool_call_correction",
    "EMPTY_TURN_MARKER",
    "finalize_run",
    "is_truncated",
    "looks_like_empty_turn",
    "looks_like_prose_tool_call",
    "PROSE_TOOL_CALL_MARKER",
    "strip_dangling_verification",
]
