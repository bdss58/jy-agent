"""Message & tool-call compaction helpers for the agent loop.

Three pure helper functions drive the loop's context-budget discipline:

* :func:`truncate_result` — shrink an oversized tool-result string with a
  head+tail window and a human-readable marker.  Error results pass
  through unchanged (we never hide errors).
* :func:`compact_messages` — multi-tier context compaction for the
  full message history (Tier 0: thinking-block pruning, Tier 1:
  observation masking, Tier 2: compaction-priority).
* :func:`truncate_tool_call_blocks` — shrink large argument fields in
  normalised assistant ``tool_call`` blocks using per-tool
  ``large_input_keys`` metadata.

Why a dedicated module?  These are **pure functions** — no closure state,
no provider I/O, no callbacks.  Keeping them out of the engine shrinks
``engine.py`` by ~200 lines and gives the compaction policy a single
import-path target for tests and external callers (e.g. analysis
tools that want to dry-run compaction on stored transcripts).

Public names: ``truncate_result``, ``compact_messages``,
``truncate_tool_call_blocks``.  Engine and step modules import these
directly from this module.  (``step.py`` aliases them locally with an
underscore prefix at a few function-scope import sites — that is a
readability convention, not a module-level back-compat shim.)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

from ...config import MAX_TOOL_USE_INPUT_CHARS, OBSERVATION_MASK_DISTANCE
from ...memory.conversation import estimate_conversation_tokens

if TYPE_CHECKING:
    from ..tools.registry import ToolBatch
    from ...llm.types import Message


# ── Dehydration — preserve recovery pointers when clearing tool results ────
#
# When an ephemeral / far-away tool result is cleared, we would normally
# replace the body with ``"[Tool result cleared]"``.  For run_shell and
# run_background, the *full* output lives in a spill file on disk
# (``/tmp/jyagent_runshell_*`` or ``/tmp/jyagent_bg_*``).  Discarding the
# path forces the agent to re-execute the command if it later needs the
# output — or worse, silently lose the data.
#
# This helper scans the result text for spill-file paths and emits a
# placeholder that tells the agent how to rehydrate on demand.  Matches
# LangChain "Deep Agents" filesystem-as-context pattern: the working set
# lives on disk, the prompt carries pointers.
#
# Pattern is restricted to the jyagent spill prefixes so we don't
# accidentally preserve arbitrary user paths that happen to look like
# tmp files.

_SPILL_PATH_RE = re.compile(
    r"(/(?:tmp|var/folders/[^\s\"'`<>]+)/jyagent_(?:runshell|bg)_[\w.\-]+)"
)


def _dehydration_placeholder(result_text: str) -> str:
    """Return the cleared-result placeholder, preserving spill-file pointers.

    Scans ``result_text`` for spill paths (jyagent's run_shell / run_background
    tmp files) and includes them so the agent can re-read the full output
    via ``run_shell cat <path>`` instead of re-running the command.
    """
    if not result_text:
        return "[Tool result cleared]"
    seen: set[str] = set()
    paths: list[str] = []
    for match in _SPILL_PATH_RE.finditer(result_text):
        p = match.group(1)
        if p not in seen:
            seen.add(p)
            paths.append(p)
        if len(paths) >= 3:  # cap to avoid pathological blowup
            break
    if not paths:
        return "[Tool result cleared]"
    joined = ", ".join(paths)
    return f"[Tool result cleared — full output on disk: {joined} (recover via run_shell: cat <path>)]"


# ── Provider-specific content-block preservation hooks ─────────────────────
#
# Some providers attach cryptographic signatures or invariants to specific
# block types that the generic compaction rules would naively strip.  The
# canonical example: Anthropic extended-thinking emits signed ``thinking``
# blocks that are bound to a following ``tool_use`` block in the same
# assistant message; stripping the thinking block invalidates the
# signature and the next provider call rejects the conversation.
#
# Provider adapters register a ``ContentPreserver`` here at import time.
# The compactor calls every active preserver on each candidate message
# and unions the protected block indices.  When no Anthropic adapter is
# loaded, the registry is empty and compaction proceeds with the
# generic rules only.
#
# Why a registry, not an inline rule?
#   * Codex's 2026-04-30 self-review flagged the inline Anthropic-
#     specific rule in compaction.py as architecturally leaky:
#     "neutrality lives in comments more than in design".  The registry
#     lets the constraint live in the adapter module that owns it.
#   * Future providers (e.g. OpenAI Responses with reasoning items, or
#     a hypothetical signed-tool-result format) can register their
#     own rules without touching this file.
#
# Contract
# --------
# ``ContentPreserver = Callable[[dict, list], frozenset[int]]``
#
# Takes the message dict and its content list.  Returns the set of
# block-indices (within ``content``) that the compactor MUST NOT strip
# in this round (Tier 0 thinking pruning).  Returning an empty
# frozenset means "no opinion" — the compactor uses its default rules.
#
# Preservers are NOT consulted by Tier 1/Tier 2 (observation masking
# and tool-result clearing) — those rules operate on tool-result blocks
# and never touch signed-thinking territory.

ContentPreserver = Callable[[dict, list], frozenset[int]]

_content_preservers: dict[str, ContentPreserver] = {}


def register_content_preserver(name: str, fn: ContentPreserver) -> None:
    """Register a provider-specific block-preservation rule.

    Idempotent: re-registering the same name replaces the previous
    function (matters for module reloads in tests).
    """
    _content_preservers[name] = fn


def unregister_content_preserver(name: str) -> None:
    """Remove a previously registered preserver.  No-op if absent."""
    _content_preservers.pop(name, None)


def _preserved_block_indices(msg: dict, content: list) -> frozenset[int]:
    """Union of every active preserver's protected indices for this message."""
    if not _content_preservers:
        return frozenset()
    indices: set[int] = set()
    for fn in _content_preservers.values():
        try:
            indices |= fn(msg, content)
        except Exception:
            # A buggy preserver must not break compaction.  Silently
            # ignore — generic rules still run.
            pass
    return frozenset(indices)


# ── Tool-result string truncation ─────────────────────────────────────────────


def truncate_result(content: str, max_chars: int, is_error: bool = False) -> str:
    """Truncate a tool result string.

    The head/tail split (85 % / 10 %) preserves the first chunk — where
    most tools put their primary output — and the tail — where summaries,
    return codes, and error lines typically live.  A human-readable
    marker documents the cut.

    Error results are **never** truncated: the engine needs the full
    stack trace to diagnose the failure, and users expect to see it.
    """
    if len(content) <= max_chars or is_error:
        return content
    head = int(max_chars * 0.85)
    tail = int(max_chars * 0.10)
    return (
        content[:head]
        + f"\n\n[... truncated {len(content) - head - tail} chars "
        + f"(total: {len(content)} chars) ...]\n\n"
        + content[-tail:]
    )


# ── Message-history compaction ────────────────────────────────────────────────


def compact_messages(
    messages: "list[Message]",
    max_tokens: int,
    compact_chars: int,
    batch: "ToolBatch",
) -> "list[Message]":
    """Multi-tier context compaction for messages sent to the LLM.

    Applied in order of aggressiveness:
      Tier 0 — Thinking block pruning: strip ``thinking`` blocks from old messages.
      Tier 1 — Observation masking: beyond ``mask_distance`` messages from the end,
               tool results are fully cleared (replaced with a short placeholder).
               Within the mask distance but outside the keep-recent zone, results
               are truncated to ``compact_chars``.
      Tier 2 — Compaction-priority awareness: "ephemeral" tool results are cleared
               more aggressively (zero chars) even within the mask distance.

    Keeps the last 2 messages fully intact (the LLM needs them for reasoning).
    Returns the original list unchanged if no modification was performed.

    ``batch`` is the per-step tool snapshot used for compaction-priority
    lookups.  Tools that have been unregistered between the step that called
    them and now degrade gracefully to ``"standard"`` priority — the
    optimisation is lost but no behaviour is incorrect.
    """
    estimated = estimate_conversation_tokens(messages)
    if estimated <= max_tokens:
        return messages

    n = len(messages)
    keep_intact = 2  # always preserve last 2 messages verbatim

    # Deep-copy to avoid mutating originals
    compacted = []
    for m in messages:
        mc = dict(m)
        c = mc.get("content")
        if isinstance(c, list):
            mc["content"] = [dict(b) if isinstance(b, dict) else b for b in c]
        compacted.append(mc)
    did_compact = False

    for i in range(n - keep_intact):
        msg = compacted[i]
        distance_from_end = n - 1 - i  # how far this message is from the newest
        far_away = distance_from_end > OBSERVATION_MASK_DISTANCE

        # ── Tier 0: Thinking block pruning ──────────────────────────
        # Strip ``thinking`` blocks from old assistant messages — they
        # carry a lot of tokens and contribute nothing to a future
        # provider call beyond the most recent few turns.
        #
        # Provider-specific preservation rules (e.g. Anthropic's signed
        # thinking blocks bound to following tool_use) are consulted via
        # the ``ContentPreserver`` registry — see the module-level
        # ``register_content_preserver`` for the mechanism.  When no
        # adapter has registered a rule (e.g. running OpenAI-only), the
        # registry is empty and every thinking block in old messages is
        # safely dropped.
        content = msg.get("content", "")
        if isinstance(content, list):
            preserved = _preserved_block_indices(msg, content)
            filtered_blocks = []
            for j, block in enumerate(content):
                if (
                    isinstance(block, dict)
                    and block.get("type") == "thinking"
                    and j not in preserved
                ):
                    did_compact = True
                    continue  # safe to drop
                filtered_blocks.append(block)
            if len(filtered_blocks) != len(content):
                compacted[i]["content"] = filtered_blocks
                content = filtered_blocks  # use filtered for subsequent tiers

        # ── Tier 1 & 2: Observation masking + priority-aware compaction ──
        # Process tool_result messages (top-level role)
        if msg.get("role") == "tool_result":
            result_text = str(msg.get("content", ""))
            tool_name = msg.get("tool_name", "")
            priority = batch.get_compaction_priority(tool_name) if tool_name else "standard"

            if far_away or priority == "ephemeral":
                # Full clear — observation masking (with rehydration pointer
                # for run_shell / run_background spill files)
                if result_text:
                    compacted[i]["content"] = _dehydration_placeholder(result_text)
                    did_compact = True
            elif len(result_text) > compact_chars and priority != "persistent":
                # Truncate to compact_chars
                compacted[i]["content"] = (
                    result_text[:compact_chars]
                    + f"\n[... compacted from {len(result_text)} chars ...]"
                )
                did_compact = True
            continue

        # Process tool_result blocks inside list content (e.g., after normalization)
        if isinstance(content, list):
            new_blocks = []
            block_changed = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    result_text = str(block.get("content", ""))
                    tool_name = block.get("tool_name", "")
                    priority = batch.get_compaction_priority(tool_name) if tool_name else "standard"

                    if far_away or priority == "ephemeral":
                        if result_text:
                            block = dict(block)
                            block["content"] = _dehydration_placeholder(result_text)
                            block_changed = True
                    elif len(result_text) > compact_chars and priority != "persistent":
                        block = dict(block)
                        block["content"] = (
                            result_text[:compact_chars]
                            + f"\n[... compacted from {len(result_text)} chars ...]"
                        )
                        block_changed = True
                new_blocks.append(block)
            if block_changed:
                compacted[i]["content"] = new_blocks
                did_compact = True

    if not did_compact:
        return messages

    return compacted


# ── Tool-call argument truncation ─────────────────────────────────────────────


def truncate_tool_call_blocks(blocks: list, batch: "ToolBatch") -> list:
    """Truncate large tool_call argument fields in normalized assistant content.

    ``batch`` is the per-step tool snapshot used for ``large_input_keys``
    metadata.  Tools not in the batch (e.g. unregistered between dispatch
    and the next assistant transformation) degrade to no-truncation —
    safer than mid-step live registry reads.
    """
    out = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_call":
            large_keys = batch.get_large_input_keys(block.get("name", ""))
            if large_keys:
                inp = block.get("arguments", {})
                truncated_inp = {}
                did_truncate = False
                for k, v in inp.items():
                    if k in large_keys and isinstance(v, str) and len(v) > MAX_TOOL_USE_INPUT_CHARS:
                        truncated_inp[k] = (
                            v[:MAX_TOOL_USE_INPUT_CHARS]
                            + f"\n[... truncated, {len(v)} chars total ...]"
                        )
                        did_truncate = True
                    else:
                        truncated_inp[k] = v
                if did_truncate:
                    block = dict(block)
                    block["arguments"] = truncated_inp
        out.append(block)
    return out


__all__ = [
    "truncate_result",
    "compact_messages",
    "truncate_tool_call_blocks",
]
