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

Engine keeps three underscore-prefixed back-compat aliases
(``_truncate_result``, ``_compact_messages``, ``_truncate_tool_call_blocks``)
so the many existing test imports and internal call sites continue to
work unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...config import MAX_TOOL_USE_INPUT_CHARS, OBSERVATION_MASK_DISTANCE
from ...memory.conversation import estimate_conversation_tokens

if TYPE_CHECKING:
    from ..tools.registry import ToolBatch


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
    messages: list,
    max_tokens: int,
    compact_chars: int,
    batch: "ToolBatch",
) -> list:
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
        # Strip thinking blocks from all but the last few messages.
        #
        # Provider-aware rule (P0 fix): Anthropic extended-thinking emits
        # cryptographically-signed `thinking` blocks that are bound to a
        # following tool-invocation block in the same assistant message.
        # If we strip the thinking block but keep the tool block, the
        # signature becomes invalid and the provider rejects the next turn.
        # So: preserve all `thinking` blocks in any message that also
        # contains at least one tool-invocation block.  We check BOTH type
        # names because the engine sees assistant messages in their
        # normalized form (`tool_call`, per runtime.types.ToolCallBlock),
        # while raw Anthropic-SDK payloads sometimes reach compaction with
        # the provider-native `tool_use` name.  Safe across providers —
        # non-Anthropic responses simply won't have the adjacency in the
        # first place.
        _TOOL_BLOCK_TYPES = ("tool_call", "tool_use")
        content = msg.get("content", "")
        if isinstance(content, list):
            has_tool_block = any(
                isinstance(b, dict) and b.get("type") in _TOOL_BLOCK_TYPES
                for b in content
            )
            filtered_blocks = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "thinking"
                    and not has_tool_block
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
                # Full clear — observation masking
                if result_text:
                    compacted[i]["content"] = "[Tool result cleared]"
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
                            block["content"] = "[Tool result cleared]"
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
