# loop_engine.py — Reusable agentic tool-use loop engine.
#
# Shared algorithm for both planner (streaming, full-featured) and sub-agent
# (non-streaming, silent).  Callers configure behaviour via LoopConfig and
# LoopCallbacks; the engine never writes to stdout directly.

from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import random
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from .runtime import RuntimeOwner, RuntimeOptions
from .runtime.types import ModelSpec
from .config import get_reasoning_config_for_provider, STREAM_TIMEOUT, MAX_TOOL_USE_INPUT_CHARS
from .registry import get_registry
from .toolresult import ToolResult
from .validation import validate_tool_input
from .memory.conversation import estimate_conversation_tokens
from .remediation import enrich_error
from .tracing import get_tracer
from .verification import should_verify, build_verification_prompt


# ─── Core types ──────────────────────────────────────────────────────────────

@dataclass
class LoopConfig:
    max_steps: int = 50
    initial_max_tokens: int = 16_384
    max_tokens_cap: int = 128_000
    auto_scale_on_truncation: bool = True
    token_scale_factor: int = 2
    concurrent_tools: bool = True
    max_tool_workers: int = 4
    tool_timeout: int = 120
    retry_attempts: int = 3
    retry_base_delay: float = 1.0
    compact_messages: bool = True
    max_working_tokens: int = 100_000
    compact_tool_result_chars: int = 2000
    max_tool_result_chars: int = 8000
    streaming: bool = False
    truncate_large_inputs: bool = True
    fallback_on_max_steps: bool = False
    # When True, streaming text deltas are buffered per-attempt and only
    # flushed via on_text_delta after a clean `done` event.  Eliminates
    # visual duplication on transient-error retry and truncation recovery
    # at the cost of losing live-token UX.  Off by default.
    buffered_streaming: bool = False
    # Persistent task-plan scratchpad (see jyagent/todos.py).  When True:
    #   * a per-loop `write_todos` tool is overlaid onto the tool source
    #     so the model can create / update the plan;
    #   * the current plan is rendered as a <system-reminder> block
    #     appended to the tail user message before each LLM call — NOT
    #     persisted into the messages list, so it survives compaction
    #     automatically.
    todos_enabled: bool = False
    # Mid-loop reflection / critic step (see jyagent/reflection.py).  Injects
    # a short progress-check prompt after every N tool calls and/or after
    # any batch that dispatched a sub-agent.  Both triggers OFF by default.
    reflect_every_n_tool_calls: int = 0   # 0 disables the cadence trigger
    reflect_after_subagent: bool = False
    # Phase-aware tool_choice shaping (see jyagent/phases.py).  When set,
    # the policy is consulted each step and may override tool_choice for
    # that LLM call (plan / act / verify / finalize).  Does NOT mutate the
    # message history — keeps Anthropic prefix caching fully intact.
    phase_policy: Any = None  # PhasePolicy | None — typed as Any to avoid cycles
    # Checkpointed replay (see jyagent/checkpoint.py).  When both are set,
    # LoopCheckpoint is written every N steps (and on terminal exits) to
    # ``<checkpoint_dir>/<run_id>/step_NNNN.json``.  Off by default.
    checkpoint_dir: str | None = None
    checkpoint_every_n_steps: int = 0
    # Harness controls
    max_cost_usd: float | None = None       # cost budget per turn — None = unlimited
    dedup_threshold: int = 3                 # same tool+args+response N times → break loop


@dataclass
class LoopCallbacks:
    # All Optional[Callable].  None = silent (sub-agent mode).
    on_text_delta: Callable[[str], None] | None = None
    on_thinking_start: Callable[[], None] | None = None
    on_thinking_stop: Callable[[], None] | None = None
    on_tool_start: Callable[[str, dict], None] | None = None
    on_tool_end: Callable[[str, str, bool], None] | None = None  # (name, content, is_error)
    on_retry: Callable[[int, Exception], None] | None = None  # (attempt, error)
    on_compaction: Callable[[int, int], None] | None = None  # (before_len, after_len)
    on_usage: Callable[[dict], None] | None = None  # raw Usage dict from response
    on_step_progress: Callable[[int, int], None] | None = None  # (step, max_steps)
    on_assistant_message: Callable[[dict], dict] | None = None  # transform before append
    on_warning: Callable[[str], None] | None = None  # runtime warnings
    on_truncation: Callable[[], None] | None = None  # response truncated, retrying
    on_tool_batch: Callable[[int], None] | None = None  # number of tools in batch
    # Fires once before a stream retry (transient error OR truncation recovery).
    # `reason` is "transient_error" or "truncation".  `partial_text` is whatever
    # the current attempt had emitted before the restart — UIs can use this to
    # mark, clear, or strikethrough the duplicated output that will follow.
    on_stream_retry: Callable[[str, str], None] | None = None  # (reason, partial_text)
    # Fires when the engine injects a mid-loop reflection prompt.  `reason`
    # is "every_n" or "after_subagent".  Callback is purely observational —
    # does not affect the loop.
    on_reflection: Callable[[str], None] | None = None
    # Fires when a PhasePolicy assigns a phase to the current step.  `phase`
    # is a short string ("plan" | "act" | "verify" | "finalize" | custom).
    # Observational only.
    on_phase_enter: Callable[[str], None] | None = None
    # Fires after a checkpoint is persisted.  Args: (path, step).  Use
    # for logging / progress display.  Step < 0 indicates a terminal
    # checkpoint (e.g. final.json).
    on_checkpoint: Callable[[str, int], None] | None = None


@dataclass
class LoopResult:
    status: str  # "completed" | "max_steps" | "error" | "interrupted" | "cost_limit" | "dedup_break"
    text: str
    final_text: str
    messages: list
    steps: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_calls_count: int = 0
    error: str | None = None
    # Final state of the task-plan scratchpad.  Empty list when todos are
    # disabled or the model never wrote any.  Outer layers (agent.py) are
    # expected to persist this across turns.
    todos: list = field(default_factory=list)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    input: dict


# Type alias: returns (schemas_list, functions_dict)
ToolSource = Callable[[], tuple[list[dict], dict[str, Callable]]]


def _t_as_dict(t: Any) -> dict:
    """Best-effort TodoItem → dict.  Tolerates raw dicts already."""
    if isinstance(t, dict):
        return t
    try:
        from dataclasses import asdict
        return asdict(t)
    except Exception:
        return {"content": str(getattr(t, "content", t))}


# ─── Shared executors ────────────────────────────────────────────────────────
# Tool execution uses two distinct mechanisms:
#
#   1. `_tool_dispatch_executor` — fixed-size ThreadPoolExecutor used by
#      `_execute_tools()` to fan out a parallel-safe batch.  Order is
#      preserved via index-keyed slots.
#
#   2. Daemon thread per invocation — used inside `_execute_tool_with_timeout`
#      to run the tool body with a hard wall-clock deadline.  Previously this
#      was a second ThreadPoolExecutor (`_tool_body_executor`), but Python
#      thread futures are not cancellable — a timed-out body permanently
#      leaked a worker slot.  Daemon threads hold no pool slot and die with
#      the process, eliminating the leak.
#
# `_tool_body_executor` is retained as a module attribute pointing at a
# small pool for any external caller that grabs it directly (e.g. legacy
# tests) but the engine itself no longer uses it.

_tool_dispatch_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="jyagent-tool-dispatch",
)
_tool_body_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="jyagent-tool-body-legacy",
)
atexit.register(_tool_dispatch_executor.shutdown, wait=False)
atexit.register(_tool_body_executor.shutdown, wait=False)

# Backwards-compat alias (callers that pass a pool explicitly).
_tool_executor = _tool_dispatch_executor


# ─── Private helpers ─────────────────────────────────────────────────────────


# ─── Harness helpers ─────────────────────────────────────────────────────────

class _CostTracker:
    """Track estimated cost within a single run() for budget enforcement.

    Uses the session_stats pricing machinery but keeps a local running total
    so the loop can check against LoopConfig.max_cost_usd each step.
    """

    def __init__(self):
        self.total_cost: float = 0.0
        self._has_unknown = False

    def record(self, usage: dict, provider: str, model: str) -> None:
        from .session_stats import _lookup_pricing
        pricing = _lookup_pricing(provider, model) if provider and model else None
        if pricing is None:
            if any(usage.get(k, 0) for k in ("input_tokens", "output_tokens")):
                self._has_unknown = True
            return
        input_t = usage.get("input_tokens", 0) or 0
        output_t = usage.get("output_tokens", 0) or 0
        cache_create = usage.get("cache_creation_input_tokens", 0) or 0
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        self.total_cost += (
            input_t * pricing.input_per_million / 1_000_000
            + output_t * pricing.output_per_million / 1_000_000
            + cache_create * (pricing.cache_creation_per_million or 0.0) / 1_000_000
            + cache_read * (pricing.cache_read_per_million or 0.0) / 1_000_000
        )

    @property
    def known_cost(self) -> float | None:
        """Return cost or None if any usage was unpriced."""
        return None if self._has_unknown else self.total_cost


class _StuckLoopDetector:
    """Detect stuck loops by tracking whether repeated calls yield new responses.

    Key insight: a loop is "stuck" only when the same tool call returns the
    same response **consecutively**.  Polling tools (``check_background``,
    ``take_snapshot``) naturally return changing responses (e.g. different
    ``elapsed_seconds``) — they are never flagged without any exemption metadata.

    Interleaved calls are also safe: if the agent alternates
    ``run_shell(A) → check_background → run_shell(A) → check_background``
    that's a polling pattern, not a stuck loop — even if ``run_shell(A)``
    returns the same result each time.  Only **truly consecutive** identical
    calls (``A → A → A``) trigger the detector.

    This replaces the old ``_DedupTracker`` which required a whitelist of
    ``dedup_exempt`` tools and a regex hack for ``sleep`` commands.

    Design:
        Track ``(tool_name, args_key) → (consecutive_identical_count, last_response_hash)``

        * If a **different** key was recorded since the last call to *this* key,
          the pattern is interleaved — reset the counter (not a stuck loop).
        * If the response hash differs from the last recorded one for the same
          ``(tool, args)`` key, the world is making progress — reset the counter.
        * If the response hash is identical, increment the counter.
        * At ``threshold``: return a feedback message so the engine can break.
    """

    def __init__(self, threshold: int = 3):
        # key → (consecutive_identical_count, last_response_hash)
        self._state: dict[str, tuple[int, str]] = {}
        self._threshold = threshold
        self._last_key: str | None = None

    @staticmethod
    def _make_key(name: str, args: dict) -> str:
        """Stable string key for a tool call."""
        try:
            args_str = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args_str = str(args)
        return f"{name}::{args_str}"

    @staticmethod
    def _hash_response(content: str) -> str:
        return hashlib.md5(content.encode(errors="replace")).hexdigest()

    def record(self, name: str, args: dict, response: str) -> str | None:
        """Record a single (tool, args, response) observation.

        Returns a feedback message when a stuck loop is detected (same tool
        called with identical arguments AND identical response ``threshold``
        times **truly consecutively**), or ``None`` if everything is fine.

        "Truly consecutive" means no other ``(tool, args)`` key was recorded
        in between.  Interleaved patterns like ``A → B → A → B → A`` never
        trigger — they represent polling, not a stuck loop.
        """
        key = self._make_key(name, args if isinstance(args, dict) else {})
        resp_hash = self._hash_response(response)

        prev_count, prev_hash = self._state.get(key, (0, ""))

        # If a different tool/args was called since our last record() call,
        # this is an interleaved pattern (e.g. polling).  Reset the counter
        # for this key so it starts fresh.
        if self._last_key is not None and self._last_key != key:
            prev_count, prev_hash = 0, ""

        self._last_key = key

        if prev_hash and resp_hash != prev_hash:
            # Response changed — progress is being made, reset.
            self._state[key] = (1, resp_hash)
            return None

        # Response identical (or first observation) — increment.
        new_count = prev_count + 1
        self._state[key] = (new_count, resp_hash)

        if new_count >= self._threshold:
            return (
                f"STUCK LOOP: Tool '{name}' was called {new_count} times with "
                f"identical arguments AND identical response.  The external "
                f"state is not changing.  Stop repeating this call and try a "
                f"different approach, or explain to the user why you're stuck."
            )
        return None


# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(message: dict) -> str:
    """Extract concatenated text blocks from an AssistantMessage."""
    return "".join(
        block.get("text", "")
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _extract_tool_calls(message: dict) -> list[ToolCallRequest]:
    """Extract tool_call blocks from an AssistantMessage."""
    return [
        ToolCallRequest(
            id=block["id"],
            name=block["name"],
            input=block.get("arguments", {}),
        )
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_call"
    ]


def _is_truncated(stop_reason: str, tool_calls: list[ToolCallRequest]) -> bool:
    """Detect if a response was truncated while emitting tool calls."""
    return stop_reason == "length" and bool(tool_calls)


def _truncate_result(content: str, max_chars: int, is_error: bool = False) -> str:
    """Truncate a tool result string.  Error results are never truncated."""
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


def _compact_messages(
    messages: list,
    max_tokens: int,
    compact_chars: int,
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
    """
    from .config import OBSERVATION_MASK_DISTANCE

    estimated = estimate_conversation_tokens(messages)
    if estimated <= max_tokens:
        return messages

    registry = get_registry()
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
        # Provider-aware rule (P0 fix, 2026-04): Anthropic extended-thinking
        # emits cryptographically-signed `thinking` blocks that are bound to
        # the following `tool_use` block in the same assistant message.  If
        # we strip the thinking block but keep the tool_use, the signature
        # becomes invalid and the provider rejects the next turn.  So:
        # preserve all `thinking` blocks in any message that also contains
        # at least one `tool_use` block.  Safe across providers — non-
        # Anthropic responses simply won't have the adjacency in the first
        # place.
        content = msg.get("content", "")
        if isinstance(content, list):
            has_tool_use = any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in content
            )
            filtered_blocks = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "thinking"
                    and not has_tool_use
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
            priority = registry.get_compaction_priority(tool_name) if tool_name else "standard"

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
                    priority = registry.get_compaction_priority(tool_name) if tool_name else "standard"

                    if far_away or priority == "ephemeral":
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


def _truncate_tool_call_blocks(blocks: list) -> list:
    """Truncate large tool_call argument fields in normalized assistant content."""
    registry = get_registry()
    out = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_call":
            large_keys = registry.get_large_input_keys(block.get("name", ""))
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


def _execute_tool(
    name: str,
    tool_input: dict,
    functions: dict[str, Callable],
    registry,
) -> ToolResult:
    """Execute a single tool call with validation.  Always returns ToolResult."""
    fn = functions.get(name)
    if fn is None:
        return enrich_error(ToolResult(
            f"Error: Unknown tool '{name}'. Available: {sorted(functions.keys())[:20]}",
            is_error=True,
        ), name)

    tool_schema = registry.get_schema(name)
    validation_error = validate_tool_input(name, tool_input, fn, tool_schema)
    if validation_error:
        return enrich_error(ToolResult(validation_error, is_error=True), name)

    try:
        if tool_input is None:
            tool_input = {}
        raw = fn(**tool_input)
        if isinstance(raw, ToolResult):
            return enrich_error(raw, name)
        return ToolResult(str(raw))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return enrich_error(ToolResult(
            f"Error calling tool {name}: {e}\n{error_detail}",
            is_error=True,
        ), name)


def _execute_tools(
    blocks: list[ToolCallRequest],
    functions: dict[str, Callable],
    registry,
    concurrent_mode: bool,
    max_workers: int,
    timeout: int,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> list[tuple[ToolCallRequest, ToolResult]]:
    """Execute tool calls with selective parallelisation.

    Parallel-safe tools run concurrently; state-mutating tools run sequentially
    as barriers between parallel batches.  Results are always in original order.
    """
    if not blocks:
        return []

    # Fast path: single tool or concurrency disabled
    if len(blocks) <= 1 or not concurrent_mode:
        results = []
        for block in blocks:
            result = _execute_tool_with_timeout(block.name, block.input, functions, registry, timeout)
            results.append((block, result))
        return results

    # Check if any tool is parallel-safe
    if not any(registry.is_parallel_safe(b.name) for b in blocks):
        results = []
        for block in blocks:
            result = _execute_tool_with_timeout(block.name, block.input, functions, registry, timeout)
            results.append((block, result))
        return results

    # Partition into contiguous groups
    results_arr: list[tuple[ToolCallRequest, ToolResult] | None] = [None] * len(blocks)
    i = 0
    while i < len(blocks):
        if registry.is_parallel_safe(blocks[i].name):
            parallel_batch = []
            while i < len(blocks) and registry.is_parallel_safe(blocks[i].name):
                parallel_batch.append((i, blocks[i]))
                i += 1

            pool = executor or _tool_dispatch_executor
            futures = {
                pool.submit(
                    _execute_tool_with_timeout, block.name, block.input, functions, registry, timeout
                ): (idx, block)
                for idx, block in parallel_batch
            }
            for future in concurrent.futures.as_completed(futures):
                idx, block = futures[future]
                try:
                    results_arr[idx] = (block, future.result())
                except Exception as exc:
                    error_detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    results_arr[idx] = (block, ToolResult(
                        f"Error calling tool {block.name}: {exc}\n{error_detail}",
                        is_error=True,
                    ))
        else:
            block = blocks[i]
            result = _execute_tool_with_timeout(block.name, block.input, functions, registry, timeout)
            results_arr[i] = (block, result)
            i += 1

    # Guard: fill any slots that are still None (e.g. executor.submit() itself failed)
    return [
        r if r is not None else (blocks[i], ToolResult("Internal dispatch error", is_error=True))
        for i, r in enumerate(results_arr)
    ]


def _execute_tool_with_timeout(
    name: str,
    tool_input: dict,
    functions: dict[str, Callable],
    registry,
    default_timeout: int,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> ToolResult:
    """Execute a tool body with a timeout.

    Uses a daemon thread per invocation rather than a shared pool.  Rationale
    (P0 fix, 2026-04): Python threads are not cancellable — `future.cancel()`
    on a running thread is a no-op, so a timed-out tool running in a shared
    pool permanently consumes a worker slot.  Under enough timeouts the pool
    starves and every subsequent tool call blocks waiting for a slot that
    never frees.

    Daemon threads sidestep this:
      * A timed-out thread keeps running but holds no pool slot.
      * Daemon status guarantees it cannot block process exit.
      * Thread creation overhead is ~0.1 ms — negligible next to any LLM call.

    The ``executor`` parameter is kept for backwards compatibility (callers
    that passed an explicit pool) but is now ignored.  Dispatch-level
    parallelism still uses ``_tool_dispatch_executor``; only the inner
    timeout wrapper changed.
    """
    timeout = default_timeout
    hint = registry.get_timeout_hint(name)
    if hint is not None:
        timeout = max(timeout, hint)

    # run_shell manages its own timeout — give extra slack
    if name == "run_shell":
        user_timeout = (tool_input or {}).get("timeout", 60)
        timeout = max(timeout, int(user_timeout) + 10)

    result_holder: list[ToolResult | None] = [None]
    exc_holder: list[BaseException | None] = [None]
    done = threading.Event()

    def _run_body() -> None:
        try:
            result_holder[0] = _execute_tool(name, tool_input, functions, registry)
        except BaseException as e:  # noqa: BLE001 — we need to propagate everything
            exc_holder[0] = e
        finally:
            done.set()

    t = threading.Thread(
        target=_run_body,
        name=f"jyagent-tool-body:{name}",
        daemon=True,
    )
    t.start()

    if not done.wait(timeout):
        # Timeout — the daemon thread continues running but holds no pool
        # slot, so there's nothing to leak.  We just report the timeout.
        return ToolResult(
            f"Error: Tool '{name}' timed out after {timeout}s. "
            f"Consider breaking the operation into smaller steps.",
            is_error=True,
        )

    if exc_holder[0] is not None:
        # KeyboardInterrupt in the worker is rare (main thread gets SIGINT)
        # but propagate anyway.  _execute_tool normally catches exceptions
        # and returns an error ToolResult, so reaching this branch implies
        # something pathological.
        if isinstance(exc_holder[0], KeyboardInterrupt):
            raise exc_holder[0]
        return enrich_error(
            ToolResult(
                f"Error: Tool '{name}' raised an uncaught exception: "
                f"{type(exc_holder[0]).__name__}: {exc_holder[0]}",
                is_error=True,
            ),
            name,
        )

    result = result_holder[0]
    if result is None:
        return ToolResult(
            f"Error: Tool '{name}' returned no result (worker finished "
            f"without producing output)",
            is_error=True,
        )
    return result


def _is_transient_error(error: BaseException) -> bool:
    """Return True if the error is likely transient and worth retrying.

    Checks concrete exception types first to avoid false positives from
    keyword-matching against arbitrary error messages.
    """
    # --- Network / transport layer (always transient) ---
    import httpx  # local import to avoid hard dependency at module level
    if isinstance(error, (
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        httpx.RemoteProtocolError,
        ConnectionResetError,
        BrokenPipeError,
        ConnectionAbortedError,
    )):
        return True

    # --- Provider SDK errors (transient if server-side) ---
    try:
        import anthropic as _anth
        if isinstance(error, _anth.APIStatusError) and error.status_code in (429, 500, 502, 503, 529):
            return True
        if isinstance(error, (_anth.APIConnectionError, _anth.APITimeoutError)):
            return True
    except ImportError:
        pass
    try:
        import openai as _oai
        if isinstance(error, _oai.APIStatusError) and error.status_code in (429, 500, 502, 503):
            return True
        if isinstance(error, (_oai.APIConnectionError, _oai.APITimeoutError)):
            return True
    except ImportError:
        pass

    # --- JSON decode failure (often a truncated stream response) ---
    if isinstance(error, json.JSONDecodeError):
        return True

    # --- Fallback: keyword match, but only for generic / unknown types ---
    msg = str(error).lower()
    transient_keywords = [
        "overloaded", "server_error", "peer closed",
        "connection reset", "broken pipe",
    ]
    return any(kw in msg for kw in transient_keywords)


def _build_runtime_options(
    runtime_owner: RuntimeOwner,
    max_output_tokens: int,
    model_spec: ModelSpec | None = None,
    metadata: dict | None = None,
) -> RuntimeOptions:
    """Build RuntimeOptions with reasoning config for the active provider."""
    spec = model_spec or runtime_owner.model_spec
    return RuntimeOptions(
        max_output_tokens=max_output_tokens,
        timeout=STREAM_TIMEOUT,
        reasoning=get_reasoning_config_for_provider(
            spec.provider,
            max_output_tokens=max_output_tokens,
            model=spec.model,
        ),
        metadata=metadata,
    )


# ─── AgentLoop ───────────────────────────────────────────────────────────────

class AgentLoop:
    """Reusable agentic tool-use loop engine.

    Supports both streaming and non-streaming modes, concurrent tool execution,
    context compaction, truncation recovery, and transient-error retry.
    """

    def __init__(
        self,
        runtime_owner: RuntimeOwner,
        config: LoopConfig,
        callbacks: LoopCallbacks | None = None,
        tool_source: ToolSource | None = None,
        model_spec: ModelSpec | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self._runtime_owner = runtime_owner
        self._config = config
        self._callbacks = callbacks or LoopCallbacks()
        self._tool_source = tool_source
        self._model_spec = model_spec  # override for sub-agent model tier
        self._cancel_event = cancel_event
        # Reuse the module-level shared executor to avoid accumulating
        # ThreadPoolExecutor objects and atexit handlers across turns and
        # sub-agent dispatches.
        self._executor = _tool_executor
        # Task-plan scratchpad (see jyagent/todos.py).  Populated via the
        # `write_todos` tool and seeded optionally via run(initial_todos=...)
        # so outer layers can carry the plan across turns.
        self._todos: list = []
        # Run id for checkpointing.  Fresh per AgentLoop; outer layers can
        # override via `set_run_id()` before calling run() to correlate
        # checkpoints with an external request/session.
        self._run_id: str = ""

    def set_run_id(self, run_id: str) -> None:
        """Override the run id used by checkpoint paths.  Must be called
        before ``run()``.  Empty string / None restores default."""
        self._run_id = run_id or ""

    def _is_cancelled(self) -> bool:
        """Check if external cancellation has been requested."""
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _cancellable_sleep(self, seconds: float) -> bool:
        """Sleep that returns early if cancellation is signalled.

        Returns True if cancelled during the wait, False otherwise.  When no
        cancel_event is attached, falls back to a plain blocking sleep.
        """
        if self._cancel_event is None:
            time.sleep(seconds)
            return False
        # Event.wait returns True when set, False on timeout.
        return self._cancel_event.wait(seconds)

    def _write_checkpoint(
        self,
        *,
        step: int | str,
        messages: list,
        total_input_tokens: int,
        total_output_tokens: int,
        tool_calls_count: int,
        status: str,
        error: str | None = None,
    ) -> None:
        """Persist a LoopCheckpoint if checkpointing is enabled.

        ``step`` may be an int (regular step boundary) or ``"final"``
        (terminal exit).  Errors are logged via ``on_warning`` — never
        propagated, checkpointing must never break a run.
        """
        cfg = self._config
        if not cfg.checkpoint_dir:
            return
        from .checkpoint import (
            LoopCheckpoint,
            checkpoint_path,
            iso_utc_now,
        )
        effective_spec = self._model_spec or self._runtime_owner.model_spec
        try:
            cp = LoopCheckpoint(
                run_id=self._run_id,
                step=step if isinstance(step, int) else -1,
                saved_at=iso_utc_now(),
                messages=list(messages),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                tool_calls_count=tool_calls_count,
                todos=[_t_as_dict(t) for t in self._todos] if cfg.todos_enabled else [],
                provider=effective_spec.provider,
                model=effective_spec.model,
                status=status,
                error=error,
            )
            path = checkpoint_path(cfg.checkpoint_dir, self._run_id, step)
            cp.save(path)
            self._fire(
                "on_checkpoint", path,
                step if isinstance(step, int) else -1,
            )
        except Exception as e:
            self._fire("on_warning", f"checkpoint write failed: {e}")

    # ── callback helpers (no-op when callback is None) ────────────────────

    def _fire(self, name: str, *args: Any) -> None:
        cb = getattr(self._callbacks, name, None)
        if cb is not None:
            try:
                cb(*args)
            except Exception:
                # Callbacks are for presentation — never abort the engine loop.
                print(f"[warning] callback {name!r} raised:", traceback.format_exc(), file=sys.stderr)

    # ── public entry point ────────────────────────────────────────────────

    def run(
        self,
        system_prompt: str,
        messages: list,
        initial_todos: list | None = None,
    ) -> LoopResult:
        """Run the agentic tool-use loop.  *messages* is mutated in-place.

        Thin wrapper around ``_run_impl`` that attaches the final todos
        scratchpad and writes a terminal checkpoint (if enabled),
        regardless of which exit path fired.
        """
        result = self._run_impl(system_prompt, messages, initial_todos)
        if self._config.todos_enabled:
            # Serialize to dict-form for easy JSON persistence by outer layers.
            from .todos import todo_to_dict
            result.todos = [todo_to_dict(t) for t in self._todos]
        if self._config.checkpoint_dir:
            # Terminal ("final") checkpoint — includes status + error.
            self._write_checkpoint(
                step="final",
                messages=result.messages,
                total_input_tokens=result.total_input_tokens,
                total_output_tokens=result.total_output_tokens,
                tool_calls_count=result.tool_calls_count,
                status=result.status,
                error=result.error,
            )
        return result

    def _run_impl(
        self,
        system_prompt: str,
        messages: list,
        initial_todos: list | None = None,
    ) -> LoopResult:
        """Core run loop.  Public entry point is ``run()`` which also
        snapshots the final todos onto the result."""
        cfg = self._config
        all_text = ""
        final_text = ""
        current_max_tokens = cfg.initial_max_tokens
        total_input_tokens = 0
        total_output_tokens = 0
        tool_calls_count = 0
        last_reflection_count = 0  # tool_calls_count at last reflection injection
        registry = get_registry()
        step = 0
        consecutive_truncations = 0  # cap truncation recovery retries
        max_truncation_retries = 3
        verification_injected = False  # only verify once per run

        # Lazy import of the reflection module so test imports of
        # loop_engine stay cheap and reflection is opt-in by config.
        if cfg.reflect_every_n_tool_calls > 0 or cfg.reflect_after_subagent:
            from . import reflection  # noqa: F401 — referenced below
        else:
            reflection = None  # type: ignore[assignment]

        # Ensure a run id is set when checkpointing is enabled (outer
        # layers may have preset one via set_run_id).
        if cfg.checkpoint_dir and not self._run_id:
            from .checkpoint import new_run_id
            self._run_id = new_run_id()

        # ── Seed todos scratchpad ─────────────────────────────────────
        # Lazy import to keep the dependency optional.
        if cfg.todos_enabled:
            from .todos import (
                WRITE_TODOS_SCHEMA,
                build_write_todos_tool,
                inject_todos_into_messages,
                normalize_todo,
            )
            if initial_todos:
                try:
                    self._todos = [normalize_todo(t) for t in initial_todos]
                except TypeError as e:
                    self._fire("on_warning", f"ignoring invalid initial_todos: {e}")
                    self._todos = []
            else:
                self._todos = []

            # Per-loop write_todos tool closing over self._todos.
            def _get_store() -> list:
                return self._todos

            def _set_store(new_list: list) -> None:
                self._todos = new_list

            _write_todos_fn = build_write_todos_tool(_get_store, _set_store)

        # ── Harness trackers ──────────────────────────────────────────
        # Effective model spec — sub-agent override wins over owner default.
        # Used for tracing and cost accounting so sub-agents on a different
        # tier are billed against the correct pricing.
        effective_spec = self._model_spec or self._runtime_owner.model_spec

        trace = get_tracer()
        if trace:
            trace.start(effective_spec.provider, effective_spec.model)
        cost_tracker = _CostTracker() if cfg.max_cost_usd is not None else None
        stuck_detector = _StuckLoopDetector(cfg.dedup_threshold)

        # Resolve tools
        tool_schemas: list[dict] = []
        tool_functions: dict[str, Callable] = {}

        reg_version: int | None = None

        try:
            for step in range(cfg.max_steps):
                self._fire("on_step_progress", step, cfg.max_steps)

                # ── Cooperative cancellation check (top of loop) ─────
                if self._is_cancelled():
                    break

                # Refresh tool source each step
                if self._tool_source is not None:
                    tool_schemas, tool_functions = self._tool_source()
                elif step == 0 or (reg_version is not None and registry.version != reg_version):
                    reg_version, tool_schemas, tool_functions = registry.snapshot()

                # Overlay the per-loop write_todos tool on top of the
                # registry snapshot when todos are enabled.  This is the
                # closure-scoped injection point recommended by the design
                # review (avoids ContextVar propagation issues with our
                # daemon-thread tool executor).
                if cfg.todos_enabled:
                    tool_schemas = list(tool_schemas) + [WRITE_TODOS_SCHEMA]
                    tool_functions = dict(tool_functions)
                    tool_functions["write_todos"] = _write_todos_fn

                # Context compaction
                if cfg.compact_messages:
                    before_len = len(messages)
                    messages_maybe = _compact_messages(
                        messages, cfg.max_working_tokens, cfg.compact_tool_result_chars,
                    )
                    if messages_maybe is not messages:
                        after_len = len(messages_maybe)
                        messages[:] = messages_maybe
                        self._fire("on_compaction", before_len, after_len)

                # Build context dict.  Todos are injected as a
                # <system-reminder> text block appended to the tail user
                # message — NOT persisted into `messages`, so compaction
                # never touches them.  The base system_prompt stays
                # untouched to preserve Anthropic prefix caching.
                if cfg.todos_enabled and self._todos:
                    context_messages = inject_todos_into_messages(messages, self._todos)
                else:
                    context_messages = messages

                context: dict[str, Any] = {
                    "system_prompt": system_prompt,
                    "messages": context_messages,
                }
                if tool_schemas:
                    context["tools"] = tool_schemas

                # LLM call with retry
                opts = _build_runtime_options(
                    self._runtime_owner,
                    current_max_tokens,
                    model_spec=self._model_spec,
                    metadata={"component": "loop_engine", "step": step + 1},
                )

                # Phase-aware tool_choice shaping (see jyagent/phases.py).
                # The policy is consulted once per step.  Returning a
                # PhaseDirective with `tool_choice=None` is informational
                # only (engine fires on_phase_enter for observability but
                # leaves tool_choice unchanged).  A non-None tool_choice
                # rebuilds `opts` so the runtime adapter sees the override.
                if cfg.phase_policy is not None:
                    try:
                        directive = cfg.phase_policy(step, cfg.max_steps, tool_calls_count)
                    except Exception as e:
                        directive = None
                        self._fire("on_warning", f"phase_policy raised: {e}")
                    if directive is not None:
                        self._fire("on_phase_enter", directive.phase)
                        if trace:
                            trace.add_span(
                                step=step, event_type="phase",
                                tool_name=directive.phase,
                            )
                        if directive.tool_choice is not None:
                            opts = RuntimeOptions(
                                max_output_tokens=opts.max_output_tokens,
                                timeout=opts.timeout,
                                reasoning=opts.reasoning,
                                metadata={**(opts.metadata or {}), "phase": directive.phase},
                                tool_choice=directive.tool_choice,
                            )

                llm_t0 = time.perf_counter()
                step_text, tool_call_blocks, stop_reason, final_message = self._call_llm_with_retry(
                    context, opts, step,
                )
                llm_dur_ms = (time.perf_counter() - llm_t0) * 1000

                # Fire runtime warnings
                for warning in final_message.get("runtime_warnings", []):
                    self._fire("on_warning", warning)

                # Accumulate usage
                usage = final_message.get("usage", {})
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)
                self._fire("on_usage", usage)

                # ── Trace LLM call ────────────────────────────────────
                if trace:
                    trace.add_span(
                        step=step,
                        event_type="llm_call",
                        duration_ms=llm_dur_ms,
                        tokens_in=usage.get("input_tokens"),
                        tokens_out=usage.get("output_tokens"),
                    )

                # ── Cost budget check ─────────────────────────────────
                if cost_tracker is not None:
                    # Use the effective spec so sub-agent model overrides are
                    # billed at the right rate (P0 fix).
                    cost_tracker.record(
                        usage,
                        effective_spec.provider,
                        effective_spec.model,
                    )
                    known = cost_tracker.known_cost
                    if known is not None and known >= cfg.max_cost_usd:
                        self._fire(
                            "on_warning",
                            f"Cost budget exceeded: ${known:.4f} >= ${cfg.max_cost_usd:.4f}",
                        )
                        if trace:
                            trace.add_span(step=step, event_type="cost_check", success=False,
                                           error=f"budget ${cfg.max_cost_usd} exceeded")
                            trace.finish(status="cost_limit", total_steps=step + 1, total_cost_usd=known)
                            trace.flush()
                        return LoopResult(
                            status="cost_limit",
                            text=all_text or "",
                            final_text=final_text,
                            messages=messages,
                            steps=step + 1,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                            tool_calls_count=tool_calls_count,
                            error=f"Cost budget exceeded: ${known:.4f} >= ${cfg.max_cost_usd:.4f}",
                        )

                all_text += step_text
                final_text = step_text

                # No tool calls → done (or verification gate)
                if not tool_call_blocks:
                    # ── Pre-completion verification gate ───────────────
                    # If we mutated files and haven't verified yet, inject a
                    # self-check prompt and loop once more instead of returning.
                    #
                    # Boundary guard (P0 fix): never inject on the final allowed
                    # step — the follow-up model reply has no iteration left to
                    # run, and the dangling `[VERIFICATION]` user message would
                    # otherwise leak into the persisted session and poison the
                    # next turn.
                    if (
                        not verification_injected
                        and should_verify(messages, tool_calls_count)
                        and step + 1 < cfg.max_steps
                    ):
                        verification_injected = True
                        if trace:
                            trace.add_span(step=step, event_type="verification")
                        # Append the assistant's response, then inject verification
                        messages.append(final_message)
                        messages.append({
                            "role": "user",
                            "content": build_verification_prompt(messages),
                        })
                        continue

                    if not step_text:
                        final_text = _extract_text(final_message)
                        all_text = final_text or all_text

                    # Apply truncation if enabled
                    if cfg.truncate_large_inputs:
                        content = final_message.get("content", [])
                        final_message = dict(final_message)
                        final_message["content"] = _truncate_tool_call_blocks(content)

                    # Allow caller to transform before append
                    cb_am = self._callbacks.on_assistant_message
                    if cb_am is not None:
                        final_message = cb_am(final_message) or final_message
                    messages.append(final_message)
                    result_text = all_text if all_text else "I processed your request but had no text response to return."

                    # ── Flush trace ────────────────────────────────────
                    if trace:
                        cost = cost_tracker.known_cost if cost_tracker else 0.0
                        trace.finish(status="completed", total_steps=step + 1, total_cost_usd=cost or 0.0)
                        trace.flush()

                    return LoopResult(
                        status="completed",
                        text=result_text,
                        final_text=final_text,
                        messages=messages,
                        steps=step + 1,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        tool_calls_count=tool_calls_count,
                    )

                # Truncation detection → scale up and retry step
                if cfg.auto_scale_on_truncation and _is_truncated(stop_reason, tool_call_blocks):
                    consecutive_truncations += 1
                    if consecutive_truncations > max_truncation_retries:
                        return LoopResult(
                            status="error",
                            text=all_text or "",
                            final_text="",
                            messages=messages,
                            steps=step + 1,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                            tool_calls_count=tool_calls_count,
                            error=f"Repeated truncation ({consecutive_truncations}x) — model output exceeds capacity",
                        )
                    self._fire("on_truncation")
                    # Also fire the unified stream-retry hook so UIs that
                    # already handle transient-error duplication can use the
                    # same visual treatment for truncation-recovery replays.
                    self._fire("on_stream_retry", "truncation", step_text or "")
                    current_max_tokens = min(
                        current_max_tokens * cfg.token_scale_factor,
                        cfg.max_tokens_cap,
                    )
                    # Remove the partial step text
                    all_text = all_text[: -len(step_text)] if step_text else all_text
                    continue

                # Successful step — reset truncation counter
                consecutive_truncations = 0

                # Append assistant message (allow caller to transform)
                if cfg.truncate_large_inputs:
                    content = final_message.get("content", [])
                    final_message = dict(final_message)
                    final_message["content"] = _truncate_tool_call_blocks(content)

                cb_am = self._callbacks.on_assistant_message
                if cb_am is not None:
                    transformed = cb_am(final_message)
                    if transformed is not None:
                        final_message = transformed
                messages.append(final_message)

                # Fire on_tool_batch for multi-tool batches
                if len(tool_call_blocks) > 1:
                    self._fire("on_tool_batch", len(tool_call_blocks))

                # Fire on_tool_start for all tool calls BEFORE execution
                for block in tool_call_blocks:
                    self._fire("on_tool_start", block.name, block.input)

                # Execute tools
                # ── Cooperative cancellation check (before tools) ────
                if self._is_cancelled():
                    # Return error results for all pending tool calls
                    for block in tool_call_blocks:
                        messages.append({
                            "role": "tool_result",
                            "tool_call_id": block.id,
                            "tool_name": block.name,
                            "content": "Cancelled",
                            "is_error": True,
                        })
                    break

                tools_t0 = time.perf_counter()
                tool_results_tuples = _execute_tools(
                    tool_call_blocks,
                    tool_functions,
                    registry,
                    cfg.concurrent_tools,
                    cfg.max_tool_workers,
                    cfg.tool_timeout,
                    executor=self._executor,
                )
                tools_dur_ms = (time.perf_counter() - tools_t0) * 1000

                for block, result in tool_results_tuples:
                    tool_calls_count += 1
                    content_str = _truncate_result(result.content, cfg.max_tool_result_chars, result.is_error)
                    self._fire("on_tool_end", block.name, content_str, result.is_error)

                    # ── Trace tool call ────────────────────────────────
                    if trace:
                        trace.add_span(
                            step=step,
                            event_type="tool_call",
                            tool_name=block.name,
                            tool_args=block.input,
                            success=not result.is_error,
                            error=content_str[:200] if result.is_error else None,
                        )

                    messages.append({
                        "role": "tool_result",
                        "tool_call_id": block.id,
                        "tool_name": block.name,
                        "content": content_str,
                        "is_error": result.is_error,
                    })

                # ── Response-aware stuck-loop detection ────────────────
                # Check AFTER execution so we can compare responses.  A tool
                # is only "stuck" when the same (tool, args) returns the same
                # response repeatedly — polling tools like check_background
                # naturally return different responses (elapsed_seconds etc.)
                # and are never flagged.
                #
                # Two correctness rules (P0 fixes, 2026-04):
                #   1. Hash the RAW tool output, not the UI-truncated string.
                #      Two different long outputs that happen to share a
                #      common prefix up to max_tool_result_chars would
                #      collide on the truncated string and look "stuck".
                #   2. Deduplicate (name, args) keys *within a single batch*.
                #      A legitimate parallel fanout of e.g. 3 identical
                #      read_file calls in one step is not a stuck loop — it's
                #      the model doing simultaneous reads.  Without this, such
                #      a batch alone can hit threshold=3 in a single step.
                stuck_feedback = None
                seen_batch_keys: set[str] = set()
                for block, result in tool_results_tuples:
                    batch_key = _StuckLoopDetector._make_key(
                        block.name, block.input if isinstance(block.input, dict) else {},
                    )
                    if batch_key in seen_batch_keys:
                        continue
                    seen_batch_keys.add(batch_key)
                    feedback = stuck_detector.record(
                        block.name,
                        block.input,
                        result.content,  # raw content — not the truncated display string
                    )
                    if feedback and not stuck_feedback:
                        stuck_feedback = feedback
                if stuck_feedback:
                    self._fire("on_warning", stuck_feedback)
                    if trace:
                        trace.add_span(step=step, event_type="dedup_break", success=False, error=stuck_feedback)
                        cost = cost_tracker.known_cost if cost_tracker else 0.0
                        trace.finish(status="dedup_break", total_steps=step + 1, total_cost_usd=cost or 0.0)
                        trace.flush()
                    return LoopResult(
                        status="dedup_break",
                        text=all_text or "",
                        final_text=final_text,
                        messages=messages,
                        steps=step + 1,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        tool_calls_count=tool_calls_count,
                        error=stuck_feedback,
                    )

                # ── Cooperative cancellation check (after tools) ─────
                if self._is_cancelled():
                    break

                # ── Mid-loop reflection / critic step ─────────────────
                # After meaningful work boundaries (every-N cadence or
                # sub-agent return), append a short progress-check user
                # message so the next LLM call re-grounds on the task.
                # Avoids drift on long-horizon rollouts.
                if cfg.reflect_every_n_tool_calls > 0 or cfg.reflect_after_subagent:
                    batch_names = [b.name for b, _ in tool_results_tuples]
                    inject, reason = reflection.should_reflect(
                        reflect_every_n=cfg.reflect_every_n_tool_calls,
                        reflect_after_subagent=cfg.reflect_after_subagent,
                        tool_calls_total=tool_calls_count,
                        tool_calls_at_last_reflection=last_reflection_count,
                        batch_tool_names=batch_names,
                        messages=messages,
                    )
                    if inject:
                        prompt = reflection.build_reflection_prompt(
                            reason, tool_calls_count,
                        )
                        messages.append({"role": "user", "content": prompt})
                        last_reflection_count = tool_calls_count
                        self._fire("on_reflection", reason)
                        if trace:
                            trace.add_span(step=step, event_type="reflection")

                # ── Periodic checkpoint ──────────────────────────────
                # At the end of each step, if a cadence is configured and
                # we're on the boundary, persist state so crashes can
                # resume from here.  No-op when checkpoint_dir is None.
                if (
                    cfg.checkpoint_every_n_steps > 0
                    and cfg.checkpoint_dir
                    and (step + 1) % cfg.checkpoint_every_n_steps == 0
                ):
                    self._write_checkpoint(
                        step=step,
                        messages=messages,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        tool_calls_count=tool_calls_count,
                        status="in_progress",
                    )

            # ── Cooperative cancellation — early exit ────────────────
            if self._is_cancelled():
                if trace:
                    trace.finish(status="interrupted", total_steps=step + 1)
                    trace.flush()
                return LoopResult(
                    status="interrupted",
                    text=all_text or "",
                    final_text=final_text,
                    messages=messages,
                    steps=step + 1,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    tool_calls_count=tool_calls_count,
                )

            # Max steps reached
            # Fallback always fires when enabled: reaching max_steps means the
            # loop never hit a no-tool terminal step, so the incidental text
            # accumulated from prior tool-use steps is NOT a real answer.
            # (Old condition `not final_text` was wrong — `final_text` is
            # written on every step including ones that also had tool calls.)
            #
            # Defense-in-depth (P0 fix): if a verification prompt was injected
            # and the model never produced a follow-up reply before we ran out
            # of steps, strip the trailing `[VERIFICATION]` user message so
            # it doesn't leak into the persisted session and poison the next
            # turn.  The boundary guard above should already prevent this,
            # but we clean up anyway in case of a race.
            if verification_injected and messages:
                tail = messages[-1]
                tail_content = tail.get("content", "") if isinstance(tail, dict) else ""
                if tail.get("role") == "user" and isinstance(tail_content, str) \
                        and tail_content.startswith("[VERIFICATION]"):
                    messages.pop()

            if cfg.fallback_on_max_steps:
                # Try one more streaming call with system instruction to avoid tools
                try:
                    fallback_context = dict(context)
                    fallback_system = context["system_prompt"] + "\n\n[SYSTEM: You have reached the maximum number of tool-use steps. Please provide your best answer now WITHOUT using any tools.]"
                    fallback_context["system_prompt"] = fallback_system

                    # Create fallback options with tool_choice=none
                    _base = _build_runtime_options(
                        self._runtime_owner,
                        cfg.initial_max_tokens,
                        model_spec=self._model_spec,
                        metadata={"component": "loop_engine", "step": cfg.max_steps + 1, "fallback": True},
                    )
                    fallback_opts = RuntimeOptions(
                        max_output_tokens=_base.max_output_tokens,
                        timeout=_base.timeout,
                        reasoning=_base.reasoning,
                        metadata=_base.metadata,
                        tool_choice={"type": "none"},
                    )

                    # Remove tools from fallback context to ensure no tool use
                    if "tools" in fallback_context:
                        del fallback_context["tools"]

                    if cfg.streaming:
                        fallback_text, _, _, fallback_message = self._call_streaming(fallback_context, fallback_opts)
                    else:
                        fallback_text, _, _, fallback_message = self._call_complete(fallback_context, fallback_opts)

                    # Accumulate usage
                    usage = fallback_message.get("usage", {})
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
                    self._fire("on_usage", usage)

                    # Apply truncation if enabled
                    if cfg.truncate_large_inputs:
                        content = fallback_message.get("content", [])
                        fallback_message = dict(fallback_message)
                        fallback_message["content"] = _truncate_tool_call_blocks(content)

                    # Append fallback response
                    messages.append(fallback_message)

                    # Return completed since we got a final answer
                    return LoopResult(
                        status="completed",
                        text=fallback_text or all_text,
                        final_text=fallback_text,
                        messages=messages,
                        steps=cfg.max_steps,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        tool_calls_count=tool_calls_count,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception:
                    # If fallback fails, fall through to normal max_steps handling
                    pass

            # ── Trace max_steps ────────────────────────────────────────
            if trace:
                cost = cost_tracker.known_cost if cost_tracker else 0.0
                trace.finish(status="max_steps", total_steps=cfg.max_steps, total_cost_usd=cost or 0.0)
                trace.flush()

            return LoopResult(
                status="max_steps",
                text=all_text or "",
                final_text=final_text,
                messages=messages,
                steps=cfg.max_steps,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                tool_calls_count=tool_calls_count,
            )

        except KeyboardInterrupt:
            if trace:
                trace.finish(status="interrupted", total_steps=step + 1)
                trace.flush()
            return LoopResult(
                status="interrupted",
                text=all_text + "\n\n[Interrupted by user]" if all_text else "[Interrupted by user]",
                final_text="",
                messages=messages,
                steps=step + 1,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                tool_calls_count=tool_calls_count,
            )
        except Exception as e:
            if trace:
                trace.finish(status="error", total_steps=step + 1)
                trace.flush()
            return LoopResult(
                status="error",
                text=all_text or "",
                final_text="",
                messages=messages,
                steps=step + 1,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                tool_calls_count=tool_calls_count,
                error=str(e),
            )

    # ── LLM call with retry ──────────────────────────────────────────────

    def _call_llm_with_retry(
        self,
        context: dict,
        options: RuntimeOptions,
        step: int,
    ) -> tuple[str, list[ToolCallRequest], str, dict]:
        """Call the LLM (streaming or complete) with transient-error retry.

        Returns (step_text, tool_call_blocks, stop_reason, final_message).
        """
        cfg = self._config
        last_error: BaseException | None = None

        for attempt in range(cfg.retry_attempts + 1):
            try:
                if cfg.streaming:
                    return self._call_streaming(context, options)
                else:
                    return self._call_complete(context, options)
            except KeyboardInterrupt:
                raise
            except Exception as err:
                last_error = err
                if _is_transient_error(err) and attempt < cfg.retry_attempts:
                    if self._is_cancelled():
                        raise
                    # Exponential backoff with "equal jitter" (AWS architecture
                    # recommendation) to avoid thundering-herd when multiple
                    # parallel sub-agents all retry a 529 at the same moment.
                    #   half the delay is deterministic exponential,
                    #   half is uniform random in [0, base * 2^attempt / 2].
                    base = cfg.retry_base_delay * (2 ** attempt)
                    delay = base / 2 + random.uniform(0, base / 2)
                    self._fire("on_retry", attempt + 1, err)
                    # Signal UI that any partial output from the failed
                    # attempt will be replayed on retry (visual de-duplication
                    # hook).  `partial_stream_text` is stashed by
                    # `_call_streaming` on the exception; missing for
                    # non-streaming path.
                    partial_text = getattr(err, "partial_stream_text", "")
                    self._fire("on_stream_retry", "transient_error", partial_text)
                    # Cancel-aware backoff: wake immediately on Ctrl-C so we
                    # don't burn through a long retry window after cancel.
                    if self._cancellable_sleep(delay):
                        raise KeyboardInterrupt("cancelled during retry backoff")
                    continue
                raise

        # Should not reach here, but just in case:
        raise last_error  # type: ignore[misc]

    # ── non-streaming call ───────────────────────────────────────────────

    def _call_complete(
        self,
        context: dict,
        options: RuntimeOptions,
    ) -> tuple[str, list[ToolCallRequest], str, dict]:
        """Non-streaming: runtime_owner.complete() -> extract text/tool_calls."""
        final_message = self._runtime_owner.complete(
            context, options=options, model_spec=self._model_spec,
        )
        stop_reason = final_message.get("stop_reason", "stop")

        if stop_reason == "error":
            error_msg = final_message.get("error_message", "Unknown error")
            raise RuntimeError(error_msg)

        step_text = _extract_text(final_message)
        if step_text:
            self._fire("on_text_delta", step_text)

        tool_calls = _extract_tool_calls(final_message)
        return step_text, tool_calls, stop_reason, final_message

    # ── streaming call ───────────────────────────────────────────────────

    def _call_streaming(
        self,
        context: dict,
        options: RuntimeOptions,
    ) -> tuple[str, list[ToolCallRequest], str, dict]:
        """Streaming: consume RuntimeStream events and fire callbacks.

        Delta-emission policy is controlled by ``cfg.buffered_streaming``:

        * ``False`` (default) — fire ``on_text_delta`` live as tokens arrive.
          On transient error mid-stream, the user sees partial output and the
          retry replays it, producing visible duplication.  The engine fires
          ``on_stream_retry`` before the retry so UIs can mark/clear the
          duplicated region.
        * ``True`` — buffer deltas locally and only flush them to
          ``on_text_delta`` after a clean ``done`` event.  A failed attempt
          discards its buffer silently.  Eliminates duplication at the cost
          of losing live-token UX.

        The buffered partial text is stashed on the raised exception as
        ``err.partial_stream_text`` so ``_call_llm_with_retry`` can pass it
        to ``on_stream_retry``.
        """
        cfg = self._config
        text_parts: list[str] = []
        # Tracks how many characters have already been flushed to
        # on_text_delta — used for buffered mode and for partial-text
        # reporting on error.
        emitted_len = 0
        final_message: dict | None = None
        thinking_active = False
        stream = None

        def _flush_pending() -> None:
            """Emit un-flushed buffered deltas (buffered mode only)."""
            nonlocal emitted_len
            if emitted_len >= sum(len(p) for p in text_parts):
                return
            pending = "".join(text_parts)[emitted_len:]
            if pending:
                self._fire("on_text_delta", pending)
                emitted_len += len(pending)

        try:
            stream = self._runtime_owner.stream(
                context, options=options, model_spec=self._model_spec,
            )
            for event in stream:
                # Cancellation check inside the stream loop so Ctrl-C
                # doesn't wait for the provider to close — latency-sensitive.
                if self._is_cancelled():
                    raise KeyboardInterrupt("cancelled during stream")
                etype = event.get("type")

                if etype == "text_delta":
                    text = event.get("text", "")
                    if thinking_active:
                        thinking_active = False
                        self._fire("on_thinking_stop")
                    text_parts.append(text)
                    if not cfg.buffered_streaming:
                        # Live mode: emit now.
                        self._fire("on_text_delta", text)
                        emitted_len += len(text)
                    # Buffered mode: accumulate, flush on `done`.

                elif etype == "thinking_start":
                    if not thinking_active:
                        thinking_active = True
                        self._fire("on_thinking_start")

                elif etype == "thinking_delta":
                    if not thinking_active:
                        thinking_active = True
                        self._fire("on_thinking_start")

                elif etype in ("tool_call_start", "tool_call_delta"):
                    if thinking_active:
                        thinking_active = False
                        self._fire("on_thinking_stop")

                elif etype == "thinking_end":
                    if thinking_active:
                        thinking_active = False
                        self._fire("on_thinking_stop")

                elif etype == "done":
                    final_message = event["message"]
                    # Buffered mode: flush the accumulated text now that we
                    # know the stream completed cleanly.
                    if cfg.buffered_streaming:
                        _flush_pending()

                elif etype == "error":
                    final_message = event["message"]

            if final_message is None:
                final_message = stream.get_final_message()

            stop_reason = final_message.get("stop_reason", "stop")
            if stop_reason == "error":
                error_msg = final_message.get("error_message", "Unknown streaming error")
                # Attach partial text so the retry layer can report it to
                # on_stream_retry.  Accumulated even in buffered mode — the
                # caller decides what (if anything) to do with it.
                err = RuntimeError(error_msg)
                err.partial_stream_text = "".join(text_parts)  # type: ignore[attr-defined]
                raise err

            # Successful completion: in live mode emitted_len already equals
            # the full text length; in buffered mode the done-handler above
            # flushed it.  Nothing more to do.
            tool_calls = _extract_tool_calls(final_message)
            return "".join(text_parts), tool_calls, stop_reason, final_message

        except BaseException as err:
            # Stash partial text on every exception path (transient network
            # errors, cancellations, etc.) so the retry layer can pass it to
            # on_stream_retry.  Intentionally set on the raised exception
            # rather than returned — the call site re-raises and catches it
            # at a different layer.
            if not hasattr(err, "partial_stream_text"):
                err.partial_stream_text = "".join(text_parts)  # type: ignore[attr-defined]
            raise

        finally:
            if thinking_active:
                self._fire("on_thinking_stop")
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
