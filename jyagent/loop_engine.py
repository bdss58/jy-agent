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


@dataclass
class ToolCallRequest:
    id: str
    name: str
    input: dict


# Type alias: returns (schemas_list, functions_dict)
ToolSource = Callable[[], tuple[list[dict], dict[str, Callable]]]


# ─── Shared executor ─────────────────────────────────────────────────────────
# Single process-wide executor shared by all AgentLoop instances (interactive
# turns *and* sub-agents).  Workers are created on demand and idle threads are
# cheap, so a slightly generous pool avoids contention during parallel tool
# batches without meaningful resource cost.

_tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
atexit.register(_tool_executor.shutdown, wait=False)


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
        content = msg.get("content", "")
        if isinstance(content, list):
            filtered_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    did_compact = True
                    continue  # drop thinking block entirely
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

            pool = executor or _tool_executor
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
    """Execute a tool via an executor with a timeout."""
    timeout = default_timeout
    hint = registry.get_timeout_hint(name)
    if hint is not None:
        timeout = max(timeout, hint)

    # run_shell manages its own timeout — give extra slack
    if name == "run_shell":
        user_timeout = (tool_input or {}).get("timeout", 60)
        timeout = max(timeout, int(user_timeout) + 10)

    pool = executor or _tool_executor
    future = pool.submit(_execute_tool, name, tool_input, functions, registry)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        future.cancel()
        return ToolResult(
            f"Error: Tool '{name}' timed out after {timeout}s. "
            f"Consider breaking the operation into smaller steps.",
            is_error=True,
        )
    except KeyboardInterrupt:
        future.cancel()
        raise


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

    def _is_cancelled(self) -> bool:
        """Check if external cancellation has been requested."""
        return self._cancel_event is not None and self._cancel_event.is_set()

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

    def run(self, system_prompt: str, messages: list) -> LoopResult:
        """Run the agentic tool-use loop.  *messages* is mutated in-place."""
        cfg = self._config
        all_text = ""
        final_text = ""
        current_max_tokens = cfg.initial_max_tokens
        total_input_tokens = 0
        total_output_tokens = 0
        tool_calls_count = 0
        registry = get_registry()
        step = 0
        consecutive_truncations = 0  # cap truncation recovery retries
        max_truncation_retries = 3
        verification_injected = False  # only verify once per run

        # ── Harness trackers ──────────────────────────────────────────
        trace = get_tracer()
        if trace:
            spec = self._model_spec or self._runtime_owner.model_spec
            trace.start(spec.provider, spec.model)
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

                # Build context dict
                context: dict[str, Any] = {
                    "system_prompt": system_prompt,
                    "messages": messages,
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
                    cost_tracker.record(
                        usage,
                        self._runtime_owner.model_spec.provider,
                        self._runtime_owner.model_spec.model,
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
                    if not verification_injected and should_verify(messages, tool_calls_count):
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
                stuck_feedback = None
                for block, result in tool_results_tuples:
                    content_str = _truncate_result(result.content, cfg.max_tool_result_chars, result.is_error)
                    feedback = stuck_detector.record(
                        block.name,
                        block.input,
                        content_str,
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
            if cfg.fallback_on_max_steps and not final_text:
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
                    delay = cfg.retry_base_delay * (2 ** attempt)
                    self._fire("on_retry", attempt + 1, err)
                    time.sleep(delay)
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
        """Streaming: consume RuntimeStream events and fire callbacks."""
        text_parts: list[str] = []
        final_message: dict | None = None
        thinking_active = False
        stream = None

        try:
            stream = self._runtime_owner.stream(
                context, options=options, model_spec=self._model_spec,
            )
            for event in stream:
                etype = event.get("type")

                if etype == "text_delta":
                    text = event.get("text", "")
                    if thinking_active:
                        thinking_active = False
                        self._fire("on_thinking_stop")
                    self._fire("on_text_delta", text)
                    text_parts.append(text)

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

                elif etype == "error":
                    final_message = event["message"]

            if final_message is None:
                final_message = stream.get_final_message()

            stop_reason = final_message.get("stop_reason", "stop")
            if stop_reason == "error":
                error_msg = final_message.get("error_message", "Unknown streaming error")
                raise RuntimeError(error_msg)

            tool_calls = _extract_tool_calls(final_message)
            return "".join(text_parts), tool_calls, stop_reason, final_message

        finally:
            if thinking_active:
                self._fire("on_thinking_stop")
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
