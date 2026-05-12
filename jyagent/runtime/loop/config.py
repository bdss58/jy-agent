"""LoopConfig and LoopResult dataclasses for the agent loop runtime.

Extracted from engine.py during the runtime-package refactor (phase 3).
Kept as plain dataclasses (no engine deps) so callers can build configs
without paying the engine import cost.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Normalized message TypedDict — used only as a type annotation on
    # ``LoopResult.messages``.  Kept under TYPE_CHECKING to avoid a
    # runtime import cycle (llm.types -> runtime.loop.llm_types).
    from ...llm.types import Message

from dataclasses import dataclass, field
from typing import Any


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
    retry_attempts: int = 10
    retry_base_delay: float = 1.0
    # When True (default), the LLM retry loop retries ANY exception up to
    # ``retry_attempts``, not just errors classified transient by
    # ``is_transient_error``.  Rationale: in practice non-transient
    # classification is heuristic (provider SDKs evolve, proxies wrap
    # errors, etc.), and the retry budget is small enough that burning it
    # on a deterministic 4xx is preferable to bailing out on a real
    # transient that wasn't recognized.  Set to False to restore the
    # strict transient-only retry policy.  Reason label passed to
    # ``on_stream_retry`` still distinguishes "transient_error" vs "error"
    # so UI/telemetry can tell them apart.
    retry_on_all_errors: bool = True
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
    # Persistent task-plan scratchpad (see jyagent/runtime/loop/todos.py).
    # When True:
    #   * a per-loop `write_todos` tool is overlaid onto the tool source
    #     so the model can create / update the plan;
    #   * the current plan is rendered as a <system-reminder> block
    #     appended to the tail user message before each LLM call — NOT
    #     persisted into the messages list, so it survives compaction
    #     automatically.
    todos_enabled: bool = False
    # Mid-loop reflection / critic step (see runtime/loop/reflection.py).
    # Injects a short progress-check prompt after every N tool calls and/or
    # after any batch that dispatched a sub-agent.  Both triggers OFF by default.
    reflect_every_n_tool_calls: int = 0   # 0 disables the cadence trigger
    reflect_after_subagent: bool = False
    # Phase-aware tool_choice shaping (see runtime/loop/phases.py).  When set,
    # the policy is consulted each step and may override tool_choice for
    # that LLM call (plan / act / verify / finalize).  Does NOT mutate the
    # message history — keeps Anthropic prefix caching fully intact.
    phase_policy: Any = None  # PhasePolicy | None — typed as Any to avoid cycles
    # Checkpointed replay (see runtime/loop/checkpoint.py).  When both are
    # set, LoopCheckpoint is written every N steps (and on terminal exits)
    # to ``<checkpoint_dir>/<run_id>/step_NNNN.json``.  Off by default.
    checkpoint_dir: str | None = None
    checkpoint_every_n_steps: int = 0
    # Harness controls
    max_cost_usd: float | None = None       # cost budget per turn — None = unlimited
    dedup_threshold: int = 3                 # same tool+args+response N times → break loop
    # Runtime validation of provider output (Codex review #3 / 2026-05).
    # When True, every assistant message returned by ``LLMClient.complete``
    # and every terminal stream event from ``LLMClient.stream`` is checked
    # against the canonical TypedDict shapes in jyagent.llm.types via
    # jyagent.llm.validation.  Adapter drift (renamed key, missing field,
    # wrong-type usage counter) raises ``MessageValidationError`` with a
    # precise pointer-style path *at the boundary* — instead of silently
    # corrupting loop state and surfacing as a token-count regression three
    # steps later.
    #
    # Off by default: the validator is microsecond-cheap but every
    # production message runs through provider SDKs that we can't fully
    # type-check, and this is a defense-in-depth check primarily useful
    # in dev / CI.  Tests force it on via the ``JYAGENT_VALIDATE_PROVIDER_OUTPUT``
    # env var (``conftest.py`` sets it for the whole pytest session) so
    # adapter changes that violate the contract fail loudly during test
    # runs without having to thread the flag through every fixture.
    validate_provider_output: bool = False


@dataclass
class LoopResult:
    status: str  # "completed" | "max_steps" | "error" | "interrupted" | "cost_limit" | "dedup_break"
    text: str
    final_text: str
    messages: "list[Message]"
    steps: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_calls_count: int = 0
    # Cache-token totals across the run.  Plumbed through for sub-agent
    # accounting so parent stats see real cache hits / writes instead of
    # dropping them to zero at the sub-agent boundary.  Zero when the
    # provider doesn't report caching (OpenAI only populates cache_read).
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    # Number of discrete LLM API calls made during the run.  Used by the
    # sub-agent accounting path so parent stats.api_calls reflects the
    # child's actual call count, not just "1 per dispatch".  Typically
    # equals ``steps`` but can differ when the loop retries a failed
    # call or issues fallback best-effort completions (max_steps path).
    api_calls: int = 0
    error: str | None = None
    # Final state of the task-plan scratchpad.  Empty list when todos are
    # disabled or the model never wrote any.  Outer layers (agent.py) are
    # expected to persist this across turns.
    todos: list = field(default_factory=list)
    # Names of mutating tools (run_shell, edit_file, write_file, mcp,
    # dispatch_agent, run_background) that hit the dispatch-loop timeout
    # during this run.  A timed-out
    # mutating tool's daemon thread keeps running past the timeout report,
    # so its side effect may have partially or fully landed in the
    # environment while the model received an "error" ToolResult.  Outer
    # layers that replay or retry a turn should consult this list to
    # reconcile environment state (e.g. re-read edited files, re-check
    # spawned backgrounds) before trusting the LLM's follow-up plan.
    # Empty list when no mutating timeouts occurred (the common case).
    partial_side_effects: list[str] = field(default_factory=list)


def build_default_loop_config(**overrides: Any) -> LoopConfig:
    """Factory: build a ``LoopConfig`` from the app's default constants.

    Centralizes the wiring between ``jyagent.config`` (DEFAULT_*) and
    ``LoopConfig`` so the main run-loop doesn't carry a 19-line inline
    constructor.  Pass ``**overrides`` to tweak fields (used by tests
    and any future per-mode config).

    Kept in this module so it lives next to ``LoopConfig`` itself; the
    one-time ``jyagent.config`` import cost is paid lazily on first call.
    """
    from ...config import (
        DEFAULT_MAX_STEPS, DEFAULT_MAX_TOKENS, MAX_TOKENS_CAP,
        DEFAULT_TOOL_TIMEOUT, MAX_WORKING_TOKENS,
        COMPACT_TOOL_RESULT_CHARS, MAX_TOOL_RESULT_CHARS,
    )
    defaults: dict[str, Any] = dict(
        max_steps=DEFAULT_MAX_STEPS,
        initial_max_tokens=DEFAULT_MAX_TOKENS,
        max_tokens_cap=MAX_TOKENS_CAP,
        auto_scale_on_truncation=True,
        token_scale_factor=2,
        concurrent_tools=True,
        max_tool_workers=4,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        retry_attempts=10,
        retry_base_delay=2.0,
        compact_messages=True,
        max_working_tokens=MAX_WORKING_TOKENS,
        compact_tool_result_chars=COMPACT_TOOL_RESULT_CHARS,
        max_tool_result_chars=MAX_TOOL_RESULT_CHARS,
        streaming=True,
        truncate_large_inputs=True,
        fallback_on_max_steps=True,
    )
    defaults.update(overrides)
    return LoopConfig(**defaults)


__all__ = ["LoopConfig", "LoopResult", "build_default_loop_config"]
