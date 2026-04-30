"""LoopConfig and LoopResult dataclasses for the agent loop runtime.

Extracted from engine.py during the runtime-package refactor (phase 3).
Kept as plain dataclasses (no engine deps) so callers can build configs
without paying the engine import cost.
"""
from __future__ import annotations

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


__all__ = ["LoopConfig", "LoopResult"]
