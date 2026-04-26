---
created: 2026-04-26T18:15:00+08:00
status: design (pre-review, pre-implementation)
author: jy-agent (Codex temporarily unavailable; pending second-opinion pass)
---

# C4 Phase 5 — Extract `_run_impl` body to `runtime/loop/step.py`

## Goal

Move the per-step loop body (~470 lines of `AgentLoop._run_impl`) into a
new `runtime/loop/step.py` module so:
1. The step body is unit-testable without a full provider mock chain.
2. `engine.py` shrinks from 1394 → ~750-800 lines and gains a clear
   responsibility split: orchestration (loop + post-loop) vs. step body.
3. Future changes to step semantics (new phase hook, new terminal
   condition, different reflection cadence) land in a single ~500-line
   file with clear inputs and outputs.

## Anatomy of `_run_impl` today (L539-L1271, 733 lines)

| Range | Lines | Concern | Where it goes |
|-------|------:|---------|---------------|
| L547-L628 | ~80 | **Setup** — init accumulators, lazy-import reflection, seed todos + bind `write_todos` closure, build cost_tracker / stuck_detector / tracer | `RunState.from_loop()` in step.py |
| L631-L1101 | ~470 | **Per-step body** — the for-loop iteration | `step.run_step(loop, state)` |
| L1103-L1232 | ~130 | **Post-loop terminal handling** — cancelled exit, max_steps fallback, max_steps exit | **stays in engine.py** |
| L1234-L1262 | ~30 | **`except KeyboardInterrupt` / `except Exception`** | **stays in engine.py** |

**Key insight**: only the per-step body and setup move out. The for-loop
itself, the post-loop handlers, and the try/except scaffolding stay in
engine — those are orchestration, not step semantics.

## Design Decisions

### D1 — Cut at the per-step body, NOT at `_run_impl` as a whole

Engine retains:
```python
def _run_impl(self, system_prompt, messages, initial_todos=None):
    state = RunState.from_loop(self, system_prompt, messages, initial_todos)
    try:
        for step in range(self._config.max_steps):
            state.step = step
            outcome = run_step(self, state)
            if isinstance(outcome, StepTerminate):
                return outcome.result
            if isinstance(outcome, StepBreak):
                break  # → cancelled-exit handler below
        # … cancelled-exit / max_steps-fallback / max_steps-exit (~130 lines)
    except KeyboardInterrupt:
        return _finalize_run(status="interrupted", …)
    except Exception as e:
        return _finalize_run(status="error", …)
```

**Why not move the for-loop too?** Coupling step.py to `_finalize_run` and
the max_steps fallback (which itself does a no-tools LLM call!) widens the
blast radius. The for-loop is engine orchestration; one step is step
semantics. Clean cut.

### D2 — Mutable `RunState` dataclass passed by reference

14 accumulators is too many for positional args, and frozen-state-with-
replace would buy zero functional purity (we're calling `messages.append`,
firing callbacks, executing tools). Mutable shared state matches today's
behavior exactly.

```python
# jyagent/runtime/loop/step.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .engine import AgentLoop
    from .cost import CostTracker
    from ..tools.registry import ToolBatch
    from ...llm.types import ModelSpec
    from ..result import LoopResult

@dataclass
class RunState:
    """Mutable per-run state threaded through every run_step call.

    Built once by RunState.from_loop() at the top of _run_impl,
    mutated in place by run_step on every iteration.
    """
    # Conversation
    system_prompt: str
    messages: list                 # mutated in place
    turn_start_idx: int            # for verification-gate boundary check

    # Step bookkeeping (set by engine before each run_step call)
    step: int = 0

    # Token / cost accumulators
    current_max_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Tool / reflection counters
    tool_calls_count: int = 0
    last_reflection_count: int = 0

    # Truncation retry state
    consecutive_truncations: int = 0
    max_truncation_retries: int = 3

    # Verification gate (one-shot)
    verification_injected: bool = False

    # Text accumulators
    all_text: str = ""
    final_text: str = ""

    # One-shot warning flag
    unpriced_warned: bool = False

    # Heavy collaborators (built in setup; never replaced after)
    cost_tracker: CostTracker | None = None
    stuck_detector: Any = None        # _StuckLoopDetector (private to engine.py)
    tools_batch: Any = None           # ToolBatch
    trace: Any = None                 # tracer or None
    effective_spec: Any = None        # ModelSpec

    # Optional collaborators (None when feature is disabled)
    write_todos_fn: Any = None        # closure over self._todos
    reflection_module: Any = None     # lazily imported

    @classmethod
    def from_loop(
        cls,
        loop: "AgentLoop",
        system_prompt: str,
        messages: list,
        initial_todos: list | None,
    ) -> "RunState":
        """Run the setup phase: init counters, build collaborators,
        seed todos, bind write_todos closure. Mutates ``loop._todos``
        and ``loop._partial_side_effects`` (those live on the loop, not
        on state, because tool_executor and the write_todos closure
        already reach for them via ``self.``)."""
        ...
```

### D3 — Tagged-union outcome with three variants

```python
@dataclass(frozen=True)
class StepContinue:
    """Step finished cleanly; outer loop continues to next iteration."""
    pass

@dataclass(frozen=True)
class StepTerminate:
    """Step produced a terminal LoopResult; outer loop returns it."""
    result: "LoopResult"

@dataclass(frozen=True)
class StepBreak:
    """Step requests outer-loop break (e.g. cancel before/after tools).
    Engine runs the post-loop cancelled-exit handler."""
    reason: Literal["cancelled"]

StepOutcome = Union[StepContinue, StepTerminate, StepBreak]


def run_step(loop: "AgentLoop", state: RunState) -> StepOutcome:
    """Execute one iteration: tools_batch refresh, compaction, LLM call,
    tool dispatch, reflection injection, checkpoint, and the inline
    early-return paths (cost_limit, stuck_loop, completed-no-tools,
    truncation-retry).

    The engine owns the for-step counter and post-loop terminal
    handling; this function owns everything within one step.
    """
    ...
```

### D4 — Subclass-override contract preserved by construction

`run_step` calls `loop._call_llm_with_retry(...)`, `loop._call_complete`,
`loop._call_streaming`, `loop._fire`, `loop._is_cancelled`,
`loop._cancellable_sleep`, `loop._write_checkpoint`. All resolved via
Python's normal attribute lookup on the `loop` argument — so test
subclasses that override these methods work automatically. **No retry
loop or LLM machinery moves into step.py.** This is the same lesson
Phase 3 navigated.

### D5 — `loop._partial_side_effects` and `loop._todos` stay on the loop

These live on the AgentLoop instance because:
- `_partial_side_effects` is read by `tool_executor.execute_tools` via
  the `partial_side_effects=` kwarg the engine passes.
- `_todos` is read+written by the `write_todos` closure built in
  `RunState.from_loop()`. The closure captures `loop._todos` by
  reading the attribute on every call (via getter/setter wrappers), so
  later mutation by the engine is visible.

**Anti-pattern**: copying these into RunState fields. They're tied to the
AgentLoop lifecycle (next-turn reuse, cross-turn persistence), not the
single-run lifecycle.

### D6 — Terminal-exit handling

| Condition | Where detected | Outcome |
|-----------|----------------|---------|
| Cost budget exceeded | inside `run_step` | `StepTerminate(LoopResult(status="cost_limit"))` |
| Completed (no tool calls, no verification needed) | inside `run_step` | `StepTerminate(LoopResult(status="completed"))` |
| Repeated truncation (>3) | inside `run_step` | `StepTerminate(LoopResult(status="error", error="Repeated truncation"))` |
| Stuck-loop (dedup_break) | inside `run_step` | `StepTerminate(LoopResult(status="dedup_break"))` |
| Cancel before/after tools | inside `run_step` | `StepBreak(reason="cancelled")` → engine runs cancelled-exit handler |
| Cancel at top of loop | inside `run_step` (first check) | `StepBreak(reason="cancelled")` |
| Cancel between iterations | engine outer loop | `_is_cancelled()` check before next `run_step` (matches today) |
| max_steps reached | engine outer loop | post-loop fallback / max_steps handler |
| Truncation-retry (continue) | inside `run_step` | `StepContinue` after rolling back `state.all_text` |
| Verification injection | inside `run_step` | `StepContinue` after appending `[VERIFICATION]` |

This mapping makes every line of the current `_run_impl` reachable in
the new layout with no behavior change.

### D7 — Helper functions inside step.py (private)

Optional Phase 5b cleanup. The 470-line `run_step` naturally splits into:

```
_refresh_tools_batch(loop, state, registry) -> ToolBatch
_compact_and_build_context(loop, state, step_batch) -> dict  # the LLM "context" arg
_apply_phase_directive(loop, state, opts, step) -> LLMOptions
_handle_no_tool_response(loop, state, final_message, step_text, step_batch)
                                                     -> StepOutcome
_handle_truncation(loop, state, stop_reason, blocks, step_text)
                                                     -> StepOutcome | None
_execute_tool_batch(loop, state, blocks, step_batch) -> list[(block, result)]
_check_stuck_loop(loop, state, tool_results) -> StepOutcome | None
_maybe_inject_reflection(loop, state, tool_results)
_maybe_checkpoint(loop, state)
```

Punted to Phase 5b. First-pass `run_step` is a verbatim move with no
internal restructuring — keeps the diff reviewable.

## Risk Ranking (highest first)

1. **Mutable-collaborator identity drift** (cost_tracker, stuck_detector,
   tools_batch). If a future contributor sets `state.cost_tracker = ...`
   instead of mutating it, the engine's view diverges. **Mitigation**:
   document in the dataclass docstring; existing end-to-end tests
   (`test_cost_budget`, `test_dedup_break`) catch the regression.

2. **`_partial_side_effects` and `_todos` accidentally moved into
   RunState** — would silently break cross-turn state. **Mitigation**:
   D5 explicitly leaves them on the loop; review any PR that touches
   them.

3. **Verification gate boundary check** (`state.step + 1 < cfg.max_steps`)
   depends on `state.step` being set by engine before each call.
   **Mitigation**: assert `state.step` was assigned by engine at top of
   `run_step`. Existing P0 test `test_verification_gate_max_steps_boundary`
   covers it.

4. **Phase 3 lesson re-emergence**: forgetting that LLM call dispatch
   must go through `loop._call_llm_with_retry` (not directly to
   `LLMRunner`). **Mitigation**: D4 is explicit; existing
   `TestCallLLMRetry::test_retry_then_success_*` catches it.

## Test Strategy

### Existing tests (regression net) — must stay 608 passing

End-to-end tests via `AgentLoop.run()` cover all terminal exits:
- `test_completed_no_tools` → StepTerminate completed
- `test_cost_budget_exceeded` → StepTerminate cost_limit
- `test_dedup_break` → StepTerminate dedup_break
- `test_max_steps_reached` → engine post-loop max_steps
- `test_max_steps_fallback` → engine post-loop fallback
- `test_cancel_before_tools` / `test_cancel_after_tools` → StepBreak
- `test_truncation_retry` → StepContinue after rollback
- `test_verification_gate` → StepContinue with [VERIFICATION] tail
- `test_phase_policy_*` → tool_choice override
- `test_reflection_*` → reflection injection
- `test_checkpoint_periodic` → checkpoint write

### New tests uniquely enabled by step extraction (`tests/test_step_runner.py`)

Tests that today require building a full AgentLoop + LLMRunner +
provider chain become 5-line dataclass constructions:

```python
def test_run_step_terminates_on_cost_limit():
    loop = build_minimal_loop()  # only needs _fire / _config / _call_llm_with_retry
    state = RunState(
        system_prompt="", messages=[], turn_start_idx=0,
        cost_tracker=CostTracker(),
        # ...
    )
    state.cost_tracker._cost = 999.0  # already over budget
    loop._call_llm_with_retry = lambda *a: ("", [], "end_turn", {"usage": {}})
    outcome = run_step(loop, state)
    assert isinstance(outcome, StepTerminate)
    assert outcome.result.status == "cost_limit"
```

Coverage targets:
- T1: cost_limit termination
- T2: completed-no-tools termination (text-only response)
- T3: completed-no-tools triggers verification gate when mutations present
- T4: dedup_break termination on repeated identical tool result
- T5: dedup_break NOT triggered by within-batch duplicate (Codex Part 2 #2 fix)
- T6: truncation retry rolls back `state.all_text`
- T7: truncation retry exhausted → StepTerminate error
- T8: cancel before tools → StepBreak
- T9: cancel after tools → StepBreak
- T10: phase_policy override applied to opts
- T11: reflection injected at every-N cadence
- T12: reflection injected after sub-agent return
- T13: periodic checkpoint written at boundary
- T14: tools_batch refresh on registry version bump
- T15: tools_batch refresh skipped when version unchanged

15 new tests target the previously-untestable seams.

## Ship Plan (4 commits)

### C1 — `feat(runtime): introduce step.py with RunState + StepOutcome` (~30 min, low risk)

- Create `jyagent/runtime/loop/step.py` with:
  - `RunState` dataclass (no `from_loop` yet)
  - `StepContinue` / `StepTerminate` / `StepBreak` dataclasses
  - Stub `run_step(loop, state)` that delegates to a NEW private method
    `loop._run_one_step(state)` containing the inline body verbatim
- Engine: extract the for-loop body (L631-L1101) into `_run_one_step`,
  return one of the three outcome types instead of `continue/break/return`.
- Engine: outer for-loop now calls `run_step(self, state)` and dispatches
  on outcome type.
- **Setup phase + post-loop handlers untouched.**
- Run full suite — must be 608 passed, 0 regressions.
- +3 tests: import-path identity, RunState field coverage, StepOutcome
  variants.

### C2 — `refactor(runtime): move _run_one_step body from engine to step.run_step` (~45 min, mechanical)

- Cut the body of `AgentLoop._run_one_step` and paste into
  `step.run_step`, replacing `self.X` → `loop.X` throughout.
- Delete `_run_one_step` from engine.
- Run full suite — must be 608 passed.

### C3 — `refactor(runtime): extract setup phase into RunState.from_loop` (~30 min, low risk)

- Cut the ~80-line setup block (L547-L628) and paste into
  `RunState.from_loop()` classmethod.
- Engine `_run_impl` opens with `state = RunState.from_loop(self, ...)`.
- Run full suite.

### C4 — `test(runtime): step-level unit tests` (~1.5 hr)

- Add `tests/test_step_runner.py` with the 15 tests listed above.
- Each constructs a minimal AgentLoop and a RunState directly, monkey-
  patches `_call_llm_with_retry` to return a scripted response, calls
  `run_step`, asserts outcome + state mutations.
- Final: 608 + 15 = 623 passing, 0 regressions.

### Optional C5 — `refactor(runtime): split run_step into private helpers` (Phase 5b)

- Apply D7 helper extraction inside step.py.
- No engine.py changes.
- Defer until C1-C4 land and stabilize.

## Effort Estimate

- C1: 30 min
- C2: 45 min
- C3: 30 min
- C4: 90 min
- **Total: ~3.5 hours focused work**, well under the original "1 day"
  estimate because the four prior phases proved the extraction template.
- C5 (optional Phase 5b): another ~2 hours if pursued.

## What gets us to "engine.py is done"

After C1-C4:
- engine.py: 1394 → ~750 lines
  - Setup gone (-80)
  - For-loop body gone (-470)
  - Outer for-loop + dispatch (~30 new lines)
  - Post-loop handlers stay (~130 lines)
  - LLM delegates stay (~90 lines)
  - Class scaffold + __init__ + run() + helpers (~430 lines)
- step.py: ~550 lines (RunState + dataclasses + run_step + from_loop)
- New owned modules total: cost(77) + tool_executor(449) + llm_runner(567)
  + compaction(245) + step(550) = 1888 lines, all single-responsibility.

This is the natural endpoint of the C4 effort. After this, the
`AgentLoop` → `LoopController` rename (original Phase 5 Option A) becomes
a 5-minute symbol change if still wanted.

## Pending: Codex critique pass

This document was authored without the Codex second-opinion pass the
user requested (Codex was unavailable on both `neo` and `models-proxy`
endpoints at design time — 403 / high-demand errors). Before C1 ships,
re-run Codex against this document with prompt:

> Critique the design at `data/memory/topics/c4-phase5-step-extraction-design.md`.
> Specifically: (a) is the cut at "per-step body only" (D1) right, or
> should the for-loop and post-loop also move? (b) does mutable RunState
> (D2) trade off testability for safety in a way that bites us in 6
> months? (c) is the StepOutcome tagged union (D3) overkill — should
> StepBreak just be a special StepTerminate? (d) any subtle invariant
> in `_run_impl` you'd flag that this design misses?

Apply Codex's critique before starting C1 if it surfaces a real concern.
