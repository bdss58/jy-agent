---
created: 2026-04-26T00:36:53+08:00
updated: 2026-04-26T13:35:00+08:00
---
# Tier C C1 (DONE) + C4 phased plan (Phases 1-4 DONE; Phase 5 needs re-scoping)

Codex's review of `jyagent.runtime` (2026-04-25) flagged two larger items. C1 landed in one commit; C4 is being delivered phase by phase — four phases shipped so far. The originally-planned Phase 5 (rename `AgentLoop` → `LoopController`) needs re-scoping because `_run_impl` is ~733 lines on its own and wasn't touched by Phases 1-4.

## C1 — Cancellation latency in blocking provider calls — DONE (commit `85e7719`)

### Problem
`cancel_event` was only checked between loop boundaries and between stream yields. A blocked provider (slow network, stuck mid-chunk, or full sync `complete()` call with zero yield points) would hang until the provider HTTP timeout (60-300 s typical).

### Fix shipped
- **`_call_complete`**: pre-call guard + daemon worker + 100 ms main-thread poll. On cancel raises `KeyboardInterrupt` within ~1.5 s latency bound. Worker thread drains in background (Python threads aren't cancellable — same trade-off as the mutating-tool-timeout path).
- **`_call_streaming`**: spawn a daemon watcher that waits on `cancel_event` with 50 ms polling and calls `stream.close()` when fired. Closing the SDK stream releases the httpx response; the blocked `for event in stream` unblocks with an exception the existing except-clause normalises. Latency bound ~0.5 s.
- **Fast path preserved** on both: when `cancel_event is None` (non-CLI path), calls run inline with zero worker-thread overhead.

### Tests
`tests/test_codex_review_fixes.py::TestC1CancellationLatency` × 4 cases covering both paths, cancel latency bounds, fast-path preservation, and pre-call guard.

---

## C4 — Split `engine.py` into 5 owned components — Phases 1-2 DONE

### engine.py trajectory
- Before C4: 2226 lines.
- After Phase 1 (cost): 2187 lines (-77 extracted to `cost.py`, some offset from accumulated Tier-B/C additions).
- After Phase 2 (tool_executor): 1862 lines (-364 extracted to `tool_executor.py`).
- After Phase 3 (llm_runner): 1568 lines (-294 extracted to `llm_runner.py`, net after inlining a small retry loop back).
- After Phase 4 (compaction): 1394 lines (-174 extracted to `compaction.py`).
- **Total across Phases 1-4: 832 lines pulled out of engine.py.**
- **Remaining bulk is `AgentLoop._run_impl` alone — ~733 lines (L539-L1271).** This is the real elephant, and it was never on the original C4 extraction plan.

### The proven pattern (verified twice)
1. Write new module with clean public API + doc header matching `cost.py` style.
2. Replace inline code with `from .newmod import X as _X` aliases in engine.py.
3. For **mutable module state** (pool, cap, lock): add a PEP-562 `__getattr__` at the bottom of engine.py that forwards to the canonical module on every read — this is critical because values get rebound on events like pool growth, and a plain `from X import Y` snapshots at import time and goes stale.
4. Run full suite — must be 0 regressions (no behavior change).
5. Add targeted import-path tests (both new name + engine re-export identity check + live-value tracking for mutable state).

### Phase 1 — `_CostTracker` → `runtime/loop/cost.py` ✅ DONE (commit `4e7b5d5`)
- Renamed public class to `CostTracker`; engine re-exports as `_CostTracker` for back-compat.
- +4 tests, 566 → 570 passed, 0 regressions.
- ~½ day effort (mostly careful dependency-audit; the code itself is a 3-method class).

### Phase 2 — tool execution helpers → `runtime/loop/tool_executor.py` ✅ DONE (commit `13fa6b2`)
- Moved: `get_tool_dispatch_executor`, `execute_tool`, `execute_tool_with_timeout`, `execute_tools`, plus the pool/lock/cap module state.
- Kept in engine.py: `_is_transient_error` (belongs to Phase 3 retry logic, not tool execution).
- Back-compat: function aliases + PEP-562 `__getattr__` passthrough.
- One existing test (`test_backcompat_alias_points_to_dispatch`) needed a touch-up: it imported `_tool_dispatch_executor` at module-load via `from engine import ...`, which captures a stale snapshot once any later test grows the pool. Rewrote the assertion to read both names live. Also fixed `TestDispatchExecutorGrowsWithConfig` to save/restore on `tool_executor.tool_dispatch_executor` rather than setting a static attribute on engine that would shadow `__getattr__`.
- +6 new tests in `TestC4Phase2ToolExecutorExtraction`. 570 → 576 passed, 0 regressions.
- **Delivery: ~80% via Claude Code delegation (died mid-task on HTTP 424 peer reset after 23 min — same provider flakiness class we documented for Codex). jy-agent finished the PEP-562 shim + new tests + the one existing-test touch-up.**
- Lessons: (a) Claude Code `--bare` mode CAN die mid-task on network blips; always budget for ~20% salvage work. (b) The PEP-562 passthrough is essential for any extraction that includes rebindable module state — I should have caught the save/restore shadowing bug in `TestDispatchExecutorGrowsWithConfig` during Phase 1 (retroactively, Phase 1 didn't need it because `CostTracker` is a class, not mutable state).

### Phase 3 — LLM call + retry/fallback → `runtime/loop/llm_runner.py` ✅ DONE (commit `7785c1e`)
- Moved to `llm_runner.py`: `call_complete`, `call_streaming`, `call_with_retry`, `extract_text`, `extract_tool_calls`, `is_transient_error`, `build_runtime_options`, plus the C1 cancel-worker (complete path) and cancel-watcher (streaming path).
- Kept in `engine.py`: **the retry loop itself, rewritten from the inline form to dispatch through `self._call_streaming` / `self._call_complete`.** That preserves the override contract: tests that subclass `AgentLoop` and override the underscore methods to inject transient failures still see their patch applied. Routing `_call_llm_with_retry` straight to `LLMRunner.call_with_retry` would have broken that (bypassing the subclass's overrides) — a real regression in `TestCallLLMRetry::test_retry_then_success_*`.
- `_call_complete` and `_call_streaming` are one-line delegates onto `LLMRunner`; the runner is built lazily in `_get_llm_runner()` (memoised) so post-`__init__` swaps of runtime_owner / callbacks / cancel_event / model_spec still apply.
- Back-compat aliases: `_extract_text`, `_extract_tool_calls`, `_is_transient_error`, `_build_runtime_options` in engine.py all point to the llm_runner originals via `from .llm_runner import X as _X`.
- **Two test patch-target moves** were needed:
  - `test_subagent.py::test_cancel_interrupts` patches `get_reasoning_config_for_provider` — target moved from `engine` to `llm_runner` (must patch where the symbol is *looked up*, not where it's defined).
  - `test_loop_engine_p0_fixes.py::test_stream_loop_has_cancel_check` inspects the source of `_call_streaming` — now inspects `LLMRunner.call_streaming` because the code moved.
- +7 new tests in `TestC4Phase3LLMRunnerExtraction`: import-path identity, runner cache behaviour, delegate forwarding for both paths, and the critical `call_with_retry` subclass-override preservation test.
- 576 → 583 passed (non-`test_web_search` suite), 0 regressions.
- **Self-delivered, not delegated** (previous session died mid-task when `jy-agent` got `Killed: 9` — likely OOM on a tight-loop step; recovered by reading worktree state and continuing from step 21 onward). Effort: ~1 h of focused work, most of it on the subclass-override contract realisation.
- Risk-management hit: Codex's review flagged Phase 3 as **HIGH risk**. The one non-obvious landmine was exactly the `_call_llm_with_retry` dispatch path — caught on the first test run (`TestC1CancellationLatency` pre-existing coverage + a transient-failure retry test both surfaced it). Lesson: extraction plans should include "audit what contract each public method exposes to subclasses / monkeypatches" as a named step, not rely on test coverage to catch it.

### Phase 4 — message compaction + truncation → `runtime/loop/compaction.py` ✅ DONE (commit `ef970e5`)
- Moved to `compaction.py`: `truncate_result`, `compact_messages` (3-tier: thinking-block pruning, observation masking, priority-aware), `truncate_tool_call_blocks`.
- Pure functions — no closure state, no provider I/O, no callbacks. Lowest-risk phase of the four.
- Back-compat aliases: `_truncate_result`, `_compact_messages`, `_truncate_tool_call_blocks` in engine.py all point to compaction-module originals via `from .compaction import X as _X`.  No PEP-562 shim needed (nothing mutable).
- +5 new tests in `TestC4Phase4CompactionExtraction`: import-path identity, truncate_result head/tail split + error-passthrough, compact_messages fast-path identity return, truncate_tool_call_blocks no-op on unknown tools.
- 583 → 588 passed (non-`test_web_search` suite), 0 regressions.
- Effort: ~30 min of focused work, self-delivered. Went according to the plan exactly — the "low risk" estimate was accurate.

### Phase 5 — needs re-scoping (original plan: rename `AgentLoop` → `LoopController`)

**The original Phase 5 estimate ("engine.py becomes LoopController, ~600-700 lines, medium risk") is off by 2×.**

Actual post-Phase-4 state of engine.py:
```
L50-L201  : core types, _StuckLoopDetector, shared dispatch executor ( ~150 lines )
L202-L333 : helpers (_is_truncated, _strip_dangling_verification, _finalize_run, compaction aliases)
L336-L537 : AgentLoop.__init__ + run() entry + helpers              ( ~200 lines )
L539-L1271: AgentLoop._run_impl                                     ( ~733 lines )   ← the real bulk
L1272-L1365: LLM call delegates (Phase 3)                           (  ~95 lines )
L1368-end : PEP 562 back-compat shim
```

**`_run_impl` is the elephant** — the single biggest method in the codebase, with ~30 control-flow branches, 7-8 step-phase handoffs, and invariants that must hold on every exit path (`_finalize_run()`, partial side-effects accumulator, run-lock release).  It was never on the C4 extraction plan because Codex's original recommendation was to split engine by *responsibility domain* (cost / tools / LLM / compaction), not by size.

Two reasonable paths forward — should be a fresh decision with the user, not continued from the old plan:

**Option A: Accept the current state.**  engine.py is 1394 lines, down from 2226 (-37 %).  The four extracted modules each have a clear single responsibility.  A 1400-line file with one big orchestrator method is a reasonable endpoint.  Rename `AgentLoop` → `LoopController` *purely* as a naming change if wanted; cost is ~20 breaking imports across the codebase.  Low value.

**Option B: Extract `_run_impl` step-body into `runtime/loop/step.py`.**  Carve out the "what happens in one loop iteration" logic as `run_step(state, ...) -> StepResult` and leave `_run_impl` as a thin `while` loop that threads state between steps.  This is a 1-day refactor, medium-to-high risk (the step state object has ~15 fields; any missed field silently breaks multi-step runs).  Needs its own test battery at step-level before merging.  High value — `_run_impl` becomes testable without provider mocking.

Recommendation: **ship Phases 1-4 as the C4 delivery, close out the codex-review branch, and decide Option A vs B as a separate project.** The branch as-is delivers the bulk of the structural value Codex called for and is ready to merge on its own merits.

---

## Cross-cutting: Delegation-flakiness patterns

Both Codex (`crs.us.bestony.com`) and Claude Code (2.1.118 via the API layer) have shown mid-task network failures on long jobs (~20-30 min mark). Mitigations in place:
- `model_reasoning_summary = "auto"` in `~/.codex/config.toml` — shifts Codex disconnect distribution toward success.
- For Phase 3 (the remaining high-risk phase), route to a stable provider (OpenAI direct or `models-proxy` profile).
- Always plan for ~20% salvage work: Claude Code in `--bare` mode dies silently at the API layer but leaves partially completed work on disk that can be finished manually.

---

## Branch state (end of 2026-04-25 session)

`runtime/codex-review-fixes-2026-04-25`:
```
ef970e5 C4 Phase 4 — extract compaction.py
7785c1e C4 Phase 3 — extract llm_runner.py
13fa6b2 C4 Phase 2 — extract tool_executor.py
4e7b5d5 C4 Phase 1 — extract cost.py
85e7719 C1 — cancellation latency
74097ba C2/C3 — reentrance guard + SessionStats locking
b3f71ed B1/B2/B3 — cost tracking, immutability, timeout coercion
e1dfcaa A1 — mutating timeout → LoopResult
b11dd3a A2/A3/A4 — dispatch executor, tracing non-fatal, path sanitisation
```
9 commits total, **588 passing tests** on the non-`test_web_search` suite (+12 new tests across the 4 C4 phases; 4 pre-existing web_search failures unrelated). Local-only, not pushed/merged. `data/memory/` work on main still stashed (`stash@{0}: wip-memory-updates`).

engine.py: 2226 → 1394 lines (-832, -37 %). Four new modules under `runtime/loop/`: `cost.py` (77), `tool_executor.py` (449), `llm_runner.py` (567), `compaction.py` (245).
