---
created: 2026-04-25T18:20:11+08:00
source: codex-cli (3 sequential `codex exec --sandbox read-only` runs)
scope: jyagent/runtime/* (engine, loop/*, tools/*, stats, skills) — ~3400 LOC
updated: 2026-04-25T18:20:11+08:00
---
# Runtime Design + Implementation Review (Codex)

## Method

Single broad review died at 103k tokens (network errors mid-summary — same pattern documented in `runtime-refactor.md`). Switched to 3 narrow scoped runs with explicit "synthesize as you go" directive. Two completed cleanly; part 3 died before final TOP 3 RISKS but had already emitted all 9 findings. Total: **34 findings, all with file:line refs**.

## Top architectural risks (consolidated across all 3 runs)

1. **Daemon-thread tool timeout leaks state** — `_execute_tool_with_timeout` (engine.py:573-650) returns a timeout `ToolResult` while the daemon body keeps running. Side effects continue after the loop moved on; `body_permits` is released early (defeating `max_tool_workers`); contextvars don't propagate; non-parallel-safe mutating tools are no longer real barriers. Compounds: model retries while first daemon still mutating.
2. **`[VERIFICATION]` cleanup is incomplete** — exit paths at engine.py:1117 (cost_limit), 1203 (repeated truncation), 1404 (cooperative cancel) skip `_strip_dangling_verification()`. Unanswered prompt poisons next turn.
3. **Registry/stats reads bypass write locks** — `is_parallel_safe()`, `get_schema()`, `get_timeout_hint()`, etc. are unlocked (registry.py:53-72). `SessionStats.provider/model/elapsed` properties read without lock (stats.py:134). `_MODEL_PRICING` mutation is unsynchronized with concurrent `_lookup_pricing()` iteration → potential `dict size changed during iteration`.
4. **`runtime` is not UI-/LLM-clean** — `SkillManager` (skills.py:458-623) directly imports `LLMOwner`, calls `complete_text`, prints colored output to stderr. Engine itself imports `from ...llm import LLMOwner, LLMOptions, ModelSpec` (engine.py:21-22). Direction is reversed.
5. **`SkillManager` doesn't belong in `runtime/`** — engine has zero dependency on it; it's prompt-routing + filesystem + LLM-routing glue. Should live in `app/` or its own package.
6. **`on_assistant_message` callback can break tool_use/tool_result pairing** — runs at engine.py:1240 before tool results appended for original tool-call IDs, no invariant check after. Also not applied to verification or fallback append paths (inconsistent contract).
7. **Cost budget drifts under concurrency** — `_CostTracker` (engine.py:104) has its own simplified pricing, doesn't account for cache reads / long-context multipliers, and doesn't see usage from concurrently-dispatched LLM-backed tools.

## Part 1 — Concurrency & State (engine.py 700-1200, stats, tools/*) — 12 findings

(See `/tmp/codex_review_part1.md` for full text. Key file:line refs:)

| # | Issue | File:line |
|---|---|---|
| 1 | Daemon tool body continues after timeout return | engine.py:621, 638-650 |
| 2 | `body_permits` released while daemon still running → `max_tool_workers` cap defeated | engine.py:629, 638-641 |
| 3 | No `contextvars.copy_context()` for tool body threads | engine.py:632 |
| 4 | Registry metadata/schema reads unlocked, racing with snapshot | registry.py:49; engine.py:466, 993-994 |
| 5 | `SessionStats.provider/model/elapsed` properties bypass `_lock` | stats.py:134 |
| 6 | `set_model_pricing()` unsynchronized with `_lookup_pricing()` iteration | stats.py:85 |
| 7 | `ToolResult` doesn't enforce string content (engine assumes it) | result.py:18; engine.py:1283 |
| 8 | Error semantics rely on caller setting `is_error` (no validation of `"Error: ..."` content) | result.py:18 |
| 9 | `_CostTracker` blind to concurrent tool LLM usage | engine.py:1097 |
| 10 | `_CostTracker` reimplements pricing without cache-read / long-context handling | engine.py:104 |
| 11 | `is_parallel_safe()` is live read, not snapshotted per batch | engine.py:520, 535, 537 |
| 12 | `snapshot()` returns shared schema dicts (mutable post-registration) | registry.py:44 |

## Part 2 — State Machine (phases/reflection/verification/remediation/checkpoint/todos) — 13 findings

(See `/tmp/codex_review_part2.md`. Key:)

| # | Issue | File:line |
|---|---|---|
| 1 | Default phase policy collapses `verify` into `finalize` for `max_steps≤2` | phases.py:73-79 |
| 2 | Verification de-dup is tail-only — TODO/reflection appends can re-arm it | verification.py:89-108 |
| 3 | Checkpoint uses `os.replace` but no `fsync()` on file or parent dir | checkpoint.py:66-73 |
| 4 | `write_todos` has no merge guard (replace-all). Mitigated only because it's not registered as parallel-safe — implicit, fragile | todos.py:212-246; engine.py:1001-1004 |
| 5 | `should_verify()` scans entire history for mutations — old replayed `write_file` triggers verify on a non-mutating new run | verification.py:111-135; engine.py:1153-1157 |
| 6 | Cost-limit exit at engine.py:1117 leaks dangling `[VERIFICATION]` |
| 7 | Repeated-truncation exit at engine.py:1203 leaks dangling `[VERIFICATION]` |
| 8 | Checkpoint replay loses `last_reflection_count`, `verification_injected`, truncation-retry, stuck-detector state | checkpoint.py:37-44 vs engine.py:916, 921 |
| 9 | `on_assistant_message` runs at engine.py:1240 before tool-result append → can desync tool_call_id pairing |
| 10 | Max-step fallback sends `tool_choice={"type":"none"}` after deleting `tools` — provider may reject; exception swallowed at engine.py:1495-1497 |
| 11 | Tool timeout breaks "non-parallel-safe = sequential barrier" invariant for mutations | engine.py:497-500 vs 632-649 |
| 12 | Cooperative cancellation path at engine.py:1404-1418 doesn't strip dangling verification (KeyboardInterrupt path does) |
| 13 | Checkpoint load has no replay contract for TODO restoration |

## Part 3 — API Surface & Coupling (runtime/__init__, callbacks, config, tracing, skills) — 9 findings

(See `/tmp/codex_review_part3.md`. Key:)

| # | Issue | File:line |
|---|---|---|
| 1 | `import jyagent.runtime` eagerly loads engine — contradicts `config.py:4-5` "no engine import cost" promise | runtime/__init__.py:2 → loop/__init__.py:2 → engine.py |
| 2 | Module-level `ThreadPoolExecutor` + `atexit.register` fires on plain import | engine.py:71-74 |
| 3 | `SkillManager` is not an engine dependency — engine never imports it | skills.py:260-768 vs engine.py:759-906 |
| 4 | `SkillManager._route_llm` instantiates `LLMOwner` and prints colored stderr UX | skills.py:458-623 |
| 5 | Engine imports `from ...llm import LLMOwner, LLMOptions, ModelSpec` — runtime → llm dependency direction reversed | engine.py:21-22 |
| 6 | `on_assistant_message` raises propagate (vs `_fire`'s isolation) — undocumented | engine.py:1180-1183, 1240-1244, 860-868 |
| 7 | All `LoopCallbacks` typed as plain `Callable` — async callbacks silently no-op | callbacks.py:16-45 |
| 8 | `on_tool_start` fires before cancellation check → can fire without matching `on_tool_end` | engine.py:1251-1253 vs 1256-1267 |
| 9 | `on_assistant_message` not applied to verification (engine.py:1161-1166) or fallback (engine.py:1473-1480) append paths — inconsistent contract |

## Recommended next refactors (ranked by ROI)

1. ✅ **DONE 2026-04-25** (commit `a043259`, branch `fix/runtime-finalize-funnel`) — `_finalize_run()` helper added, all 9 `_run_impl` exits funneled through it. Fixes Part 2 #6 (cost_limit), #7 (repeated truncation), #12 (cooperative cancel) + missing trace.finish on truncation-error and fallback-success paths + Part 3 #8 (`on_tool_end` on cancellation-during-tool-batch). Static invariant test `test_no_bare_LoopResult_returns_in_run_impl` prevents regression. 491 passed, 1 skipped.
2. **Snapshot tool metadata atomically per batch** (Part 1 #4, #11, #12). Extend `registry.snapshot()` to return a frozen `ToolBatch` namedtuple containing `(functions, schemas, parallel_safe_set, timeout_hints, version)`. Engine consumes that snapshot for the whole step. Add a test that mutates registry mid-batch and asserts dispatch sees consistent metadata.
3. **Lift `SkillManager` out of `runtime/`** into `jyagent/skills/` (or `jyagent/app/skills/`). Make it depend on `runtime`, not the other way around. Engine never needed it. This also fixes the runtime → llm dependency direction issue when paired with extracting an `LLMClient` Protocol that runtime defines and providers implement.

## What the runtime got right (per Codex)

- Daemon-thread design correctly avoids the pool-starvation failure mode the previous pool-based timeout had (engine.py:584-594 docstring is accurate about that win).
- `os.replace` for checkpoint atomicity is the right base layer (just missing fsync for true crash durability).
- Closure-factory design for `write_todos` (avoiding ContextVar) — rationale at todos.py + test_todos_scratchpad.py:224-238 is sound.
- `_strip_dangling_verification()` exists and is called from KeyboardInterrupt + max_steps + outer `Exception` paths — the design is right; it just needs the 3 missing exit paths.
- Tests cover the historical pool-starvation regression (test_loop_engine_p0_fixes.py:933+ daemon-thread spawn check).

## Raw codex outputs

- `/tmp/codex_review_part1.md` — concurrency & state
- `/tmp/codex_review_part2.md` — state machine
- `/tmp/codex_review_part3.md` — API & coupling
- `/tmp/jyagent_bg_z__wwkq2.out`, `d2zgs4xn.out`, `7o5p_24l.out` — full Codex sessions with all probing
