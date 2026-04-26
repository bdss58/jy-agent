---
created: 2026-04-25T18:20:11+08:00
source: codex-cli (3 sequential `codex exec --sandbox read-only` runs)
scope: jyagent/runtime/* (engine, loop/*, tools/*, stats, skills) — ~3400 LOC
updated: 2026-04-25T22:55:36+08:00
status: 15 of 34 findings CLOSED (5 shipped 2026-04-25 evening session)
---
# Runtime Design + Implementation Review (Codex)

## Method

Single broad review died at 103k tokens (network errors mid-summary — same pattern documented in `runtime-refactor.md`). Switched to 3 narrow scoped runs with explicit "synthesize as you go" directive. Two completed cleanly; part 3 died before final TOP 3 RISKS but had already emitted all 9 findings. Total: **34 findings, all with file:line refs**.

A second-opinion run on 2026-04-25 evening (during the followup-shipping session) also network-died at 71k tokens — third instance of this pattern. Got one early observation that validated fix #5 plan: *"LLMClient is a clean behavioral extraction at the type level, but the engine still imports concrete value types from jyagent.llm."*

## Status snapshot (2026-04-25 evening)

**CLOSED via earlier refactors:**
- LLMClient Protocol extraction (`4046792`) → Part 3 #5 first half
- SkillManager lift-out of runtime (`c8a0cc0`) → Part 3 #3, #4 + risk #5
- ToolBatch atomic per-step snapshot (`6c453fe`) → Part 1 #4, #11, #12
- _finalize_run consolidation (`a043259`) → Part 2 #6, #7, #12

**CLOSED via 2026-04-25 evening session (5 commits, branch `fix/runtime-review-followups`):**
- `80315bb` checkpoint fsync → Part 2 #3
- `662fde1` _MODEL_PRICING RLock → Part 1 #6
- `a733f8c` verification.should_verify(since_index=) → Part 2 #5
- `3044a57` LLMOptions/ModelSpec move into runtime → Part 3 #5 second half (runtime now llm-clean, pinned by subprocess test)
- `f360cda` cost unification (compute_call_cost) → Part 1 #9, #10

**STILL OPEN (lower-priority / acknowledged trade-offs):**
- Part 1 #1, #2, #3 — daemon-thread tool-body design (documented trade-off; Python threads aren't cancellable)
- Part 1 #5 — SessionStats.provider/model property reads bypass lock (benign in CPython for str reads)
- Part 1 #7, #8 — ToolResult string-content typing (cosmetic)
- Part 1 #11 — ToolRegistry's live-read accessors retained as footguns; should @deprecated
- Part 2 #1 — phase policy collapse for max_steps≤2
- Part 2 #4 — write_todos no merge guard (acknowledged)
- Part 2 #8 — checkpoint replay state loss (v1 "recovery semantics" by design)
- Part 2 #9 — on_assistant_message tool_call_id pairing (caller responsibility)
- Part 2 #10 — max-step fallback tool_choice="none" provider-dependence
- Part 3 #1 — runtime/__init__.py eagerly loads engine (needs PEP-562 __getattr__)
- Part 3 #2 — module-level _tool_dispatch_executor + atexit.register
- My fresh observations (1) engine.py 1898 LOC (defer until next feature), (8) skills.py 28KB single file

## Top architectural risks (consolidated across all 3 runs)

1. ~~Daemon-thread tool timeout leaks state~~ → still open, acknowledged trade-off
2. ~~`[VERIFICATION]` cleanup is incomplete~~ → CLOSED by `a043259` (_finalize_run funnel)
3. ~~Registry/stats reads bypass write locks~~ → mostly CLOSED (ToolBatch + pricing lock); minor SessionStats props open
4. ~~`runtime` is not UI-/LLM-clean~~ → CLOSED (LLMClient Protocol + LLMOptions/ModelSpec move + SkillManager lift-out)
5. ~~`SkillManager` doesn't belong in `runtime/`~~ → CLOSED by `c8a0cc0`
6. **`on_assistant_message` callback can break tool_use/tool_result pairing** — runs at engine.py:1240 before tool results appended for original tool-call IDs, no invariant check after. Also not applied to verification or fallback append paths (inconsistent contract).
7. ~~Cost budget drifts under concurrency~~ → unification CLOSES the within-process drift; cross-process (concurrent LLM-backed tools) still open

## Part 1 — Concurrency & State (engine.py 700-1200, stats, tools/*) — 12 findings

(See `/tmp/codex_review_part1.md` for full text. Key file:line refs:)

| # | Issue | File:line | Status |
|---|---|---|---|
| 1 | Daemon tool body continues after timeout return | engine.py:621, 638-650 | OPEN (trade-off) |
| 2 | `body_permits` released while daemon still running → `max_tool_workers` cap defeated | engine.py:629, 638-641 | OPEN (trade-off) |
| 3 | No `contextvars.copy_context()` for tool body threads | engine.py:632 | OPEN |
| 4 | Registry metadata/schema reads unlocked, racing with snapshot | registry.py:49; engine.py:466, 993-994 | CLOSED (`6c453fe` ToolBatch) |
| 5 | `SessionStats.provider/model/elapsed` properties bypass `_lock` | stats.py:134 | OPEN (benign) |
| 6 | `set_model_pricing()` unsynchronized with `_lookup_pricing()` iteration | stats.py:85 | CLOSED (`662fde1` RLock) |
| 7 | `ToolResult` doesn't enforce string content (engine assumes it) | result.py:18; engine.py:1283 | OPEN (cosmetic) |
| 8 | Error semantics rely on caller setting `is_error` (no validation) | result.py:18 | OPEN (cosmetic) |
| 9 | `_CostTracker` blind to concurrent tool LLM usage | engine.py:1097 | OPEN (cross-process) |
| 10 | `_CostTracker` reimplements pricing without cache-read / long-context handling | engine.py:104 | CLOSED (`f360cda` unification) |
| 11 | `is_parallel_safe()` is live read, not snapshotted per batch | engine.py:520, 535, 537 | CLOSED (`6c453fe` ToolBatch) |
| 12 | `snapshot()` returns shared schema dicts (mutable post-registration) | registry.py:44 | CLOSED (`6c453fe` deep-copy) |

## Part 2 — State Machine (phases/reflection/verification/remediation/checkpoint/todos) — 13 findings

(See `/tmp/codex_review_part2.md`. Key:)

| # | Issue | File:line | Status |
|---|---|---|---|
| 1 | Default phase policy collapses `verify` into `finalize` for `max_steps≤2` | phases.py:73-79 | OPEN |
| 2 | Verification de-dup is tail-only — TODO/reflection appends can re-arm it | verification.py:89-108 | OPEN (defensive depth) |
| 3 | Checkpoint uses `os.replace` but no `fsync()` on file or parent dir | checkpoint.py:66-73 | CLOSED (`80315bb`) |
| 4 | `write_todos` has no merge guard (replace-all). Mitigated only because it's not registered as parallel-safe — implicit, fragile | todos.py:212-246; engine.py:1001-1004 | OPEN (acknowledged) |
| 5 | `should_verify()` scans entire history for mutations — old replayed `write_file` triggers verify on a non-mutating new run | verification.py:111-135 | CLOSED (`a733f8c` since_index) |
| 6 | Cost-limit exit at engine.py:1117 leaks dangling `[VERIFICATION]` | | CLOSED (`a043259` funnel) |
| 7 | Repeated-truncation exit at engine.py:1203 leaks dangling `[VERIFICATION]` | | CLOSED (`a043259` funnel) |
| 8 | Checkpoint replay loses `last_reflection_count`, `verification_injected`, truncation-retry, stuck-detector state | checkpoint.py:37-44 vs engine.py:916, 921 | OPEN (v1 recovery semantics) |
| 9 | `on_assistant_message` runs at engine.py:1240 before tool-result append → can desync tool_call_id pairing | | OPEN (caller responsibility) |
| 10 | Max-step fallback sends `tool_choice={"type":"none"}` after deleting `tools` — provider may reject; exception swallowed at engine.py:1495-1497 | | OPEN |
| 11 | Tool timeout breaks "non-parallel-safe = sequential barrier" invariant for mutations | engine.py:497-500 vs 632-649 | OPEN (trade-off) |
| 12 | Cooperative cancellation path at engine.py:1404-1418 doesn't strip dangling verification | | CLOSED (`a043259` funnel) |
| 13 | Checkpoint load has no replay contract for TODO restoration | | OPEN (v1 design) |

## Part 3 — API Surface & Coupling (runtime/__init__, callbacks, config, tracing, skills) — 9 findings

(See `/tmp/codex_review_part3.md`. Key:)

| # | Issue | File:line | Status |
|---|---|---|---|
| 1 | `import jyagent.runtime` eagerly loads engine — contradicts `config.py:4-5` "no engine import cost" promise | runtime/__init__.py:2 → loop/__init__.py:2 → engine.py | OPEN (needs PEP-562) |
| 2 | Module-level `ThreadPoolExecutor` + `atexit.register` fires on plain import | engine.py:71-74 | OPEN (cosmetic) |
| 3 | `SkillManager` is not an engine dependency — engine never imports it | skills.py:260-768 vs engine.py:759-906 | CLOSED (`c8a0cc0` lift-out) |
| 4 | `SkillManager._route_llm` instantiates `LLMOwner` and prints colored stderr UX | skills.py:458-623 | CLOSED (`c8a0cc0`) |
| 5 | Engine imports `from ...llm import LLMOwner, LLMOptions, ModelSpec` — runtime → llm dependency direction reversed | engine.py:21-22 | CLOSED (`4046792` Protocol + `3044a57` value-type move) |
| 6 | `on_assistant_message` raises propagate (vs `_fire`'s isolation) — undocumented | engine.py:1180-1183, 1240-1244, 860-868 | OPEN (cosmetic) |
| 7 | All `LoopCallbacks` typed as plain `Callable` (no Protocol) | callbacks.py:* | OPEN (style) |
| 8 | `tracing.py` writes to a fixed `~/.jyagent/traces` path — not configurable | tracing.py:* | OPEN |
| 9 | Recommended next refactors (omitted before death) | | N/A |

## What got SHIPPED in 2026-04-25 evening session

5 commits, +764/-96 LOC across 13 files, branch `fix/runtime-review-followups`, fast-forward merged to main. 13 new regression tests (final suite 490 passed, was 477).

| Commit | Closes | Test added |
|---|---|---|
| `80315bb` | Part 2 #3 | `test_save_fsyncs_for_crash_durability` (os.fsync spy) |
| `662fde1` | Part 1 #6 | `test_stats_concurrency.py` (no-deadlock + visibility) |
| `a733f8c` | Part 2 #5 | 2× since_index tests (excludes prior, includes current) |
| `3044a57` | Part 3 #5 | `test_runtime_llm_independence.py` (subprocess-isolated import-leak check) |
| `f360cda` | Part 1 #9, #10 | `test_cost_unification.py` 6 tests (incl. long-context multiplier) |

## Lessons banked

- **Subprocess-isolate any in-test `sys.modules` surgery.** An in-process `del sys.modules[key]` for jyagent.* modules broke 23 unrelated later tests (cached top-level `from X import Y` references no longer pointed at the patched module). Use `subprocess.run([sys.executable, "-c", ...])` for import-cleanliness checks.
- **CPython 3.14 GIL hides many concurrency bugs.** The `_MODEL_PRICING` race was nearly unreproducible; lock added defensively for PEP-703 / many-writer scenarios.
- **Worktree editable-venv gotcha** still bites — needed `uv sync --all-extras` (not just `uv sync`) to get pytest in the worktree's venv.

## Things the runtime got right (per Codex)

- ToolBatch design — clean immutable snapshot per step, deep-copied schemas.
- _finalize_run() funnel — one canonical exit; tested by static-source assertion that no other LoopResult return exists in _run_impl.
- Checkpoint atomicity is the right base layer (now with fsync).
- Closure-factory design for write_todos (avoiding ContextVar) — rationale at todos.py + test_todos_scratchpad.py:224-238 is sound.
- _strip_dangling_verification() exists and is called from KeyboardInterrupt + max_steps + outer Exception paths — design is right.
- Tests cover the historical pool-starvation regression (test_loop_engine_p0_fixes.py:933+ daemon-thread spawn check).

## Raw codex outputs

- `/tmp/codex_review_part1.md` — concurrency & state
- `/tmp/codex_review_part2.md` — state machine
- `/tmp/codex_review_part3.md` — API & coupling
- `/tmp/jyagent_bg_p516jq3m.out` — 2026-04-25 evening second-opinion (network-died at 71k tokens, no synthesis emitted)
