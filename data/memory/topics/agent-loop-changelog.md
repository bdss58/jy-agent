# Agent Loop & Background Runner — Changelog and Internals

Detailed reference for the loop_engine + background tooling work shipped 2026-04-18.
**Index pointer in MEMORY.md is one line; this file holds the detail.**

## 2026-04-18 — Agent-loop upgrade FULLY COMPLETE

13 P0 correctness bugs + 5 P1 capability items shipped across 7 batches.

### P0 batches 1–5 (correctness)
- nested-pool deadlock
- `cost_tracker` effective_spec
- `fallback_on_max_steps`
- cancellable retry sleep
- stream-loop cancel check
- stuck detector raw content
- per-batch dedup
- verification gate boundary
- retry jitter
- `on_stream_retry` + `buffered_streaming`
- compaction preserves thinking/tool_use adjacency
- daemon-thread tool timeouts

### P1 batches 6–7 (capabilities)
1. **Persistent TODO scratchpad** — `jyagent/todos.py`
   - `write_todos` tool via closure-scoped factory; replace-all semantics
   - Renders as `<system-reminder>` block appended to tail user message
     (Anthropic prefix-cache preservation — never mutates the system prompt)
   - `initial_todos` on `run()`, `LoopResult.todos` for persistence
2. **Reflection / critic step** — `jyagent/reflection.py`
   - Triggers: every-N-tool-calls + after-subagent
   - Guards against back-to-back injection
   - `on_reflection` callback
3. **Phase-aware tool_choice shaping** — `jyagent/phases.py`
   - `PhaseDirective` + `default_phase_policy(plan/verify/finalize)`
   - `on_phase_enter` callback
4. **Checkpointed replay** — `jyagent/checkpoint.py`
   - `LoopCheckpoint` dataclass; atomic `.tmp + rename` save
   - Periodic `step_NNNN.json` + terminal `final.json`
   - `on_checkpoint` callback
5. **Structured sub-agent envelope** — `_format_subagent_envelope` in `tools/subagent.py`
   - Markdown "## Sub-agent Result" with Status / Stats / Response sections
   - `JY_SUBAGENT_FLAT_RESULT=1` opt-out

### Tests
6 new test files — 94 new regression tests:
24 P0 + 29 todos + 17 reflection + 16 phases + 21 checkpoint + 11 envelope.
Plus 286 pre-existing → **404/404** full suite green.

### File deltas
- `loop_engine.py`: net +608 / −40
- `subagent.py`: +67
- 4 new modules (todos / reflection / phases / checkpoint)
- 6 new test files

### Lessons captured
Codex design review of the TODO scratchpad corrected two initial design errors,
saving a refactor:
1. **ContextVar vs closure** — daemon threads don't propagate `ContextVar`;
   per-loop closures are correct.
2. **System-prompt mutation vs tail-message injection** — mutating the system
   prompt invalidates the Anthropic prefix cache. Inject as
   `<system-reminder>` on the tail user message instead.
   *(This rule is also kept in MEMORY.md as a 1-liner — important enough.)*

---

## `run_background` / `check_background` hardening

Shipped as part of Tier 1 bug fixes + Tier 2 ergonomics. Lifecycle covered by
`tests/test_background.py` (26 tests).

### Status taxonomy
`running | succeeded | failed | killed | timed_out`

### Key features
- `timeout_seconds` auto-kills past deadline; status becomes `timed_out`
- `cwd` parameter (preferred over `cd X && cmd`)
- `stdin_null=True` by default — children can't hang on prompts
- `action="wait"` blocks up to 300s — saves turns vs polling
- Global concurrency cap = **8 live jobs**

### Output bounding
- ~50 KB tail (seek-from-end) by default
- 256 KB backward scan cap prevents single-line OOM
