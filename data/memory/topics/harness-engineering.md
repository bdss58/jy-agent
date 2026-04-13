---
created: 2026-04-10T12:27:24Z
updated: 2026-04-11T04:20:39Z
---
# Harness Engineering

## Status
- Current maturity: ~3.8/5 (was 3.3/5)
- Plan: originally in `data/harness-improvement-plan.md` (deleted), now tracked here

## Implemented Quick Wins
- **QW-1**: `jyagent/remediation.py` — 13 error pattern→remediation pairs, wired into `_execute_tool()`
- **QW-2**: `LoopConfig.max_cost_usd` + `_CostTracker` — per-turn cost budget, terminates with `status="cost_limit"`
- **QW-3**: `LoopConfig.dedup_threshold` + `_DedupTracker` — identical tool+args called N times → break loop with `status="dedup_break"`
- **QW-4**: `jyagent/tracing.py` — JSONL trace logger (RunTrace/SpanEvent), writes to `data/traces/`, controlled by `AGENT_TRACE_ENABLED=1`. Spans: llm_call, tool_call, compaction, cost_check, dedup_break, verification. All exit paths in loop_engine flush trace.
- **QW-5**: `jyagent/verification.py` — pre-completion verification gate. When model stops calling tools after file mutations (edit_file/write_file/run_shell), injects a self-check user message asking it to verify syntax, correctness, tests, consistency. Controlled by `AGENT_VERIFICATION_ENABLED=1`. Only fires once per run (idempotent via `[VERIFICATION]` marker detection).
- Tests: `tests/test_harness_quick_wins.py` (34 tests), `tests/test_tracing_and_verification.py` (33 tests), all passing.

## Remaining Work
- Phase 3: Loop Control enhancements (progress file, termination rules)
- Phase 4: Sensors & drift detection (quality scoring, behavioral guardrails)
- Tool gateway (least-privilege / capability shaping per task type)
- Evaluator agent (separate model judges output quality — Anthropic GAN-style)
