---
created: 2026-05-15T08:54:00+08:00
updated: 2026-05-15T08:54:00+08:00
---
# Over-Design Audit (2026-05-15, jointly with Codex)

User asked for source-tree audit and simplification. User pre-accepts all recommendations.

## Verdicts (Codex + me, agree)

| # | Hot-spot | Verdict | Action |
|---|---|---|---|
| 1 | PEP-562 lazy loaders in `runtime/__init__.py` + `runtime/loop/__init__.py` | over-designed | Delete `_LAZY_ATTRS`, `__getattr__`, `__dir__`, eagerly import |
| 2 | `LLMClient` Protocol 200-line docstring | borderline | Keep tiny Protocol; move contract to types.py / docs |
| 3 | `RunContext` Protocol in step.py | over-designed | Delete; annotate as `AgentLoop` under TYPE_CHECKING, or move `run_step` back to method |
| 4 | step.py / llm_runner.py / engine.py 3-way split | over-designed | Move `run_step` back to `AgentLoop._run_step`; keep `LLMRunner` only if unit-tested |
| 5 | 23-module `runtime/loop/` | over-designed | Inline `cost.py`, `stuck_loop.py`, `_thread_helpers.py`, `phases.py` |
| 6 | 17-file `memory/` with 8 private `_*.py` | borderline-fragmented | Merge `_paths`, `_markdown`, `_consolidation`, `_extraction_*` into callers |
| 7 | `LoopThreadHelper` string-attr mixin | over-designed | Rename LLMRunner.cancel_event→_cancel_event, callbacks→_callbacks, then plain mixin or inline 30 LOC |
| 8 | `LoopConfig` 25-field accretion | over-designed | Delete unused flags: `phase_policy`, `reflect_*`, `checkpoint_*`, `todos_enabled`, `buffered_streaming` if dead in CLI |
| 9 | Top-level fuzzy boundaries | borderline | Move `_safe_checkpoint` out of `agent_commands.py`; split `skills.py` only if hurts |

## Execution Plan

### Phase 1 — Cheap de-enterprise (low risk)
- 1.1 Delete PEP-562 lazy loaders → eager imports
- 1.2 Rename `LLMRunner.cancel_event`/`callbacks` to `_cancel_event`/`_callbacks`, drop string-attr indirection in `LoopThreadHelper`
- 1.3 Shrink `LLMClient` docstring to ~30 lines
- 1.4 Update affected tests (import-laziness assertions)

### Phase 2 — Runtime loop consolidation (medium risk)
- 2.1 Delete `RunContext`, annotate `run_step` helpers with `AgentLoop`
- 2.2 Inline `cost.py`, `stuck_loop.py`, `_thread_helpers.py`, `phases.py` into engine.py
- 2.3 Audit + delete dead `LoopConfig` flags
- 2.4 Optionally re-merge `run_step` → `AgentLoop._run_step`

### Phase 3 — Memory + top-level (medium risk)
- 3.1 Merge `memory/_paths.py`, `_markdown.py`, `_consolidation.py`, `_extraction_directives.py`, `_extraction_security.py` into their callers
- 3.2 Trim `memory/__init__.py` re-exports to public API only
- 3.3 Move `_safe_checkpoint` from `agent_commands.py` → `agent.py` (or a `runtime/durability.py`)
- 3.4 Defer `skills.py` split unless editing pain

## Process Rules

1. Run `pytest -x` baseline before each phase; don't proceed if red.
2. One sub-step per commit (each step independently revertable).
3. Run codex review on each phase's diff before declaring done.
4. Test changes are part of the same step (don't leave broken tests).
