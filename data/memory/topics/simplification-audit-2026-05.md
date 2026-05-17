---
created: 2026-05-15T08:54:00+08:00
updated: 2026-05-17T01:38:39+08:00
status: SUPERSEDED — see data/memory/journal/2026-05.md (2026-05-17 entry)
---
# Over-Design Audit (2026-05-15, jointly with Codex) — SUPERSEDED 2026-05-17

**This audit is now historical.** All recommendations were either shipped,
re-validated and dropped, or rendered invalid by codebase evolution between
2026-05-15 and 2026-05-17. See the 2026-05-17 journal entry for the
re-validation outcome and what actually landed.

Key methodology lesson preserved here: **design-audit plans go stale within
days.** Re-validate item-by-item before executing; treat the topic file as a
hypothesis list, not an execution plan.

## Final outcome (after 2026-05-17 re-validation)

| Original item | Final disposition |
|---|---|
| 1.1 Delete PEP-562 lazy loaders | ✅ shipped between 2026-05-15 and 2026-05-17 |
| 1.2 Rename `_cancel_event`/`_callbacks` (private attrs) | ✅ shipped (idem) |
| 1.3 Shrink LLMClient docstring | ✅ shipped (idem) |
| 2.1 Delete `RunContext` Protocol | ✅ shipped (idem) |
| 2.2 Inline `cost.py` / `stuck_loop.py` | ❌ DROP — each has independent tests; the split IS a test boundary |
| 2.2 Inline `_thread_helpers.py` | ❌ DROP — 2 production callers (engine + llm_runner) |
| 2.2 Inline `phases.py` | ❌ DROP — public PhasePolicy injection surface |
| 2.3 Delete 6 "dead" LoopConfig flags | ❌ DROP — all 7 flags are now actively consumed |
| 2.4 Move `run_step` back to `AgentLoop._run_step` | ❌ DROP — would create a 1700+ LOC class |
| 3.1 Inline `_paths.py` | ❌ DROP — 4 production callers |
| 3.1 Inline `_markdown.py` | ❌ DROP — 2 production callers |
| 3.1 Inline `_consolidation.py` | ✅ shipped 2026-05-17 (commit 1a7723a) |
| 3.1 Inline `_extraction_directives.py` | ❌ DROP — 2 production callers |
| 3.1 Inline `_extraction_security.py` | ✅ shipped 2026-05-17 (commit 29dca5f) |
| 3.2 Trim memory/__init__.py re-exports | ✅ shipped 2026-05-17 (commit 3726ca5) |
| 3.3 Move `_safe_checkpoint` → own module | ✅ shipped (now in `jyagent/durability.py`) |
| 3.4 Defer `skills.py` split | ✅ confirmed in 2026-05-16 audit (FINE-AS-IS) |

## Original verdict table (preserved as historical record)

| # | Hot-spot | Original verdict | Original action |
|---|---|---|---|
| 1 | PEP-562 lazy loaders in `runtime/__init__.py` + `runtime/loop/__init__.py` | over-designed | Delete `_LAZY_ATTRS`, `__getattr__`, `__dir__`, eagerly import |
| 2 | `LLMClient` Protocol 200-line docstring | borderline | Keep tiny Protocol; move contract to types.py / docs |
| 3 | `RunContext` Protocol in step.py | over-designed | Delete; annotate as `AgentLoop` under TYPE_CHECKING |
| 4 | step.py / llm_runner.py / engine.py 3-way split | over-designed | Move `run_step` back to `AgentLoop._run_step` |
| 5 | 23-module `runtime/loop/` | over-designed | Inline `cost.py`, `stuck_loop.py`, `_thread_helpers.py`, `phases.py` |
| 6 | 17-file `memory/` with 8 private `_*.py` | borderline-fragmented | Merge `_paths`, `_markdown`, `_consolidation`, `_extraction_*` into callers |
| 7 | `LoopThreadHelper` string-attr mixin | over-designed | Rename and drop string-attr indirection |
| 8 | `LoopConfig` 25-field accretion | over-designed | Delete unused flags |
| 9 | Top-level fuzzy boundaries | borderline | Move `_safe_checkpoint` out of `agent_commands.py`; split `skills.py` only if hurts |

## Why the plan went stale

Between 2026-05-15 and 2026-05-17 (~2 days), the codebase grew consumers
for nearly every "dead" item flagged. The runtime/loop modules in
particular accumulated real users for flags and helpers the prior audit
called speculative. **The lesson is not "the audit was wrong" — it's
"design audits are perishable."** A 2-day-old plan needs full
re-validation before execution.

## Process rules (still valid)

1. Run `pytest -x` baseline before each phase; don't proceed if red.
2. One sub-step per commit (each step independently revertable).
3. Run codex review on each phase's diff before declaring done.
4. Test changes are part of the same step (don't leave broken tests).
5. **NEW: re-validate every pending item against the current tree before
   executing — don't trust an audit plan more than 24-48 hours old.**
