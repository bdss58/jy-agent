---
created: 2026-04-25T16:09:03+08:00
updated: 2026-04-25T16:28:48+08:00
---
# Runtime Package Refactor

**Branch**: `refactor/runtime-package` (worktree at `../jy-agent-runtime-refactor`)
**Status**: Phases 1-4 + 6 COMPLETE & verified. Phase 5 (Codex review) partially done — Codex died mid-summary on network errors **both** times after substantial useful probing (extracted findings below).

## Result

| Metric | Value |
|---|---|
| Commits on branch | 4 (`f4674e0`, `02e1147`, `90538be`, `9d4ce40` — last is README) |
| Files moved (`git mv`, history preserved) | 13 |
| New files in `runtime/` | 5 (`__init__.py` × 3, `loop/callbacks.py`, `loop/config.py`) |
| Backward-compat shims at old paths | 13 |
| Tests | 486 passed, 1 skipped (no test files modified) |
| Wall time | ~12 min (Claude Code in background) |

## Final layout (matches plan exactly)
```
jyagent/runtime/
├── __init__.py              # AgentLoop, LoopConfig, LoopResult, LoopCallbacks, get_registry, get_stats, ToolResult, SessionStats
├── loop/
│   ├── __init__.py
│   ├── engine.py            # AgentLoop + dispatch (was loop_engine.py)
│   ├── callbacks.py         # LoopCallbacks protocol
│   ├── config.py            # LoopConfig, LoopResult dataclasses
│   ├── phases.py, reflection.py, checkpoint.py, todos.py
│   └── verification.py, remediation.py, tracing.py
├── tools/
│   ├── __init__.py          # get_registry, ToolResult
│   ├── registry.py
│   ├── result.py            # (was toolresult.py)
│   └── validation.py
├── stats.py                 # (was session_stats.py)
└── skills.py
```

## Verification (all ✅, both Claude Code + my independent run + Codex probes)
- Public-API import: `from jyagent.runtime import AgentLoop, LoopConfig, LoopResult, LoopCallbacks` works
- Class identity: `jyagent.loop_engine.AgentLoop is jyagent.runtime.loop.engine.AgentLoop` (same for LoopConfig × 4 paths, LoopResult × 4, LoopCallbacks, ToolResult, SkillManager)
- Singleton identity: `jyagent.registry.get_registry() is jyagent.runtime.tools.registry.get_registry()` ✓ (same for `get_stats()`)
- DeprecationWarning fires once per import (verified via `warnings.catch_warnings`)
- `import jyagent.agent` succeeds end-to-end
- Full pytest suite green

## ⚠️ Documented gotcha — shim asymmetry on module-level constants

**Discovered by Codex.** Shims do `globals().update(__dict__)` once at import time, so:
- Classes & singletons stay identical (good — `old.X is new.X` because dict update copies the **same object reference**)
- Module-level **mutable constants** (e.g. `VERIFICATION_ENABLED`, `TRACES_DIR`, `TRACE_ENABLED`) get a **value snapshot** in the shim namespace. Functions defined in the new module read from `new.__dict__`, NOT `old.__dict__`.

**Concrete example** (Codex probe output):
```
new.VERIFICATION_ENABLED = False
old.VERIFICATION_ENABLED = True
old.should_verify(msg, 1)  →  False   # because should_verify reads new.VERIFICATION_ENABLED
```

**Impact**:
- ✓ No tests rely on this pattern today (full suite green)
- ✗ Future test code using `monkeypatch.setattr('jyagent.verification', 'VERIFICATION_ENABLED', True)` would silently no-op
- **Mitigation**: tests should target the new path (`jyagent.runtime.loop.verification`). Document in CONTRIBUTING / README.

## Remaining work
- Phase 5b: Re-run Codex review when network is healthier (got 85k tokens of useful probes before dying both times) — or skip; findings already documented
- Open PR / merge to main
- (Future) Remove shims one release after merge

## Out of scope (deferred — explicitly NOT done)
- Splitting `skills.py` into a subpackage (still 771-line single file in runtime/)
- Moving `mcp_*` into `runtime/`
- Moving `agent.py` into `jyagent/app/`
- Cleaning up the runtime → cli/console dependency in skills.py (see Codex Q5 — flagged but not investigated due to mid-summary disconnect)
