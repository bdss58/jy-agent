---
created: 2026-04-28T13:21:06+08:00
updated: 2026-04-28T13:21:06+08:00
---
# Deprecation Tracker

Tracking deprecated items in the jyagent source tree, with concrete removal criteria
and target dates. Update on every deprecation add/remove.

Lifecycle policy: mark with `@deprecated` + warning → migrate internal callers →
hold ≥1 month or 1 minor-version boundary → remove (and delete the
warning-verification tests in the same commit).

## Active deprecations

### 1. `_LegacyClientRuntimeOwner` (Anthropic-client compat shim)
- **File:** `jyagent/tools/subagent.py` L86 onwards
- **Marked:** unclear — comment says "exists only to keep existing tests working during the migration"
- **Replacement:** `LLMOwner` + mock provider adapters
- **Reason:** tests should not inject raw Anthropic SDK clients — bypasses the provider abstraction
- **Internal callers:** ⚠ TBD — audit `tests/` for `_get_client` monkeypatching / fake Anthropic client injection
- **Target removal:** **after test audit + migration** (no calendar deadline; criterion-driven)
- **Removal checklist:**
  - [ ] `grep -r "_get_client\|_LegacyClientRuntimeOwner\|fake.*anthropic" tests/` to find live consumers
  - [ ] Migrate each test to use `LLMOwner` with a mock adapter
  - [ ] Delete `_LegacyClientRuntimeOwner` class and `_get_client` helper

### 2. `manage_memory` action `'note'` (alias for `'journal'`)
- **File:** `jyagent/tools/facades.py` L8, `jyagent/tools/schemas.py` L107 (description strings); aliasing logic in `jyagent/memory/operations.py`
- **Marked:** in description strings only — no `@deprecated` decorator (aliasing is implicit)
- **Replacement:** action `'journal'`
- **Reason:** `'note'` was the original name; `'journal'` better reflects the three-tier memory model
- **Cost of keeping:** ~1 line of routing code; negligible
- **Target removal:** **low priority — keep indefinitely** unless a future tool-schema cleanup pass wants the simplification. Remove only if a deprecation warning is added first and observed unused for ≥1 month.

## Removed (audit log)

### `ToolRegistry.snapshot()` / `.is_parallel_safe()` / `.is_mutating()` / `.get_timeout_hint()` / `.get_large_input_keys()` / `.get_compaction_priority()`
- **Marked deprecated:** 2026-04-27 (Codex review 2026-04-25 Part 1 #11/#12)
- **Removed:** 2026-04-28 (per user decision; original target was 2026-06-01)
- **Why early:** zero internal callers (already on `ToolBatch.freeze()`); only callers were meta-tests asserting the warning fires; jy-agent is not a published library so the "external/long-lived branch" grace argument did not apply.
- **Removed in same commit:** the 6 deprecation-verification tests in `tests/test_codex_review_fixes.py` (`TestC4ToolRegistryLiveReadDeprecation` class) and the `test_no_in_tree_callers_remain` static-grep guardrail
- **Replacement:** `ToolRegistry.freeze()` → `ToolBatch.is_parallel_safe(name)` / `.is_mutating(name)` / `.get_timeout_hint(name)` / `.get_large_input_keys(name)` / `.get_compaction_priority(name)`
- **Bonus cleanup:** dropped the now-unused `from typing_extensions import deprecated` import; updated README example from `registry.snapshot()[1:]` to `registry.freeze()`
- **Test result:** 640 passed, 1 skipped (full suite, 37s)
- **Lesson reinforced:** the in-tree static-grep test (`test_no_in_tree_callers_remain`) was useful — confirmed migration was complete before removal. Consider keeping that pattern for future deprecations even after the deprecated API is gone (rename the test to e.g. `test_legacy_methods_not_reintroduced`).

## Notes

- Don't ship "deprecation theater" — removing 2 days after marking defeats the purpose. The warning needs time to surface to long-lived branches and external users.
- Always delete deprecation-verification tests **in the same commit** that removes the deprecated API. Otherwise the tests fail with `AttributeError` instead of `DeprecationWarning`, which is confusing.
- See also: durable gotcha in MEMORY.md — `warnings.deprecated` is 3.13+, must use `typing_extensions.deprecated` while pyproject `requires-python = ">=3.12"`.
