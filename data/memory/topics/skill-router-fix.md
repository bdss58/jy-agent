# Skill LLM Router — Silent-Breakage Fix (2025-11-21)

**One-line in MEMORY.md.** Detail here.

## Symptom
`jyagent/skills.py::_route_llm` was silently broken: it always fell back to
the keyword router because every LLM call raised an exception that was
swallowed by a blanket `except Exception: return None`.

## Root cause
`complete_text()` in `runtime/core.py` unconditionally injected adaptive
thinking via `get_reasoning_config_for_provider()`. For any Anthropic model
older than Claude 4.6 (Haiku, Sonnet 4.5, Opus 4.5), this triggers a
`ValueError` from `validate_anthropic_reasoning`. The router's blanket
`except` made the failure invisible.

## Fix
1. Added `reasoning=` kwarg to `complete_text()` with a `_UNSET` sentinel.
2. Router now passes `reasoning=None` (the routing call is cheap; no thinking
   needed).
3. The `except` in the router now logs a dim warning instead of silently
   swallowing.

## Regression coverage
`tests/test_skill_router.py` — 11 tests pinning:
- successful LLM routing
- correct fallback to keyword router on real failures
- visible warning instead of silent swallow

## Lesson
Blanket `except Exception: return None` around an LLM call hides exactly the
bugs you most need to see. Always log; even a one-line dim warning would
have surfaced this in seconds.
