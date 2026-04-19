# Agent Memory

This is the memory index. Keep it concise.
Detailed knowledge lives in topic files under `data/memory/topics/`.

## User Profile
- Name: Jianyong
- Role: infrastructure/DevOps engineer
- OS: macOS
- Language: Chinese (Mandarin), bilingual with English
- Timezone: Asia/Shanghai
- Communication style: direct, holds agents accountable, often speaks Chinese
- Tech stack: Python, Docker, Kubernetes, Bash, FastAPI, Chrome DevTools/MCP, MCP, Anthropic API, OpenAI API, AWS Bedrock, vllm, OpenClaw, GitLab

## Behavioral Rules (CRITICAL)
- Never fabricate command results or claim actions were done without executing them
- Verify with tools (`pwd`, `find`, `date`, etc.) before answering filesystem or environment questions
- Use live search/browser tools for current information; do not rely on stale model memory
- Verify current date with `date` before time-sensitive research
- Keep `MEMORY.md` concise and move detailed notes into topic files
- For self-upgrades to jy-agent runtime/source code, use `git worktree` to avoid disrupting the running agent
- **Codex as second opinion**: On significant tasks (code review, bug investigation, architecture planning, research, non-trivial analysis), proactively run Codex for a complementary perspective and synthesize both viewpoints before presenting results. Use `codex review` for code reviews, `codex exec --sandbox read-only` for analysis/planning, `web_search(engine="codex")` for deep research. Skip Codex for trivial tasks (typos, simple lookups, quick edits).

## User Preferences
- Dual output: keep raw streaming + rendered markdown panel (not a bug)
- Chrome MCP: use independent instance, not user's Chrome
- MCP as primary tool integration mechanism
- Prefers robust solutions (CLI args, config) over fragile source patches
- When delegating to Claude Code, prefer the locally configured default model; only pass `--model` for an intentional tier override

## Environment
- macOS, username: jyxc-dz-0100398
- Project dir: `/Users/jyxc-dz-0100398/jy-agent` (package `jyagent/`)
- Python 3.14 `.venv` has broken CA certs; HTTP clients often need `verify=False` fallback
- Dependency lockfile exists at `uv.lock`

## Topic Files Index
- **nano-vllm-learning.md** — Long-term plan to master LLM inference via nano-vLLM. Tracks current phase, session log, checkpoints, questions. Read this file on any session mentioning nano-vLLM / learning / LLM study.

## Repo Snapshot
- Runtime is provider-neutral: Anthropic Messages + OpenAI Responses adapters under `jyagent/runtime/`
- `RuntimeOwner` owns the active `provider:model`; `/model <provider> <model>` switches future turns
- Core native tools live in `jyagent/tools/` and register with per-tool metadata in `jyagent/tools/__init__.py`
- Session stats track provider/model, cache tokens, subagent usage, and cost; unknown OpenAI snapshots can fetch pricing from docs
- CLI history rendering avoids Rich markup parsing on dynamic text and formats normalized assistant/tool blocks safely
- Silent completions now reuse the streaming runtime path, including subagents

[gotcha] `contextvars.ContextVar` is NOT auto-propagated by `ThreadPoolExecutor.submit()` (verified Python 3.14.3 — worker sees default, not caller's value). Must use `ctx = contextvars.copy_context(); executor.submit(ctx.run, fn, ...)` to explicitly propagate.
[preference] Don't create memory topic files for obvious/training knowledge already covered by concise behavioral rules
[gotcha] Codex CLI `--sandbox` flag (read-only / workspace-write / danger-full-access) only restricts **filesystem access** for model-generated shell commands. It does NOT restrict network access, web search, or any other capability. Never blame sandbox mode for web search issues.
[goal] Long-term learning plan: Guide user through nano-vLLM to master LLM inference. Progress tracked in `data/memory/topics/nano-vllm-learning.md`. On any new session mentioning nano-vLLM, learning, or LLM study — read that file first to resume where we left off.
[gotcha] User launches jy-agent from home dir (~), NOT from the project root. All file paths in memory are relative to project root `/Users/jyxc-dz-0100398/jy-agent/`. Always prefix with the project root when using read_file, glob_files, find, etc. Never use bare relative paths like `data/memory/...` — they resolve to the wrong directory.
[tip] jy-agent's run_background/check_background has been hardened (Tier 1 bugs + Tier 2 ergonomics). Status taxonomy: running|succeeded|failed|killed|timed_out. Key features: `timeout_seconds` auto-kills deadlines (status becomes 'timed_out'); `cwd` + `stdin_null=True` by default; `action="wait"` blocks up to 300s (saves turns vs polling); global concurrency cap = 8 live jobs. Output bounded to ~50 KB (seek-from-end); 256 KB backward scan cap prevents single-line OOM. Lifecycle tested in tests/test_background.py (26 tests).
[tip] Daemon threads don't propagate ContextVar after spawn; use per-loop tool closures instead of ContextVar for tool state.
[tip] Mutating Anthropic system_prompt breaks prompt caching; inject dynamic context as a non-persisted tail message block instead.
[user_stated] AWS Bedrock account lacks access to claude-opus-4-7 (permission_error on InvokeModelWithResponseStream).
[note] 2026-04-18 — Agent-loop upgrade FULLY COMPLETE: 13 P0 correctness bugs + 5 P1 capability items shipped across 7 batches. P0 batches 1-5: nested-pool deadlock, cost_tracker effective_spec, fallback_on_max_steps, cancellable retry sleep, stream-loop cancel check, stuck detector raw content, per-batch dedup, verification gate boundary, retry jitter, on_stream_retry + buffered_streaming, compaction preserves thinking/tool_use adjacency, daemon-thread tool timeouts. P1 batches 6-7: (1) Persistent TODO scratchpad (jyagent/todos.py — write_todos tool via closure-scoped factory, replace-all semantics, renders as <system-reminder> block appended to tail user msg for Anthropic prefix-cache preservation; initial_todos on run(), LoopResult.todos for persistence); (2) Reflection / critic step (jyagent/reflection.py — every-N-tool-calls + after-subagent triggers, guards against back-to-back injection, on_reflection callback); (3) Phase-aware tool_choice shaping (jyagent/phases.py — PhaseDirective + default_phase_policy(plan/verify/finalize), on_phase_enter callback); (4) Checkpointed replay (jyagent/checkpoint.py — LoopCheckpoint dataclass, atomic .tmp+rename save, periodic step_NNNN.json + terminal final.json, on_checkpoint callback); (5) Structured sub-agent envelope (_format_subagent_envelope in tools/subagent.py — Markdown "## Sub-agent Result" with Status/Stats/Response sections, JY_SUBAGENT_FLAT_RESULT=1 opt-out). Tests: 6 new test files totaling 94 new regression tests (24 P0 + 29 todos + 17 reflection + 16 phases + 21 checkpoint + 11 envelope) plus 286 pre-existing = 404/404 full suite green. Files: loop_engine.py net +608/-40, subagent.py +67; 4 new modules (todos/reflection/phases/checkpoint) + 6 new test files. Codex design review of TODO scratchpad corrected two initial design errors (ContextVar vs closure; system-prompt mutation vs tail-message injection) saving a refactor.
[tip] Warp.dev deprecated `warp-cli` in favor of `oz` CLI (auto-updates); Cloudflare WARP's `warp-cli` is unrelated and still active.
[correction] User corrected assistant's assumption that "warp-cli" meant Cloudflare WARP; context was Warp.dev terminal/Oz platform.
