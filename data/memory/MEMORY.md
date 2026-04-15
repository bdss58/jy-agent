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
