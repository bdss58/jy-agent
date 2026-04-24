# Agent Memory

This is the always-loaded index. **Hard cap: 200 lines / 25 KB.**
Detail lives in `data/memory/topics/<name>.md` (curated, on demand).
Session notes live in `data/memory/journal/YYYY-MM.md` (never auto-loaded).

## User Profile
- Name: Jianyong
- Role: infrastructure/DevOps engineer
- OS: macOS, username: jyxc-dz-0100398, timezone Asia/Shanghai
- Language: Chinese (Mandarin), bilingual with English; direct, holds agents accountable
- Stack: Python, Docker, Kubernetes, Bash, FastAPI, Chrome DevTools/MCP, Anthropic / OpenAI / AWS Bedrock APIs, vllm, OpenClaw, GitLab

## Behavioral Rules (CRITICAL)
- Never fabricate command results or claim actions were done without executing them
- Verify with tools (`pwd`, `find`, `date`, …) before answering filesystem or environment questions
- Use live search/browser tools for current information; do not rely on stale model memory
- Verify current date with `date` before time-sensitive research
- Keep MEMORY.md concise (≤200 lines); move detail to topic files; chronological notes go to journal
- For self-upgrades to jy-agent runtime/source, use `git worktree` to avoid disrupting the running agent
- **Codex as second opinion** on significant tasks (code review, bug investigation, planning, research). `codex review` for code reviews, `codex exec --sandbox read-only` for analysis, `web_search(engine="codex")` for deep research. Skip for trivial tasks.

## User Preferences
- Dual output: keep raw streaming + rendered markdown panel (not a bug)
- Chrome MCP: use independent instance, not user's Chrome
- MCP as primary tool integration mechanism
- Prefers robust solutions (CLI args, config) over fragile source patches
- When delegating to Claude Code, prefer the locally configured default model; only pass `--model` for an intentional tier override

## Environment
- Project dir: `/Users/jyxc-dz-0100398/jy-agent` (package `jyagent/`)
- User launches jy-agent from home dir (`~`), NOT project root — always prefix paths with the project root in tool calls
- Python 3.14 `.venv` has broken CA certs; HTTP clients often need `verify=False` fallback
- Dependency lockfile: `uv.lock`

## Topic Files Index
- **memory-design.md** — Three-tier memory architecture (this file's design rationale + routing rules). Read this if asked to refactor or extend the memory system.
- **agent-loop-changelog.md** — Loop engine internals (TODO scratchpad, reflection, phases, checkpoints, sub-agent envelope) + `run_background` hardening detail. Read on questions about loop_engine, todos, reflection, phases, checkpoint, or background tooling.
- **skill-router-fix.md** — Why the skill LLM router was silently broken and the `complete_text(reasoning=...)` fix. Read on questions about skill routing, `_route_llm`, or the `validate_anthropic_reasoning` ValueError.
- **gfw-proxy-fallback.md** — SSH SOCKS5 tunnel workflow for GFW-blocked hosts (ghcr.io, raw.githubusercontent.com, huggingface.co, etc.). Read when a network call fails against a likely-blocked host.
- **nano-vllm-learning.md** — Long-term plan to master LLM inference via nano-vLLM. Tracks current phase, session log, checkpoints, questions. Read on any session mentioning nano-vLLM / learning / LLM study.

## Repo Snapshot
- Provider-neutral runtime: Anthropic Messages + OpenAI Responses adapters under `jyagent/runtime/`
- `RuntimeOwner` owns the active `provider:model`; `/model <provider> <model>` switches future turns
- Native tools in `jyagent/tools/`; per-tool metadata in `jyagent/tools/__init__.py`
- Session stats track provider/model, cache tokens, subagent usage, cost; unknown OpenAI snapshots can fetch pricing from docs
- CLI history rendering avoids Rich markup parsing on dynamic text
- Silent completions reuse the streaming runtime path, including subagents

## Durable Gotchas
- `contextvars.ContextVar` is NOT auto-propagated by `ThreadPoolExecutor.submit()` (Python 3.14.3). Use `ctx = contextvars.copy_context(); executor.submit(ctx.run, fn, ...)` to propagate.
- Daemon threads don't propagate `ContextVar` after spawn — use per-loop closures for tool state, not `ContextVar`.
- Codex CLI `--sandbox` flag (read-only / workspace-write / danger-full-access) only restricts **filesystem access** for model-generated shell commands. It does NOT restrict network / web search. Never blame sandbox mode for web-search issues.

## Durable Tips
- **Mutating Anthropic `system_prompt` breaks prompt caching** — inject dynamic context as a non-persisted tail message block instead. (This is also why MEMORY.md must stay stable across a session.)
- AWS Bedrock account lacks access to `claude-opus-4-7` (`permission_error` on `InvokeModelWithResponseStream`).
- Warp.dev deprecated `warp-cli` in favor of `oz` CLI (auto-updates). Cloudflare WARP's `warp-cli` is unrelated and still active.

## Goals
- Long-term: guide user through nano-vLLM to master LLM inference. Progress in `topics/nano-vllm-learning.md`. On any session mentioning nano-vLLM / learning / LLM study, read that file first.
