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
- **Codex as second opinion** on significant tasks (code review, bug investigation, planning, research). `codex review` for code reviews, `codex exec --sandbox read-only` for analysis. Skip for trivial tasks.

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
- Working on offline image update process for OpenClaw shipped to customer hosts (see `topics/openclaw-offline-update.md`)
- K8s test host: `wan2` (`wan2.think-force.com`), `kubectl port-forward` on port 31555

## Topic Files Index
- **memory-design.md** — Three-tier memory architecture (this file's design rationale + routing rules). Read this if asked to refactor or extend the memory system.
- **gfw-proxy-fallback.md** — SSH SOCKS5 tunnel workflow for GFW-blocked hosts (ghcr.io, raw.githubusercontent.com, huggingface.co, etc.). Read when a network call fails against a likely-blocked host.
- **openclaw-offline-update.md** — OpenClaw offline bundle update process (preflight, docker-compose schema migration, build-artifact staleness). Read on any OpenClaw customer-deployment question.
- **nano-vllm-learning.md** — Long-term plan to master LLM inference via nano-vLLM. Tracks current phase, session log, checkpoints, questions. Read on any session mentioning nano-vLLM / learning / LLM study.
## Repo Snapshot
- Provider-neutral LLM layer: Anthropic Messages + OpenAI Responses adapters under `jyagent/llm/`
- `LLMOwner` owns the active `provider:model`; `/model <provider> <model>` switches future turns
- Native tools in `jyagent/tools/`; per-tool metadata in `jyagent/tools/__init__.py`
- Session stats track provider/model, cache tokens, subagent usage, cost; unknown OpenAI snapshots can fetch pricing from docs
- CLI history rendering avoids Rich markup parsing on dynamic text
- Silent completions reuse the streaming runtime path, including subagents

## Durable Gotchas
- `contextvars.ContextVar` is NOT auto-propagated by `ThreadPoolExecutor.submit()` (Python 3.14.3). Use `ctx = contextvars.copy_context(); executor.submit(ctx.run, fn, ...)` to propagate.
- Daemon threads don't propagate `ContextVar` after spawn — use per-loop closures for tool state, not `ContextVar`.
- Codex CLI `--sandbox` flag (read-only / workspace-write / danger-full-access) only restricts **filesystem access** for model-generated shell commands. It does NOT restrict network / web search. Never blame sandbox mode for web-search issues.
- `kubectl port-forward` defaults to binding 127.0.0.1 only. Use `--address 0.0.0.0` to expose externally.
- After modifying a build artifact (tarball/bundle), **re-run the build and verify** (`tar tzf | grep <file>`). Patching the build script alone does not refresh the artifact, and `scp`'ing a file into an already-extracted remote dir hides the bug.

## Durable Tips
- **Mutating Anthropic `system_prompt` breaks prompt caching** — inject dynamic context as a non-persisted tail message block instead. (This is also why MEMORY.md must stay stable across a session.)
- AWS Bedrock account lacks access to `claude-opus-4-7` (`permission_error` on `InvokeModelWithResponseStream`).
- Warp.dev deprecated `warp-cli` in favor of `oz` CLI (auto-updates). Cloudflare WARP's `warp-cli` is unrelated and still active.

## Goals
- Long-term: guide user through nano-vLLM to master LLM inference. Progress in `topics/nano-vllm-learning.md`. On any session mentioning nano-vLLM / learning / LLM study, read that file first.
[user_stated] OpenClaw architecture: ocaw-run pod runs openclaw supervisor + openclaw-gateway (Python WebSocket on port 18789); browser/OCR/LLM delegated to separate services (browser-service, processor-service, model-gateway)
[user_stated] OpenClaw ocaw-run pods steady-state: ~510 MiB RSS+cache (max 652), 1-3 millicores CPU; 0 OOMs across 1600+ pod-hours observed
[preference] For sizing recommendations, prefer empirical measurement (live pod inspection, cgroup memory.events) over guessing; recommend Burstable QoS over Gu
[gotcha] git worktree shares the parent repo's editable-install `.venv` — `uv run` from the worktree imports the package from the MAIN worktree's path (per `__editable___*_finder.py` MAPPING), NOT the worktree's source. Tests run in a worktree may silently exercise the WRONG code. Fix: install a local editable venv in the worktree (`uv sync` inside it) before testing, OR run tests on main after merge.
[gotcha] When moving a Python module deeper in a package tree, audit every `os.path.dirname(__file__)` chain and `Path(__file__).parents[N]` — depth count is coupled to file location and silently resolves to the wrong path (no ImportError). Check skills.py-style "no items found" symptoms first.
[tip] SearxNG settings.yml supports `use_default_settings: true` — write only your overrides; everything else (incl. 281 engines) inherits from the default in the searxng/searxng image. Must still override `server.secret_key` (default placeholder `ultrasecretkey` makes the app refuse to start).
[tip] SearxNG only honors a few env overrides (SEARXNG_PORT, BIND_ADDRESS, SECRET_KEY, LIMITER, PUBLIC_INSTANCE); search.formats must come from settings.yml
[tip] Docker Compose v2.23+ supports inline `configs.content:` to embed file contents in docker-compose.yml, avoiding companion files
~~[tip] web_search cascade order: SearxNG → Brave → Mojeek → DDG (DDG last due to flakiness). SEARXNG_URL activates SearxNG. WEB_SEARCH_ENGINE=name forces single engine. Tests in test_web_search.py must reflect this order.~~  (superseded 2026-04-27: web_search cascade order: SearxNG (SEARXNG_URL) → Brave API …)
~~[tip] web_search cascade order: SearxNG (SEARXNG_URL) → Brave API (BRAVE_SEARCH_API_KEY, official JSON) → Mojeek (HTML, often IP-blocked) → DDG (HTML, last). Brave HTML scraper retired 2026-04 due to PoW captcha. WEB_SEARCH_ENGINE=name forces single engine.~~  (superseded 2026-04-27: web_search cascade order: SearxNG (SEARXNG_URL) → DDG. Brave…)
[tip] web_search cascade order: SearxNG (SEARXNG_URL) → DDG. Brave + Mojeek removed 2026-04-26: Brave serves PoW captcha to scrapers, Mojeek IP-blocks all datacenter+WARP exits. Run SearxNG for richer aggregation. WEB_SEARCH_ENGINE=name forces single engine.
[gotcha] Before `git commit`, always `git status --short` — `git mv` stages the rename but later edits to OTHER files via edit_file/sed remain unstaged and will be silently dropped from the commit. Use `git add -A` or stage explicitly.
[gotcha] After C4 Phase 5, `runtime/loop/step.py::run_step` calls `tool_executor.execute_tools` directly (not via `engine._execute_tools` alias). Patches against `jyagent.runtime.loop.engine._execute_tools` DO NOT intercept per-step tool calls. Patch `jyagent.runtime.loop.tool_executor` instead.
