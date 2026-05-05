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

## Goals
- Long-term: guide user through nano-vLLM to master LLM inference. Progress in `topics/nano-vllm-learning.md`. On any session mentioning nano-vLLM / learning / LLM study, read that file first.
[user_stated] OpenClaw architecture: ocaw-run pod runs openclaw supervisor + openclaw-gateway (Python WebSocket on port 18789); browser/OCR/LLM delegated to separate services (browser-service, processor-service, model-gateway)
[user_stated] OpenClaw ocaw-run pods steady-state: ~510 MiB RSS+cache (max 652), 1-3 millicores CPU; 0 OOMs across 1600+ pod-hours observed
[preference] For sizing recommendations, prefer empirical measurement (live pod inspection, cgroup memory.events) over guessing; recommend Burstable QoS over Guaranteed for bursty workloads.
[gotcha] git worktree shares the parent repo's editable-install `.venv` — `uv run` from the worktree imports the package from the MAIN worktree's path (per `__editable___*_finder.py` MAPPING), NOT the worktree's source. Tests run in a worktree may silently exercise the WRONG code. Fix: install a local editable venv in the worktree (`uv sync` inside it) before testing, OR run tests on main after merge.
[gotcha] When moving a Python module deeper in a package tree, audit every `os.path.dirname(__file__)` chain and `Path(__file__).parents[N]` — depth count is coupled to file location and silently resolves to the wrong path (no ImportError). Check skills.py-style "no items found" symptoms first.
[tip] SearxNG settings.yml supports `use_default_settings: true` — write only your overrides; everything else (incl. 281 engines) inherits from the default in the searxng/searxng image. Must still override `server.secret_key` (default placeholder `ultrasecretkey` makes the app refuse to start).
[tip] SearxNG only honors a few env overrides (SEARXNG_PORT, BIND_ADDRESS, SECRET_KEY, LIMITER, PUBLIC_INSTANCE); search.formats must come from settings.yml
[tip] Docker Compose v2.23+ supports inline `configs.content:` to embed file contents in docker-compose.yml, avoiding companion files
[tip] web_search cascade order: SearxNG (SEARXNG_URL) → DDG. Brave + Mojeek removed 2026-04-26: Brave serves PoW captcha to scrapers, Mojeek IP-blocks all datacenter+WARP exits. Run SearxNG for richer aggregation. WEB_SEARCH_ENGINE=name forces single engine.
[gotcha] Before `git commit`, always `git status --short` — `git mv` stages the rename but later edits to OTHER files via edit_file/sed remain unstaged and will be silently dropped from the commit. Use `git add -A` or stage explicitly.
[gotcha] After C4 Phase 5, `runtime/loop/step.py::run_step` calls `tool_executor.execute_tools` directly (not via `engine._execute_tools` alias). Patches against `jyagent.runtime.loop.engine._execute_tools` DO NOT intercept per-step tool calls. Patch `jyagent.runtime.loop.tool_executor` instead.
[gotcha] containerd 2.x renamed CRI config keys: `sandbox_image` → `sandbox`, and `config_path` defaults to empty `''` (not `/etc/containerd/certs.d`). Old sed patterns silently no-op; mirror dirs are silently ignored. After `containerd config default`, must explicitly set both, then `systemctl restart containerd`.
[gotcha] `pgrep -af <name>` matches its own wrapper bash command line when invoked via `bash -c "while pgrep -af X; do ..."` (X appears in the wrapper's argv) → infinite loop. Use `pgrep -x <exact_binary>` or `fuser /var/lib/dpkg/lock-frontend` for apt locks instead.
[tip] `ctr image pull` does NOT honor `/etc/containerd/certs.d/` mirror config — that's CRI-plugin-only. To test containerd registry mirrors, deploy a pod via kubectl, not `ctr`.
[gotcha] `warnings.deprecated` (PEP 702) requires Python 3.13+; use `typing_extensions.deprecated` for 3.12 compat
[tip] For generic build pipelines that need both build-host and customer-host validation: bake the validator INTO the image (e.g. /app/preflight.sh + /app/preflight.d/*.sh). Both sides run `docker exec <ct> /app/preflight.sh` — eliminates the dual-list drift bug.
[tip] Dockerfile build-arg cache: declare `ARG FOO_VERSION` JUST BEFORE the RUN that uses it, not at the top of the stage. Adding/changing an ARG invalidates the cache for every downstream RUN, even ones that don't reference it.
[gotcha] `claude -p --bare` buffers ALL output until completion. If SIGTERMed at deadline, you get an empty `output` field BUT disk changes already persisted — recover via `git status` / `git diff`. Don't conclude "Claude Code did nothing".
[tip] Anthropic prompt caching is enabled by default via top-level `cache_control={"type":"ephemeral"}` in `_anthropic_helpers.build_request_kwargs` (commit 38f049a). Set `ANTHROPIC_PROMPT_CACHE=0` to disable, `ANTHROPIC_PROMPT_CACHE_TTL=1h` for 1h cache. Requires anthropic-sdk-python >=0.83.0 (the version that added the top-level kwarg).
[tip] To revise an existing MEMORY.md rule: journal the change first ([memory_revision] category), then `forget` the old keyword, then `remember` the new fact. The `supersede` action was removed 2026-04-30 — Tier 1 stays lean, audit trail lives in Tier 3.
[tip] When extracting helpers from a large function, every keyword arg that was wired inline at the old call-site is a regression candidate — unit tests built via `__new__` with defaulted loop attrs (e.g. `_cancel_event=None`) silently mask dropped kwargs. Always `codex review` such refactors and/or add a source-level kwarg-presence test.
[gotcha] `subprocess.run(capture_output=True, text=True)` buffers the ENTIRE child output in RAM before any truncation — a runaway child can OOM-kill the parent (we saw ~74 GiB → macOS jetsam, 2026-04-30). For unbounded-output children, drain via threads into bounded head+tail bytearrays and SIGKILL pgroup on overflow. See `_BoundedStreamReader` in `jyagent/tools/core.py`.
[gotcha] When applying both a char cap and an overflow/error marker to tool output, ALWAYS apply the cap first, then append the marker. Marker-then-cap silently slices off the very signal you care about for runaway-output cases.
[gotcha] Don't claim a patch "solves" a reported bug without
[gotcha] Head+tail bounded buffer: "tail has everything" iff `total <= tail_max` (NOT `total <= head_max + tail_max`). The latter is when head+tail cover the full output WITHOUT a gap (regime 2), but tail alone has dropped its early bytes. Three regimes: tail-only / dedup-merge / head+marker+tail.
[tip] Lazy spill activation pattern: when streaming with bounded head+tail buffers, open the spill file the moment `total + chunk_size > tail_max`. At that instant tail still has bytes [0, total) — dump it to seed the spill, then tee subsequent chunks. No pre-spill buffer needed; memory stays O(head + tail).
[tip] For env-var-configurable knobs, parse via a helper that warns to stderr on malformed/below-minimum values; never silently fall back. Claude Code's BASH_MAX_OUTPUT_LENGTH regression (#17944, v2.1.2+) was a silent-ignore bug that broke users' workflows for releases. See `_env_int` / `_env_bool` in jyagent/tools/core.py.
[gotcha] For time-sensitive web_search queries, run `date` in a SEPARATE earlier tool batch — not parallel with the searches — so the verified year is read before composing queries. Training-data anchors to old "current year" otherwise. Or omit the year and let the engine rank by recency.
[gotcha] Test-source-inspection assertions (e.g. `engine_src[idx : idx + N]` window checks) break on accumulator additions even when behaviour is preserved. Fix by widening the bound, not by restructuring — the assertion's intent (`cost_tracker.record` lives in the fallback block) is orthogonal to distance.
[tip] For mass edits across N call sites with a stable pattern (e.g. adding kwargs to every `_finalize_run(...)` call): use a Python regex script via run_shell with an assertion-guarded count check (`new.count(...) == count_before`) rather than sed. Catches partial-replacement bugs safely.
[correction] Skills have NO auto-router — catalog is advertised in the system prompt and YOU must self-activate via `manage_skills(action='activate', name=...)` when a request matches a skill's TRIGGER. Skipping activation because "I already know how" defeats the skill's checklists. (Per-turn SKILL_PRE_ROUTER was removed 2026-05; see journal.)
[workflow] codex review/exec on `neo` provider hangs in reconnect loop; force `-c model_provider=models-proxy` (or whichever provider has working keys). Always pre-budget ≥25 min for substantive reviews — if it times out mid-stream, the reasoning trail in /tmp/jyagent_bg_*.out still has actionable findings even without the final summary.
[gotcha] Manual `forget` requires ≥6-char keyword and skips Behavioral Rules / User Profile / User Preferences. UPDATE/replace deletes by matched-line index, not substring. To bypass, use `forget_from_memory_md(k, min_keyword_len=0, protect_sections=False)` (internal only).
