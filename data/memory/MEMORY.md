# Agent Memory

This is the memory index. Detailed knowledge lives in topic files under
`data/memory/topics/`. Keep this file concise and read topic files on demand
with `read_file` when details are needed.

## User Profile
- Name: Jianyong
- Role: infrastructure/DevOps engineer
- OS: macOS
- Language: Chinese (Mandarin), bilingual with English
- Timezone: Asia/Shanghai
- Communication style: direct, holds agents accountable, often speaks Chinese
- Tech stack: Python, Docker, Kubernetes, Bash, FastAPI, Chrome DevTools/MCP, MCP, Anthropic API, OpenAI API, AWS Bedrock, vllm, OpenClaw, GitLab
- Projects: `jy-agent`, `openclaw-enterprise`, `snake-h5-game`

## Behavioral Rules (CRITICAL)
- Never fabricate command results or claim actions were done without executing them
- Verify with tools (`pwd`, `find`, `date`, etc.) before answering filesystem or environment questions
- Use live search/browser tools for current information; do not rely on stale model memory
- Verify current date with `date` before time-sensitive research
- Keep `MEMORY.md` concise and move detailed notes into topic files
- For self-upgrades to jy-agent runtime/source code, use `git worktree` (see `git-worktree` topic)
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

## Repo Snapshot
- Runtime is provider-neutral: Anthropic Messages + OpenAI Responses adapters under `jyagent/runtime/`
- `RuntimeOwner` owns the active `provider:model`; `/model <provider> <model>` switches future turns
- Core native tools live in `jyagent/tools/` and register with per-tool metadata in `jyagent/tools/__init__.py`
- Session stats track provider/model, cache tokens, subagent usage, and cost; unknown OpenAI snapshots can fetch pricing from docs
- CLI history rendering avoids Rich markup parsing on dynamic text and formats normalized assistant/tool blocks safely
- Silent completions now reuse the streaming runtime path, including subagents

## Topic Index
- **architecture**: current tool/runtime/memory/CLI architecture
- **chrome-mcp**: Chrome DevTools MCP behavior, dead-browser recovery, lifecycle gotchas
- **web-fetch**: 5-tier cascade, JS-heavy routing, Chrome tier details, fake-success detection
- **git-worktree**: required workflow for jy-agent self-upgrades
- **claude-code-best-practices**: delegation patterns, verification rules, local model-selection guidance
[tip] Memory Phase 1 shipped: (1) topic file frontmatter timestamps, (2) session persistence with /continue, (3) MAX_MEMORY_PROMPT_CHARS 5K→10K, (4) proactive extraction hook (background thread, every 4 turns)
[tip] Subagent improvements shipped (2025-07-08): (1) P0: model label overwrite fix, (2) P1: per-subagent cost tracking via subagent_runs list, (3) P1: consolidated _SubagentTracker spinner for parallel dispatch, (4) P2: memory injection into subagent system prompt, (5) P2: step progress in spinner via on_step_progress callback, (6) P2: declarative agent definitions in jyagent/agents/ + data/agents/*.md (explorer/researcher/reviewer built-in)
[tip] `contextvars.
[gotcha] `contextvars.ContextVar` is NOT auto-propagated by `ThreadPoolExecutor.submit()` (verified Python 3.14.3 — worker sees default, not caller's value). Must use `ctx = contextvars.copy_context(); executor.submit(ctx.run, fn, ...)` to explicitly propagate.

[tip] Harness engineering research completed (2026-04-10). Current jy-agent maturity: 3.3/5. Plan in data/harness-improvement-plan.md. Key gaps: output verification (2/5), tracing (2/5). Quick wins: cost budget, duplicate-call detection, remediation messages.
[tip] Context compaction upgrade shipped (Phase 1+2). Multi-tier system: Tier 0 = thinking block pruning, Tier 1 = observation masking (full clear beyond OBSERVATION_MASK_DISTANCE=5), Tier 2 = compaction_priority awareness (ephemeral/standard/persistent per tool). Also: cache-friendly compaction (reuses system prompt), file re-injection post-compaction (FileAccessTracker), 9-section summary prompt. Config: AGENT_OBSERVATION_MASK_DISTANCE, AGENT_FILE_REINJECTION_COUNT, AGENT_FILE_REINJECTION_MAX_TOKENS.
[tip] Chrome MCP concurrency fix shipped (2025-07-17): (1) _chrome_page_lock serializes multi-step Chrome operations (new_page→select_page→evaluate/snapshot→close_page) as atomic critical sections, (2) _chrome_acquire/_chrome_release with refcounting replaces broken was_connected boolean — only last caller disconnects, (3) explicit select_page before every evaluate_script/take_snapshot, (4) _fetch_chrome in web_fetch.py now delegates to MCPManager.chrome_fetch_page() — single codepath, 94 lines removed. Root cause: Chrome MCP evaluate_script/take_snapshot have NO pageId param, always use selected page cursor.
[tip] QW-4 (tracing) and QW-5 (verification) shipped (2026-04-11). (1) `jyagent/tracing.py`: JSONL trace logger — RunTrace/SpanEvent, flushes to `data/traces/`, controlled by `AGENT_TRACE_ENABLED=1`. (2) `jyagent/verification.py`: pre-completion verification gate — injects self-check prompt when file mutations detected, controlled by `AGENT_VERIFICATION_ENABLED=1`. Both wired into loop_engine.py. Tests: 33 new in `tests/test_tracing_and_verification.py`, all passing.
[tip] Runtime review & hardening shipped (2026-04-11). Dual-agent review (Claude Code + Codex CLI) found 22 issues. 10 fixes implemented: (1) double-iteration guard on streams, (2) SSL default flipped to verify=True (env SSL_VERIFY=0 to opt out), (3) partial content on OpenAI stream errors, (4) narrowed ImportError catch in provider registration, (5) OpenAI reasoning validation, (6) malformed JSON logging + _parse_error key, (7) Required[] on TypedDict discriminators, (8) BaseStream DRY class, (9) auto-close on stream exhaustion, (10) unknown block passthrough in transform_messages_for_target. 116 new tests in tests/test_runtime.py. SSL_VERIFY env var semantics changed — must set SSL_VERIFY=0 explicitly to disable (was disabled by default).
[tip] Response-aware stuck-loop detection shipped (2026-07-09). Replaced _DedupTracker (whitelist: dedup_exempt flag + sleep regex) with _StuckLoopDetector (tracks (tool, args, response_hash) triples). If response changes → progress → reset counter. If response identical N times → stuck → break. Eliminates all exemption metadata, sleep regex, MCP tool gaps. Polling tools like check_background naturally exempt because elapsed_seconds changes each response. Check moved from pre-execution to post-execution in loop_engine.py.
[tip] web_search native tool shipped (2026-04-13). New `web_search` tool in `jyagent/tools/web_search_tool.py` with 3 engines: (1) `ddg` — DuckDuckGo HTML parsing via BeautifulSoup, fast/free, (2) `codex` — delegates to Codex CLI with `--output-schema` for structured JSON results + synthesis, (3) `auto` — DDG first, Codex fallback if <3 results. Registered in __init__.py as parallel_safe=True. web-search skill updated to v4.0 with `web_search()` as primary interface. 18 unit tests in tests/test_web_search.py.
