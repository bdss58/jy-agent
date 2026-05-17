# Agent Memory

Always-loaded index. **Hard cap: 200 lines / 25 KB.**
Extended detail → `data/memory/topics/<name>.md` (curated, on demand).
Chronological notes → `data/memory/journal/YYYY-MM.md` (never auto-loaded).

## User Profile
- Name: Jianyong
- Role: infrastructure/DevOps engineer
- OS: macOS (Apple Silicon), username `jyxc-dz-0100398`, timezone Asia/Shanghai
- Languages: Mandarin Chinese (primary) + English; direct, holds agents accountable
- Stack: Python, Docker, Kubernetes, Bash, FastAPI, Chrome DevTools/MCP, Anthropic/OpenAI/AWS Bedrock APIs, vllm, OpenClaw, GitLab

## Behavioral Rules (CRITICAL)
- Never fabricate command results or claim actions were done without executing them.
- Verify with tools (`pwd`, `find`, `date`, `ls`, `git log`, …) before answering filesystem/environment/system-state questions. Memory may be stale.
- Run `date` BEFORE composing any search query containing a year token (incl. inside `dispatch_agent` prompts for deep-research). Training prior defaults to ~2025; web-search skill's Step 0 must be inherited by subagents.
- Keep MEMORY.md concise (≤200 lines); move detail → topic files; chronological "what I did" → journal.
- **For self-upgrades to jy-agent runtime/source, use `git worktree`** to avoid disrupting concurrent agent sessions. User runs jy-agent in multiple Ghostty/Zellij panes simultaneously.
- Before any irreversible UI action (send message, submit form, delete), screenshot + verify target identity AND pause for user confirmation. Never press the final ⏎ on automation alone.
- Stop importing enterprise / multi-tenant / coding-agent "best practices" without first asking: *does the premise transfer to a single-user personal laptop with one hand-configured MCP server?* Default skeptical when threat model is multi-tenant/marketplace.

## User Preferences
- **Codex CLI consulted during DESIGN and PLANNING**, not only post-hoc review. Workflow: draft a plan → `codex exec` (or `codex review` on plan doc) for second opinion → THEN code. Skip for trivial tasks.
- Dual output: keep raw streaming + rendered markdown panel (not a bug).
- Chrome MCP: use independent instance, not user's Chrome.
- MCP as primary tool-integration mechanism.
- Prefers robust solutions (CLI args, config) over fragile source patches.
- When delegating to Claude Code, prefer the locally-configured default model; only pass `--model` for an intentional tier override.

## Environment
- Project: `/Users/jyxc-dz-0100398/jy-agent` (package `jyagent/`).
- User launches jy-agent from home dir (`~`), NOT project root — prefix paths with project root in tool calls.
- Terminal: Ghostty.app + Zellij multiplexer; `osascript` runs as a child of Ghostty for TCC purposes.
- Python 3.14 `.venv` has no pip by default — bootstrap with `.venv/bin/python -m ensurepip --upgrade`. System `/usr/bin/python3` has arm64/x86_64 mismatch issues with PIL/numpy from `~/Library/Python`.
- Dependency lockfile: `uv.lock`.

## Topic Files Index
- **wechat-mac-automation.md** — WeChat for Mac UI-automation playbook (⌘F search quirks, AX-tree dead end, pixel-row classification, Quartz click synthesis, clipboard PNG paste). Read for any WeChat-on-Mac task.
- **memory-upgrades-2026-05.md** — Recency boost + curated query expansion shipped 2026-05-17. Synonym-curation criteria, deliberate non-features (no embeddings), deferred roadmap. Read BEFORE invasive changes to `jyagent/memory/search.py`.
- **simplification-audit-2026-05.md** — Over-Design Audit (2026-05-15, jointly with Codex) — SUPERSEDED 2026-05-17; methodology lesson preserved ("design-audit plans go stale within days; re-validate item-by-item").

- **memory-upgrades-2026-05.md** — Memory Subsystem Upgrades — 2026-05-17
## Repo Snapshot
- Provider-neutral runtime: Anthropic Messages + OpenAI Responses adapters under `jyagent/llm/`; agent loop under `jyagent/runtime/`.
- Native tools in `jyagent/tools/`; per-tool metadata in `jyagent/tools/__init__.py`.
- Memory subsystem in `jyagent/memory/`; macOS automation (WeChat etc.) in `jyagent/macos/`; MCP integration in `jyagent/mcp/`; skills in `jyagent/skills.py`.
- Session stats track provider/model, cache tokens, subagent usage, cost.
- Subagent outcomes are idempotent + disk-persisted (since 2026-05-15): `check_agent(agent_id=N)` can be called repeatedly and survives wait-timeout drops via fallback to `data/sessions/subagents/<pid>-N.json`.

## Durable Gotchas
- `contextvars.ContextVar` is NOT auto-propagated by `ThreadPoolExecutor.submit()` (Python 3.14). Use `ctx = contextvars.copy_context(); executor.submit(ctx.run, fn, ...)`. Daemon threads don't propagate post-spawn either — use per-loop closures for tool state.
- macOS TCC has SEPARATE permission buckets: **Automation** (apple-events between apps) vs **Accessibility** (read/click UI elements via AX). Granting one does NOT grant the other. Error `-25211 "不允许辅助访问"` = Accessibility missing. For Ghostty, both must be enabled in System Settings → Privacy & Security.
- SQLAlchemy `server_default=sa.text('0')` on a `Boolean` column works on SQLite (no real BOOLEAN type) but emits `BOOLEAN DEFAULT 0` on Postgres, which is REJECTED at CREATE TABLE. Use `sa.false()` / `sa.true()` for cross-dialect Boolean defaults.
- Codex CLI `--sandbox` flag (read-only / workspace-write / danger-full-access) only restricts **filesystem access** for model-generated shell commands. It does NOT restrict network/web search. Never blame sandbox mode for web-search issues.

## Durable Tips
- **Mutating Anthropic `system_prompt` breaks prompt caching** (~12× per-token cost penalty). Inject dynamic context as a non-persisted tail message block. This is also why MEMORY.md must stay stable across a session.
- Codex's "over-designed / fragmented" structural critiques sometimes confuse **file count with concept count**. For each consolidation, ask: does inlining reduce independent concerns, or just relocate them? If the latter, file boundaries ARE the documentation — keep them.
- For non-AppKit apps (Electron, custom canvas like WeChat), pixel-structural analysis with PIL beats vision models when vision API is flaky.
[gotcha] Never emit `[Tool call: NAME]{json}` or ```tool_use``` fenced blocks as prose — those are NOT real invocations and trip `[MALFORMED_TOOL_CALL]`. Always use the structured function-calling channel. Especially watch this after a long string of successful tool calls in a session, and re-issue the call promptly when the malformed-call guard fires.
[user_stated] cashcat-api 的 claw/cashclaw 产品线底层运行时确认为开源 OpenClaw；dispatcher 编排每用户一个 Pod，Pod 通过 hooksToken 回调 /openapi/v1/*。
[workflow] Codex exec design reviews on step-managed-agents: short doc (<500 lines) + "Start your response with VERDICT: ship|fold|redraft" at TOP of prompt = reliable clean output in 140-510s. Phases M/N/O/P all completed cleanly. The earlier "codex times out" concern only applied to one 800+ line Phase L doc.
[preference] Embeddings rejected for memory search on personal-laptop assistant
