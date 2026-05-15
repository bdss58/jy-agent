# System prompt assembly — base prompt + memory + skill catalog.
#
# Owns the agent's behavioral constitution (the long string literal) and the
# cache-friendly assembly used by every turn. Extracted from agent.py so that
# the orchestrator stays focused on REPL flow, and prompt content is reviewable
# / testable in isolation.
#
# Cache strategy (do not break casually):
#   - The BASE template is constant.
#   - The memory portion (build_memory_context) is rebuilt only on compaction
#     or after explicit invalidation (see invalidate_memory_cache()).
#   - The skill catalog (name+description only) changes only when the on-disk
#     skills/ directory changes.
#   - Pinned skill BODIES are NOT included here — callers attach them as a
#     tail block on the last user message so per-turn pin diffs do not
#     invalidate the system-prompt prefix cache. See
#     ``SkillManager.build_pinned_bodies_block``.

from __future__ import annotations

import os

from .memory import build_memory_context
from .skills import SkillManager


# ─── Behavioral constitution (BASE template) ─────────────────────────────────

def base_system_prompt() -> str:
    """Render the static base prompt.

    Templated only on ``LAUNCH_DIR`` (where the user invoked jy-agent) and
    the source ``project_root`` — both stable for the life of the process,
    so the result is effectively constant per session.
    """
    from .config import LAUNCH_DIR
    launch_info = f"\nThe user launched you from: {LAUNCH_DIR}" if LAUNCH_DIR else ""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return f"""You are jy-agent, a self-assembled AI agent built by Jianyong, bootstrapped from a single API call.
You have access to tools for running shell commands, reading/writing files, and listing directories.
Think step by step. Use tools when needed to accomplish tasks.
Be helpful, precise, and concise.
Your source code project root is {project_root} (package: jyagent/).{launch_info}

CRITICAL BEHAVIORAL PRINCIPLES:

1. TOOL-FIRST PRINCIPLE: Before writing ad-hoc code or workarounds, check what tools are available. Use existing tools (web_fetch, chrome_browser, manage_memory, etc.) directly.

2. HONESTY PRINCIPLE: Never pretend to have looked something up when you have not. If you are answering from training data alone, say so clearly. If the user asks for recent/current information, use your tools (web_fetch, run_shell with curl, etc.) to actually fetch it. Do not fabricate citations, changelogs, or sources. If you are uncertain, say so.

3. TOOL-VERIFICATION REQUIREMENT:
   When the user asks about system state, environment, file contents, available tools,
   connected services (MCP, Chrome, etc.), or past interactions — ALWAYS verify with
   tools before answering. Never answer from memory or training data alone for factual
   claims about the current environment. Say "Let me check..." then use the appropriate
   tool. If the tool call fails, say so rather than falling back to unverified memory.

4. MEMORY AWARENESS: You have a three-tier self-use memory system (consensus design across Claude Code, Letta, Mem0, LangMem):
   - **MEMORY.md** (Tier 1, ALWAYS LOADED, hard cap 200 lines / 25 KB): the index. Durable, data-independent rules / facts only. Bloating it degrades attention and invalidates the prompt cache (~12× cost penalty).
   - **data/memory/topics/<name>.md** (Tier 2, on-demand): curated extended knowledge — architecture notes, library quirks, ongoing project state. Read with `read_file` when relevant.
   - **data/memory/journal/YYYY-MM.md** (Tier 3, on-demand, NEVER auto-loaded): append-only chronological notes — "what I worked on today", debug session logs. Equivalent of a lab notebook.

   Memory workflow:
   - Durable rule that prevents future mistakes? → `manage_memory(action='remember', category=..., text=...)` (1-line entry in MEMORY.md). Before adding, ask: "Would removing this cause the agent to make mistakes?" If not, do not add it.
   - Extended detail on a topic? → `manage_memory(action='topic', text='write:<name>|<content>')` (auto-indexes in MEMORY.md).
   - Chronological "what I did" note? → `manage_memory(action='journal', text=..., category=...)`. NEVER put dated session notes in MEMORY.md — that's the bug we just fixed.
   - Read topic detail: `read_file('data/memory/topics/<name>.md')`.
   - Read past journal: `read_file('data/memory/journal/<YYYY-MM>.md')`.
   - Audit MEMORY.md health: `manage_memory(action='consolidate')` (read-only dedup / bloat report).
   - Reorganize: rewrite MEMORY.md and topic files with `write_file` directly when needed.
   
   IMPORTANT: Memory provides hints and context, but for factual claims about the filesystem,
   environment, system state, available tools, connected services, or agent capabilities,
   ALWAYS verify with tools before presenting as fact. Memory may be stale or inaccurate.

5. SKILLS AWARENESS: You have an Agent Skills system (agentskills.io standard) that provides procedural knowledge.
   Skills are advertised in the `<available_skills>` block of your system prompt (progressive disclosure — the
   catalog is visible but full bodies are NOT loaded until requested). There is NO automatic router — YOU must
   bring a matching skill into context BEFORE executing the task whenever the user's request clearly matches a
   listed skill's TRIGGER clauses. Skipping that defeats the skill's checklists (e.g. web-search Step 0 verifies
   the date and would have prevented the 2025/2026 year bug on 2026-05-01).

   Two ways to bring a skill into context:
   - `manage_skills(action='load', name=X)` — PREFERRED. One-shot: returns the full SKILL.md body
     as a tool result. Use this for normal task-scoped skill use. Do NOT call `load` again for the same
     skill if its instructions are already visible above in the conversation.
   - `manage_skills(action='pin', name=X)` — only when the user EXPLICITLY asks to keep a skill on
     for the whole session. Pinned skills re-inject their full body on every user message and are
     token-expensive.

   - /skills — list all available skills and their status
   - /skill <name> — user-driven pin (the user's explicit pin command)
   - manage_skills tool — full skill management (load, pin, deactivate, list, create, …)
"""


# ─── Memory context cache ────────────────────────────────────────────────────

# The memory block does not depend on the user's query for assembly cost
# (build_memory_context internally retrieves matched topics, but the result
# changes only when memory itself changes — not per turn). We cache the
# rendered string and rebuild it on:
#   1. first turn (None sentinel),
#   2. after compaction (force_rebuild=True from the planner),
#   3. after the planner runs (memory tools may have written to MEMORY.md),
#   4. after /new (explicit invalidation).
_cached_memory_context: str | None = None


def invalidate_memory_cache() -> None:
    """Force the next build_system_prompt call to rebuild the memory block."""
    global _cached_memory_context
    _cached_memory_context = None


def _ensure_memory_context(user_input: str, *, force_rebuild: bool) -> str:
    global _cached_memory_context
    if force_rebuild or _cached_memory_context is None:
        _cached_memory_context = build_memory_context(query=user_input) or ""
    return _cached_memory_context


# ─── Project context block (G1-lite) ─────────────────────────────────────────
#
# When the user launches jy-agent from inside a project repo that already has
# an AGENTS.md / CLAUDE.md (the cross-tool 2025-26 convention for repo-rooted
# agent instructions), surface its content as a one-shot block in the system
# prompt — visible to the agent for the whole session, cache-stable across
# turns (LAUNCH_DIR is constant per session).
#
# Design notes:
#   * AGENTS.md is preferred over CLAUDE.md (broader 2026 adoption — Codex,
#     Cursor, Aider, Copilot, Gemini CLI, Windsurf, Amp, Amazon Q all read it).
#     CLAUDE.md is the Claude-Code-specific fallback.
#   * Search starts at LAUNCH_DIR and walks up to the user's home directory
#     (or filesystem root, whichever comes first). The nearest match wins.
#   * Hard caps:  PROJECT_CONTEXT_MAX_BYTES (12 KB) and PROJECT_CONTEXT_MAX_LINES
#     (300). Files exceeding either are truncated with a clear marker so the
#     agent knows there's more.
#   * Cached after first build for the lifetime of the process — content is
#     not re-read on every turn (no inotify, deliberately). To refresh after
#     editing the file, restart the agent.

PROJECT_CONTEXT_MAX_BYTES = 12 * 1024
PROJECT_CONTEXT_MAX_LINES = 300
_PROJECT_CONTEXT_FILENAMES = ("AGENTS.md", "CLAUDE.md")
_cached_project_context: str | None = None  # None = not yet computed; "" = no file
_cached_project_context_path: str | None = None


def _find_project_context_file(start_dir: str) -> str | None:
    """Walk ancestors of ``start_dir`` looking for AGENTS.md / CLAUDE.md.

    Stops at the user's home directory or the filesystem root, whichever
    comes first. AGENTS.md beats CLAUDE.md at the same level.
    """
    if not start_dir or not os.path.isdir(start_dir):
        return None
    home = os.path.realpath(os.path.expanduser("~"))
    cur = os.path.realpath(os.path.abspath(start_dir))
    # Cap the walk at 25 levels — defensive against weird mount layouts.
    for _ in range(25):
        # Home-guard: a file directly at $HOME (or anywhere above) is
        # treated as global, not project-scoped, and is intentionally
        # ignored. Use MEMORY.md for global rules; the project-context
        # block is reserved for repo-local norms. Stop the walk BEFORE
        # peeking inside $HOME so a `cd ~ && jy-agent` from a user that
        # happens to keep an AGENTS.md at $HOME doesn't always preload it.
        if cur == home:
            return None
        for name in _PROJECT_CONTEXT_FILENAMES:
            candidate = os.path.join(cur, name)
            if os.path.isfile(candidate):
                return candidate
        parent = os.path.dirname(cur)
        if parent == cur:           # filesystem root
            return None
        cur = parent
    return None


def _read_project_context_file(path: str) -> str:
    """Read and cap a project context file. Returns the wrapped block body."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read(PROJECT_CONTEXT_MAX_BYTES + 1)
    except Exception:
        return ""
    truncated_by_bytes = len(raw) > PROJECT_CONTEXT_MAX_BYTES
    if truncated_by_bytes:
        raw = raw[:PROJECT_CONTEXT_MAX_BYTES]
    lines = raw.splitlines()
    truncated_by_lines = len(lines) > PROJECT_CONTEXT_MAX_LINES
    if truncated_by_lines:
        lines = lines[:PROJECT_CONTEXT_MAX_LINES]
    body = "\n".join(lines)
    if truncated_by_bytes or truncated_by_lines:
        body += (
            "\n\n[... truncated by jy-agent: file exceeds "
            f"{PROJECT_CONTEXT_MAX_BYTES} bytes / {PROJECT_CONTEXT_MAX_LINES} lines. "
            "Use read_file to see the full content on demand.]"
        )
    return body


def _build_project_context_block() -> str:
    """Locate and render the project context block. Cached after first call."""
    global _cached_project_context, _cached_project_context_path
    if _cached_project_context is not None:
        return _cached_project_context
    from .config import LAUNCH_DIR
    path = _find_project_context_file(LAUNCH_DIR)
    if path is None:
        _cached_project_context = ""
        return ""
    body = _read_project_context_file(path)
    if not body.strip():
        _cached_project_context = ""
        return ""
    _cached_project_context_path = path
    # XML-wrap for clear delimitation; same pattern as memory + skills blocks.
    _cached_project_context = (
        "<project_context>\n"
        f"Source file: {path}\n"
        "This file was authored by a human (or another agent) to describe "
        "this project's conventions. Treat it as ground truth for project-"
        "specific norms (build commands, test framework, code style, etc.).\n\n"
        f"{body}\n"
        "</project_context>"
    )
    return _cached_project_context


def project_context_source() -> str | None:
    """Public accessor: returns the on-disk path of the active project
    context file, or None if none was found. Useful for CLI banner /
    debugging."""
    _build_project_context_block()  # ensure cache is populated
    return _cached_project_context_path


def invalidate_project_context_cache() -> None:
    """Force re-detection on the next build (e.g. after the user moves
    or edits the project file). Not currently wired to any slash command
    — restart the agent to refresh project context."""
    global _cached_project_context, _cached_project_context_path
    _cached_project_context = None
    _cached_project_context_path = None


# ─── Public builders ─────────────────────────────────────────────────────────

def build_system_prompt(
    user_input: str,
    skill_mgr: SkillManager | None = None,
    *,
    force_rebuild: bool = False,
    include_skills: bool = True,
) -> str:
    """Assemble the full system prompt: base + memory + project context +
    (optional) skill catalog.

    Layer order (cache-prefix-friendly: most stable first):

      1. base_system_prompt()       — constant per process
      2. memory block               — changes only on memory writes/compaction
      3. project context block      — constant per process (LAUNCH_DIR-keyed)
      4. skill catalog              — changes only when skills/ dir changes

    ``include_skills=False`` is used by compaction so the summarizer sees the
    same base+memory prefix as a regular turn (cache-friendly) without the
    skill catalog churn.
    """
    memory = _ensure_memory_context(user_input, force_rebuild=force_rebuild)
    project_ctx = _build_project_context_block()

    prompt = base_system_prompt()
    if memory:
        prompt = prompt + "\n\n" + memory
    if project_ctx:
        prompt = prompt + "\n\n" + project_ctx

    if include_skills and skill_mgr is not None:
        catalog = skill_mgr.build_catalog_block()
        if catalog:
            prompt = prompt + "\n\n" + catalog

    return prompt
