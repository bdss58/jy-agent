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


# ─── Public builders ─────────────────────────────────────────────────────────

def build_system_prompt(
    user_input: str,
    skill_mgr: SkillManager | None = None,
    *,
    force_rebuild: bool = False,
    include_skills: bool = True,
) -> str:
    """Assemble the full system prompt: base + memory + (optional) skill catalog.

    ``include_skills=False`` is used by compaction so the summarizer sees the
    same base+memory prefix as a regular turn (cache-friendly) without the
    skill catalog churn.
    """
    memory = _ensure_memory_context(user_input, force_rebuild=force_rebuild)

    prompt = base_system_prompt()
    if memory:
        prompt = prompt + "\n\n" + memory

    if include_skills and skill_mgr is not None:
        catalog = skill_mgr.build_catalog_block()
        if catalog:
            prompt = prompt + "\n\n" + catalog

    return prompt
