# Agent — Main run loop and command handlers.

import os
import sys
from .runtime.tools.registry import get_registry
import jyagent.tools  # noqa: F401 — triggers tool registration
from .memory import (
    ConversationMemory, summarize_if_needed,
    build_memory_context,
    save_session, load_session, has_saved_session, archive_session,
    list_sessions, find_session,
    should_extract, extract_and_remember,
    record_file_access,
)
from .ui.terminal import build_streaming_callbacks, _interrupted_msg
from .runtime.loop.engine import AgentLoop, LoopConfig, LoopResult
from .ui.cli import CLI, console
from .skills import SkillManager, get_skill_manager, init_skills
from .llm import LLMOwner
from .runtime.stats import get_stats
from .config import (
    DEFAULT_MAX_TOKENS, MAX_TOKENS_CAP, DEFAULT_MAX_STEPS,
    MAX_TOOL_RESULT_CHARS, MAX_WORKING_TOKENS, DEFAULT_TOOL_TIMEOUT,
    COMPACT_TOOL_RESULT_CHARS,
)


# ─── System prompt (externalized from run()) ──────────────────────────────────

def _base_system_prompt() -> str:
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
   `activate` is a deprecated alias of `pin`; prefer `load` or `pin` explicitly."""



# ─── Command handlers ────────────────────────────────────────────────────────

def _cmd_help(cli, **_):
    cli.print_help()

def _cmd_multi(cli, **_):
    cli.toggle_multiline()

def _cmd_markdown(cli, state, **_):
    state["use_markdown"] = not state["use_markdown"]
    cli.print_system(f"Markdown rendering {'ON' if state['use_markdown'] else 'OFF'}")

def _cmd_history(cli, conversation, **_):
    recent = conversation.get_recent(10)
    cli.print_history(recent)

def _cmd_new(cli, runtime_owner, conversation, **_):
    """Clear current conversation state and start fresh."""
    global _cached_memory_context

    # Archive current conversation before clearing (recoverable, but /continue
    # still points to the last *exited* session, not this one).
    if conversation.messages:
        try:
            from .runtime.stats import get_stats
            stats = get_stats()
            archive_session(conversation, metadata={
                "provider": stats.provider or "",
                "model": stats.model or "",
                "reason": "new",
            })
        except Exception:
            pass  # Don't let archive failure block /new

    # Clear conversation history
    conversation.clear()
    runtime_owner.set_session_id(conversation.session_id)

    # Clear pinned skills
    get_skill_manager().unpin_all()

    # Reset session stats
    stats = get_stats()
    stats.reset()
    stats.set_active_model(runtime_owner.model_spec.provider, runtime_owner.model_spec.model)

    # Force memory context rebuild on next turn
    _cached_memory_context = None

    cli.print_system("Conversation archived and cleared. Starting fresh.")

def _cmd_tools(cli, **_):
    tools = get_registry().list_tools()
    cli.print_system(f"Registered tools: {tools}")



def _cmd_skills(cli, **_):
    """List all available skills and their status."""
    mgr = get_skill_manager()
    catalog = mgr.get_catalog()
    if not catalog:
        cli.print_system("📦 No skills found. Create skills in the skills/ directory.")
        return
    lines = ["📦 Agent Skills:"]
    for entry in catalog:
        status = "📌" if entry["pinned"] else "📦"
        lines.append(f"  {status} {entry['name']}: {entry['description'][:80]}")
    lines.append(f"\n  Total: {len(catalog)} skills, {sum(1 for e in catalog if e['active'])} active")
    lines.append("  Use '/skill <name>' to activate, '/skill -<name>' to deactivate")
    cli.print_system("\n".join(lines))

def _cmd_skill(cli, user_input, **_):
    """Activate or deactivate a specific skill."""
    parts = user_input.split(None, 1)
    if len(parts) < 2:
        cli.print_error("Usage: /skill <name> (activate) or /skill -<name> (deactivate)")
        return
    
    name = parts[1].strip()
    mgr = get_skill_manager()
    
    if name.startswith("-"):
        # Deactivate
        skill_name = name[1:]
        if mgr.unpin(skill_name):
            cli.print_system(f"📦 Skill '{skill_name}' un-pinned.")
        else:
            cli.print_error(f"Skill '{skill_name}' is not pinned or not found.")
    else:
        # Pin
        if mgr.pin(name):
            cli.print_system(f"📌 Skill '{name}' pinned — its instructions are now in context.")
        else:
            cli.print_error(f"Skill '{name}' not found. Use /skills to list available skills.")


def _cmd_stats(cli, **_):
    """Show detailed session statistics."""
    stats = get_stats()
    model_label = (
        f"{stats.provider}:{stats.model}"
        if stats.provider and stats.model
        else "(none yet)"
    )
    lines = [
        f"Model:         {model_label}",
        f"Turns:         {stats.turns}",
        f"API calls:     {stats.api_calls}",
        f"Tool calls:    {stats.tool_calls}",
        f"Input tokens:  {stats.format_tokens(stats.total_input_tokens)} ({stats.total_input_tokens:,})",
        f"Output tokens: {stats.format_tokens(stats.total_output_tokens)} ({stats.total_output_tokens:,})",
        f"Cache create:  {stats.format_tokens(stats.total_cache_creation_tokens)}",
        f"Cache read:    {stats.format_tokens(stats.total_cache_read_tokens)}",
        f"Total cost:    {stats.format_cost(stats.total_cost)}",
        f"Elapsed:       {stats.elapsed/60:.1f} min",
    ]
    cli.print_system("\n".join(lines))


def _cmd_model(cli, runtime_owner: LLMOwner, user_input: str, **_):
    """Show or switch the active provider:model for future turns."""
    parts = user_input.split()
    if len(parts) == 1:
        cli.print_system(f"Active model: {runtime_owner.label()}")
        return
    if len(parts) < 3:
        cli.print_error("Usage: /model <provider> <model>")
        return
    provider = parts[1].strip()
    model = " ".join(parts[2:]).strip()
    try:
        runtime_owner.switch_model(provider, model)
    except ValueError as err:
        cli.print_error(str(err))
        return
    get_stats().set_active_model(provider, model)
    cli.print_system(f"Switched model to {provider}:{model}")


def _cmd_continue(cli, conversation, runtime_owner, user_input: str = "/continue", **_):
    """Load a saved session and continue where we left off.

    Usage:
        /continue                  resume latest.json (default)
        /continue latest           resume latest.json (explicit)
        /continue <session_id>     resume by session id (full or unique prefix)
        /continue <timestamp>      resume by archive filename / saved_at prefix
                                   (e.g. 20260430_215012)
    """
    parts = user_input.split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""

    if conversation.messages:
        cli.print_system("⚠ Current conversation is not empty. Use /new first to clear, then /continue.")
        return

    target_path = None
    if arg:
        entry = find_session(arg)
        if entry is None:
            cli.print_error(
                f"No session matched '{arg}'. Use /sessions to list available sessions."
            )
            return
        target_path = entry["path"]
    else:
        if not has_saved_session():
            cli.print_system("No saved session found. Start a new conversation.")
            return

    result = load_session(conversation, path=target_path)
    if result.get("loaded"):
        runtime_owner.set_session_id(conversation.session_id)
        sid = result.get("session_id", "")
        sid_short = (sid[:12] + "…") if len(sid) > 13 else sid
        cli.print_system(
            f"✅ Resumed session {sid_short} from {result['saved_at']} "
            f"({result['message_count']} messages, ~{conversation.estimated_tokens()} tokens)"
        )
    else:
        cli.print_system(f"Failed to load session: {result.get('error', 'unknown error')}")


def _cmd_sessions(cli, **_):
    """List saved sessions (newest first)."""
    entries = list_sessions(limit=20)
    if not entries:
        cli.print_system("No saved sessions found.")
        return
    lines = ["💾 Saved sessions (newest first):"]
    for i, e in enumerate(entries, start=1):
        marker = "●" if e["is_latest"] else " "
        sid = e["session_id"] or "(no-id)"
        sid_short = (sid[:12] + "…") if len(sid) > 13 else sid
        meta = e.get("metadata") or {}
        model = ""
        if meta.get("provider") or meta.get("model"):
            model = f"  [{meta.get('provider','?')}:{meta.get('model','?')}]"
        reason = f"  ({meta['reason']})" if meta.get("reason") else ""
        lines.append(
            f"  {marker} {i:>2}. {e['saved_at']:<25} {sid_short:<14} "
            f"{e['message_count']:>3} msgs{model}{reason}"
        )
    lines.append("")
    lines.append("  ● = latest (default for /continue)")
    lines.append("  Resume:  /continue <session_id>   or   /continue <timestamp-prefix>")
    cli.print_system("\n".join(lines))



# Command dispatch table
COMMAND_TABLE = {
    "/help": _cmd_help,
    "/multi": _cmd_multi,
    "/markdown": _cmd_markdown,
    "/history": _cmd_history,
    "/new": _cmd_new,
    "/continue": _cmd_continue,
    "/sessions": _cmd_sessions,
    "/tools": _cmd_tools,
    "/skills": _cmd_skills,
    "/stats": _cmd_stats,
}


# ─── Graceful exit helper ────────────────────────────────────────────────────

def _print_unexpected_error(cli, error: Exception):
    """Render fatal errors without Rich markup parsing dynamic exception text."""
    message = f"Unexpected error in agent: {error}"
    try:
        if cli is not None:
            cli.print_error(message)
        else:
            console.print(f"✖ {message}", style="error", markup=False)
    except Exception:
        print(message, file=sys.stderr)


def _graceful_exit(cli, conversation=None):
    """Print goodbye, save session, and disconnect background services."""
    # Save session before saying goodbye (fast, silent)
    if conversation and conversation.messages:
        try:
            from .runtime.stats import get_stats
            stats = get_stats()
            save_session(conversation, metadata={
                "provider": stats.provider or "",
                "model": stats.model or "",
            })
        except Exception:
            pass  # Don't let session save failure block exit

    cli.goodbye()  # Say goodbye — user sees immediate response

    # Disconnect all MCP servers (kills Chrome, etc.) so they don't linger as stale processes
    try:
        from .mcp.manager import _manager
        if _manager and _manager._clients:
            _manager.disconnect_all()
    except Exception:
        pass


# ─── System prompt builder ───────────────────────────────────────────────────

# Cache the memory portion of the system prompt (does not depend on user query).
# Invalidated when _force_rebuild_context is set (after compaction or memory writes).
_cached_memory_context: str | None = None


def _build_full_system_prompt(user_input: str, skill_mgr: SkillManager,
                              force_rebuild: bool = False) -> str:
    """Build the complete system prompt: base + memory + skill catalog.

    All three components are stable across most turns, so the assembled
    prefix is cache-friendly:
      - ``_base_system_prompt()`` is constant.
      - ``_cached_memory_context`` is rebuilt only on compaction or memory writes.
      - The skill catalog (Stage 1, name+description only) changes only when
        the on-disk skills/ directory changes.

    Pinned skill bodies (Stage 2) are NOT part of the system prompt — the
    caller attaches them as a tail block on the last user message so that
    pin diffs do not invalidate the prefix cache. See
    ``SkillManager.build_pinned_bodies_block``.
    """
    global _cached_memory_context

    if force_rebuild or _cached_memory_context is None:
        _cached_memory_context = build_memory_context(query=user_input) or ""

    base_prompt = _base_system_prompt()
    full_system_prompt = base_prompt
    if _cached_memory_context:
        full_system_prompt = base_prompt + "\n\n" + _cached_memory_context

    catalog = skill_mgr.build_catalog_block()
    if catalog:
        full_system_prompt = full_system_prompt + "\n\n" + catalog

    return full_system_prompt


def _build_compaction_system_prompt(user_input: str) -> str:
    """Build the base+memory prompt prefix used for cache-friendly compaction."""
    global _cached_memory_context

    if _cached_memory_context is None:
        _cached_memory_context = build_memory_context(query=user_input) or ""
    compact_sys_prompt = _base_system_prompt()
    if _cached_memory_context:
        compact_sys_prompt += "\n\n" + _cached_memory_context
    return compact_sys_prompt


# ─── Main agent loop ─────────────────────────────────────────────────────────

def run(runtime_owner: LLMOwner) -> None:
    global _cached_memory_context
    # Initialize before try block to avoid unbound variable risk in outer except
    cli = None
    conversation = None

    try:
        cli = CLI()
        state = {"use_markdown": True}
        stats = get_stats()
        stats.set_active_model(runtime_owner.model_spec.provider, runtime_owner.model_spec.model)

        cli.print_banner(runtime_owner.label())

        conversation = ConversationMemory()
        runtime_owner.set_session_id(conversation.session_id)
        from .tools.subagent import set_runtime_owner
        set_runtime_owner(runtime_owner)

        # Notify if a previous session can be resumed
        if has_saved_session():
            cli.print_system("💾 Previous session available. Type /continue to resume.")

        # Initialize Agent Skills
        skill_mgr = init_skills()
        discovered_skills = skill_mgr.list_skills()
        if discovered_skills:
            cli.print_system(f"📦 Skills loaded: {', '.join(discovered_skills)} ({len(discovered_skills)} total)")
        else:
            cli.print_system("📦 No skills found. Create them in skills/ directory.")

        while True:
            try:
                user_input = cli.get_input()
            except Exception as e:
                cli.print_error(f"Input error: {e}")
                continue

            if user_input is None:
                _graceful_exit(cli, conversation)
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            try:
                # ─── Quit ─────────────────────────────
                if user_input == "/quit":
                    _graceful_exit(cli, conversation)
                    break

                # ─── /skill <name> (prefix match) ─────
                if user_input.startswith("/skill "):
                    _cmd_skill(cli=cli, user_input=user_input)
                    continue

                if user_input.startswith("/model"):
                    _cmd_model(cli=cli, runtime_owner=runtime_owner, user_input=user_input)
                    continue

                if user_input.startswith("/continue "):
                    _cmd_continue(
                        cli=cli,
                        runtime_owner=runtime_owner,
                        conversation=conversation,
                        user_input=user_input,
                    )
                    continue


                # ─── Dispatch table commands ──────────
                handler = COMMAND_TABLE.get(user_input)
                if handler:
                    handler(
                        cli=cli,
                        runtime_owner=runtime_owner,
                        conversation=conversation,
                        state=state,
                        user_input=user_input,
                    )
                    continue

                # ─── Regular interaction ──────────────
                conversation.add_message("user", user_input)

                # Auto-compact check (token-based, with memory re-injection callback)
                def _on_compacted():
                    """Callback after auto-compaction: signal context rebuild."""
                    state["_force_rebuild_context"] = True

                # Pass the same base+memory prefix shape used for normal turns
                # so compaction remains cache-friendly without dropping rules.
                compact_sys_prompt = _build_compaction_system_prompt(user_input)
                summarize_if_needed(
                    conversation, runtime_owner,
                    system_prompt_rebuilder=_on_compacted,
                    system_prompt=compact_sys_prompt,
                )

                messages = conversation.get_history()
                history_len = len(messages)  # snapshot before loop mutates in-place

                # Build system prompt: base + memory(cached) + skill catalog(stable).
                # Active skill bodies are attached separately as a tail block on
                # the last user message (below) so per-turn activation diffs do
                # NOT invalidate the system-prompt prefix cache.
                force_rebuild = state.pop("_force_rebuild_context", False)
                full_system_prompt = _build_full_system_prompt(
                    user_input, skill_mgr,
                    force_rebuild=force_rebuild,
                )

                # Stage 2 — attach pinned skill bodies to the LAST user message
                # as a prepended text block. ``get_history()`` returned a
                # shallow list-copy, but the dicts inside are shared with the
                # ConversationMemory store; clone the last dict before mutating
                # its ``content`` so the persisted history stays clean.
                pinned_bodies = skill_mgr.build_pinned_bodies_block()
                if pinned_bodies and messages and messages[-1].get("role") == "user":
                    last = dict(messages[-1])
                    orig_content = last.get("content", "")
                    if isinstance(orig_content, list):
                        last["content"] = (
                            [{"type": "text", "text": pinned_bodies}] + list(orig_content)
                        )
                    else:
                        last["content"] = pinned_bodies + "\n\n" + str(orig_content)
                    messages[-1] = last

                cli.print_separator()

                sys.stdout.write("\033[1;32mAgent ▶ \033[0m")
                sys.stdout.flush()

                try:
                    # Build LoopConfig inline
                    config = LoopConfig(
                        max_steps=DEFAULT_MAX_STEPS,
                        initial_max_tokens=DEFAULT_MAX_TOKENS,
                        max_tokens_cap=MAX_TOKENS_CAP,
                        auto_scale_on_truncation=True,
                        token_scale_factor=2,
                        concurrent_tools=True,
                        max_tool_workers=4,
                        tool_timeout=DEFAULT_TOOL_TIMEOUT,
                        retry_attempts=10,
                        retry_base_delay=2.0,
                        compact_messages=True,
                        max_working_tokens=MAX_WORKING_TOKENS,
                        compact_tool_result_chars=COMPACT_TOOL_RESULT_CHARS,
                        max_tool_result_chars=MAX_TOOL_RESULT_CHARS,
                        streaming=True,
                        truncate_large_inputs=True,
                        fallback_on_max_steps=True,
                    )

                    # Build streaming callbacks
                    stats = get_stats()
                    stats.new_turn()
                    callbacks, spinner = build_streaming_callbacks(stats, runtime_owner)

                    # Create tool source factory.
                    # ``freeze()`` returns a batch-atomic deep-copy
                    # ``ToolBatch`` snapshot.  The engine builds its own
                    # per-step ToolBatch from this output anyway; freeze
                    # gives us cross-call atomicity for free.
                    registry = get_registry()
                    def _tool_source():
                        batch = registry.freeze()
                        return (list(batch.schemas), dict(batch.functions))
                    tool_source = _tool_source

                    # Create AgentLoop and run
                    loop = AgentLoop(
                        runtime_owner,
                        config,
                        callbacks=callbacks,
                        tool_source=tool_source,
                        session_id=conversation.session_id,
                    )
                    try:
                        result: LoopResult = loop.run(full_system_prompt, messages)
                    except KeyboardInterrupt:
                        spinner.stop()
                        raise
                    finally:
                        spinner.stop()  # ensure spinner is always cleaned up

                    # Handle LoopResult status branches
                    if result.status == "completed":
                        if callbacks._stream_state.needs_newline:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        response = result.text
                        final_text = result.final_text
                        planner_messages = result.messages
                    elif result.status == "max_steps":
                        max_step_msg = f"\n\n⚠️ Reached maximum reasoning steps ({config.max_steps}). My response may be incomplete."
                        sys.stdout.flush()
                        console.print(f"[bold yellow]{max_step_msg}[/bold yellow]")

                        response = result.text or "I've reached my maximum reasoning steps. Please try rephrasing your request."
                        final_text = result.final_text
                        planner_messages = result.messages
                    elif result.status == "interrupted":
                        _interrupted_msg()
                        response = result.text
                        final_text = ""
                        planner_messages = result.messages
                    elif result.status == "error":
                        error_msg = f"\n[Error: {result.error}]"
                        sys.stdout.flush()
                        console.print(error_msg, style="bold red", markup=False)
                        if result.text:
                            response = result.text + f"\n\n[Error: {result.error}]"
                        else:
                            response = f"Error during planning: {result.error}"
                        final_text = ""
                        planner_messages = result.messages
                    elif result.status == "cost_limit":
                        cost_msg = f"\n\n⚠️ {result.error}"
                        sys.stdout.flush()
                        console.print(cost_msg, style="bold yellow", markup=False)
                        response = result.text + cost_msg if result.text else cost_msg
                        final_text = result.final_text
                        planner_messages = result.messages
                    elif result.status == "dedup_break":
                        dedup_msg = f"\n\n⚠️ Loop detected — stopped to prevent infinite loop."
                        sys.stdout.flush()
                        console.print(dedup_msg, style="bold yellow", markup=False)
                        response = result.text + dedup_msg if result.text else dedup_msg
                        final_text = result.final_text
                        planner_messages = result.messages
                    else:
                        # Fallback — should not happen
                        response = result.text or "Unknown error"
                        final_text = ""
                        planner_messages = result.messages

                except KeyboardInterrupt:
                    cli.print_system("\n⚠ Interrupted — returning to prompt.")
                    response = "[Response interrupted by user]"
                    final_text = ""
                    planner_messages = []

                # Invalidate memory cache after planner runs (tools may have written memory)
                _cached_memory_context = None

                # Preserve structured tool_use/tool_result messages from the planner loop.
                # loop.run() mutates messages in-place, so use the pre-loop snapshot.
                new_messages = planner_messages[history_len:]
                if new_messages:
                    conversation.messages.extend(new_messages)
                else:
                    conversation.add_message("assistant", response)

                # Proactive memory extraction (background, non-blocking)
                if should_extract(user_input):
                    from .memory.extraction import _extract_text
                    asst_text = _extract_text(response) if response else ""
                    if asst_text:
                        extract_and_remember(runtime_owner, user_input, asst_text)

                # Render final LLM output (not intermediate tool-use text) with rich markdown
                if state["use_markdown"] and final_text.strip() and not final_text.strip().startswith("["):
                    try:
                        from rich.markdown import Markdown
                        from rich.panel import Panel
                        md = Markdown(final_text, code_theme="monokai")
                        console.print()
                        console.print(Panel(
                            md,
                            title="[bold green]📝 Rendered[/bold green]",
                            border_style="green",
                            padding=(0, 1),
                            subtitle="[dim]/markdown to toggle[/dim]",
                        ))
                    except Exception:
                        pass

                # Show turn stats (tokens + cost)
                cli.print_turn_summary()
                cli.print_separator()

            except KeyboardInterrupt:
                cli.print_system("\n⚠ Interrupted — returning to prompt.")
                continue
            except Exception as e:
                cli.print_error(f"Error: {e}")

    except KeyboardInterrupt:
        try:
            console.print("\n[system]⚠ Interrupted.[/system]")
            if cli and conversation:
                _graceful_exit(cli, conversation)
            else:
                console.print("[system]👋 Goodbye![/system]")
        except Exception:
            console.print("\n[system]👋 Goodbye![/system]")
    except Exception as e:
        _print_unexpected_error(cli, e)
