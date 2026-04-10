# Agent — Main run loop and command handlers.

import sys
from .registry import get_registry
import jyagent.tools  # noqa: F401 — triggers tool registration
from .memory import (
    ConversationMemory, summarize_if_needed,
    build_memory_context,
    save_session, load_session, has_saved_session,
    should_extract, extract_and_remember,
    record_file_access,
)
from .terminal_ux import build_streaming_callbacks, _interrupted_msg
from .loop_engine import AgentLoop, LoopConfig, LoopResult
from .cli import CLI, console
from .skills import SkillManager, get_skill_manager, init_skills
from .runtime import RuntimeOwner
from .session_stats import get_stats
from .config import (
    DEFAULT_MAX_TOKENS, MAX_TOKENS_CAP, DEFAULT_MAX_STEPS,
    MAX_TOOL_RESULT_CHARS, MAX_WORKING_TOKENS, DEFAULT_TOOL_TIMEOUT,
    COMPACT_TOOL_RESULT_CHARS,
)


# ─── System prompt (externalized from run()) ──────────────────────────────────

SYSTEM_PROMPT = """You are jy-agent, a self-assembled AI agent built by Jianyong, bootstrapped from a single API call.
You have access to tools for running shell commands, reading/writing files, and listing directories.
Think step by step. Use tools when needed to accomplish tasks.
Be helpful, precise, and concise.
Your source code lives in the jyagent/ directory.

CRITICAL BEHAVIORAL PRINCIPLES:

1. TOOL-FIRST PRINCIPLE: Before writing ad-hoc code or workarounds, check what tools are available. Use existing tools (web_fetch, chrome_browser, manage_memory, etc.) directly.

2. HONESTY PRINCIPLE: Never pretend to have looked something up when you have not. If you are answering from training data alone, say so clearly. If the user asks for recent/current information, use your tools (web_fetch, run_shell with curl, etc.) to actually fetch it. Do not fabricate citations, changelogs, or sources. If you are uncertain, say so.

3. TOOL-VERIFICATION REQUIREMENT:
   When the user asks about system state, environment, file contents, available tools,
   connected services (MCP, Chrome, etc.), or past interactions — ALWAYS verify with
   tools before answering. Never answer from memory or training data alone for factual
   claims about the current environment. Say "Let me check..." then use the appropriate
   tool. If the tool call fails, say so rather than falling back to unverified memory.

4. MEMORY AWARENESS: You have a self-use memory system (inspired by Claude Code):
   - MEMORY.md: the index file, always loaded (first 200 lines / 25KB). Keep it concise.
   - Topic files: detailed knowledge in data/memory/topics/<name>.md. Read on-demand with read_file.
   
   Memory workflow:
   - To remember something: use manage_memory(action='remember') or directly write to files.
   - When MEMORY.md grows large: move detailed sections into topic files, keep MEMORY.md as an index.
   - To read topic details: use read_file('data/memory/topics/<name>.md') on demand.
   - To reorganize memory: rewrite MEMORY.md and topic files with write_file.
   - Use manage_memory(action='topic', text='list/read:<name>/write:<name>|<content>/delete:<name>')
   
   IMPORTANT: Memory provides hints and context, but for factual claims about the filesystem,
   environment, system state, available tools, connected services, or agent capabilities,
   ALWAYS verify with tools before presenting as fact. Memory may be stale or inaccurate.

5. SKILLS AWARENESS: You have an Agent Skills system (agentskills.io standard) that provides procedural knowledge.
   Skills auto-activate when your query matches their description, or you can manually control them:
   - /skills — list all available skills and their status
   - /skill <name> — activate a specific skill
   - manage_skills tool — full skill management (list, activate, deactivate, create, etc.)
   Active skills inject their instructions into your context. Use them to follow best practices for specific tasks."""



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

    # Clear conversation history
    conversation.clear()

    # Clear active skills
    get_skill_manager().deactivate_all()

    # Reset session stats
    stats = get_stats()
    stats.reset()
    stats.set_active_model(runtime_owner.model_spec.provider, runtime_owner.model_spec.model)

    # Force memory context rebuild on next turn
    _cached_memory_context = None

    cli.print_system("Conversation cleared. Starting fresh.")

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
        status = "✅" if entry["active"] else "📦"
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
        if mgr.deactivate(skill_name):
            cli.print_system(f"📦 Skill '{skill_name}' deactivated.")
        else:
            cli.print_error(f"Skill '{skill_name}' is not active or not found.")
    else:
        # Activate
        if mgr.activate(name):
            cli.print_system(f"✅ Skill '{name}' activated — its instructions are now in context.")
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


def _cmd_model(cli, runtime_owner: RuntimeOwner, user_input: str, **_):
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


def _cmd_continue(cli, conversation, runtime_owner, **_):
    """Load the last saved session and continue where we left off."""
    if not has_saved_session():
        cli.print_system("No saved session found. Start a new conversation.")
        return

    if conversation.messages:
        cli.print_system("⚠ Current conversation is not empty. Use /new first to clear, then /continue.")
        return

    result = load_session(conversation)
    if result.get("loaded"):
        cli.print_system(
            f"✅ Resumed session from {result['saved_at']} "
            f"({result['message_count']} messages, ~{conversation.estimated_tokens()} tokens)"
        )
    else:
        cli.print_system(f"Failed to load session: {result.get('error', 'unknown error')}")


# Command dispatch table
COMMAND_TABLE = {
    "/help": _cmd_help,
    "/multi": _cmd_multi,
    "/markdown": _cmd_markdown,
    "/history": _cmd_history,
    "/new": _cmd_new,
    "/continue": _cmd_continue,
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
            from .session_stats import get_stats
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
        from .mcp_manager import _manager
        if _manager and _manager._clients:
            _manager.disconnect_all()
    except Exception:
        pass


# ─── System prompt builder ───────────────────────────────────────────────────

# Cache the memory portion of the system prompt (does not depend on user query).
# Invalidated when _force_rebuild_context is set (after compaction or memory writes).
_cached_memory_context: str | None = None


def _build_full_system_prompt(user_input: str, skill_mgr: SkillManager, runtime_owner: RuntimeOwner,
                              force_rebuild: bool = False,
                              recent_messages: list | None = None) -> str:
    """Build the complete system prompt with memory, skills, and verification context.

    Memory context is cached between turns (invalidated by force_rebuild).
    Skills context is always rebuilt — diff-based routing re-evaluates every turn.
    """
    global _cached_memory_context

    if force_rebuild or _cached_memory_context is None:
        _cached_memory_context = build_memory_context(query=user_input) or ""

    full_system_prompt = SYSTEM_PROMPT
    if _cached_memory_context:
        full_system_prompt = SYSTEM_PROMPT + "\n\n" + _cached_memory_context

    # Skills: diff-based routing with conversation context for multi-turn continuity
    skills_context = skill_mgr.build_prompt_context(
        query=user_input, runtime_owner=runtime_owner,
        recent_messages=recent_messages,
    )
    if skills_context:
        full_system_prompt = full_system_prompt + "\n\n" + skills_context

    return full_system_prompt


# ─── Main agent loop ─────────────────────────────────────────────────────────

def run(runtime_owner: RuntimeOwner) -> None:
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

                # Pass system_prompt so compaction can reuse it (cache-friendly).
                # Use cached system prompt if available, otherwise the base prompt.
                compact_sys_prompt = _cached_memory_context or SYSTEM_PROMPT
                summarize_if_needed(
                    conversation, runtime_owner,
                    system_prompt_rebuilder=_on_compacted,
                    system_prompt=compact_sys_prompt,
                )

                messages = conversation.get_history()
                history_len = len(messages)  # snapshot before loop mutates in-place

                # Build system prompt (memory cached, skills diff-routed per turn)
                # Filter to user/assistant only (skip tool messages) and exclude
                # the current query (already passed as `query` to the router).
                # Take last 4 conversational messages ≈ 2 full prior exchanges.
                force_rebuild = state.pop("_force_rebuild_context", False)
                conv_messages = [
                    m for m in messages[:-1]
                    if m.get("role") in ("user", "assistant")
                ][-4:]  # last 4 = ~2 prior turns
                full_system_prompt = _build_full_system_prompt(
                    user_input, skill_mgr, runtime_owner,
                    force_rebuild=force_rebuild,
                    recent_messages=conv_messages or None,
                )

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
                        retry_attempts=2,
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

                    # Create tool source factory
                    registry = get_registry()
                    tool_source = lambda: registry.snapshot()[1:]  # (schemas, functions) — skip version

                    # Create AgentLoop and run
                    loop = AgentLoop(runtime_owner, config, callbacks=callbacks, tool_source=tool_source)
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
