# Agent — Main run loop, command handlers, session lifecycle.

import json
import os
import sys
from .tools import set_client
from .registry import get_registry
from .self_memory import (
    ConversationMemory, PersistentMemory, summarize_if_needed,
    build_memory_context, on_session_start, on_session_end,
    compact_conversation,
)
from .planner import plan_next_action
from .cli import CLI, console
from .skills import SkillManager, get_skill_manager, init_skills


# ─── System prompt (externalized from run()) ──────────────────────────────────

SYSTEM_PROMPT = """You are jy-agent, a self-assembled AI agent built by Jianyong, bootstrapped from a single API call.
You have access to tools for running shell commands, reading/writing files, listing directories, and evolving your own source code.
You can also create NEW tools at runtime using the add_tool function — use it when you need a capability you don't have.
Think step by step. Use tools when needed to accomplish tasks.
Be helpful, precise, and concise.
You can improve yourself: use edit_file to modify your source code in jyagent/, then use evolve_self to validate and hot-reload the changed module.
Your source code lives in the jyagent/ directory.

CRITICAL BEHAVIORAL PRINCIPLES:

1. TOOL-FIRST PRINCIPLE: Before writing ad-hoc code or workarounds, check what tools are available. Use existing tools (web_fetch, chrome_browser, manage_memory, etc.) directly. If a needed capability doesn't exist as a tool, create it with add_tool so it persists for future use.

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
   - Session summaries: recent session history, always loaded.
   
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

def _cmd_clear(cli, conversation, **_):
    conversation.clear()
    cli.print_system("Conversation history cleared.")

def _cmd_tools(cli, **_):
    tools = get_registry().list_tools()
    cli.print_system(f"Registered tools: {tools}")


def _cmd_compact(cli, client, conversation, user_input, state, **_):
    """Manual /compact command — like Claude Code's /compact [instruction]."""
    # Extract optional custom instruction
    parts = user_input.split(None, 1)
    custom_instruction = parts[1] if len(parts) > 1 else ""

    if len(conversation) < 4:
        cli.print_system("⚡ Nothing to compact — conversation is too short.")
        return

    from .self_memory import estimate_conversation_tokens
    est_tokens = estimate_conversation_tokens(conversation.messages)
    cli.print_system(
        f"⚡ Compacting conversation (~{est_tokens} tokens, {len(conversation)} messages)..."
    )

    result = compact_conversation(
        conversation, client,
        custom_instruction=custom_instruction,
    )

    if result.get("compacted"):
        cli.print_system(
            f"✅ Compacted: ~{result['before_tokens']} → ~{result['after_tokens']} tokens "
            f"(saved ~{result['before_tokens'] - result['after_tokens']} tokens)"
        )
        # Re-inject memory and skills (like Claude Code's CLAUDE.md re-injection)
        state["_force_rebuild_context"] = True
    elif result.get("error"):
        cli.print_error(f"Compact failed: {result['error']}")
    else:
        cli.print_system("Nothing to compact.")

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


# Command dispatch table
COMMAND_TABLE = {
    "/help": _cmd_help,
    "/multi": _cmd_multi,
    "/markdown": _cmd_markdown,
    "/history": _cmd_history,
    "/clear": _cmd_clear,
    "/tools": _cmd_tools,
    "/skills": _cmd_skills,
}


# ─── Graceful exit helper ────────────────────────────────────────────────────

def _graceful_exit(client, conversation, cli):
    """Save session and print goodbye on exit. Fast — no API calls."""
    cli.goodbye()  # Say goodbye FIRST — user sees immediate response
    try:
        on_session_end(client, conversation.get_history())  # File I/O only, no API
    except Exception:
        pass


# ─── System prompt builder ───────────────────────────────────────────────────

def _build_full_system_prompt(user_input: str, skill_mgr: SkillManager) -> str:
    """Build the complete system prompt with memory, skills, and verification context.

    This is extracted as a function so it can be called:
    1. On every regular interaction
    2. After compaction (to re-inject MEMORY.md and skills fresh from disk)

    Like Claude Code's behavior: "CLAUDE.md fully survives compaction —
    after /compact, Claude re-reads CLAUDE.md from disk and re-injects it fresh."
    """
    # Re-read memory from disk (ensures fresh state after compaction)
    memory_context = build_memory_context(query=user_input)
    full_system_prompt = SYSTEM_PROMPT
    if memory_context:
        full_system_prompt = SYSTEM_PROMPT + "\n\n" + memory_context

    # Re-read skills (ensures fresh state after compaction)
    # Skills context uses official agentskills.io XML format (<available_skills>, <active_skill>)
    skills_context = skill_mgr.build_prompt_context(query=user_input)
    if skills_context:
        full_system_prompt = full_system_prompt + "\n\n" + skills_context

    return full_system_prompt


# ─── Main agent loop ─────────────────────────────────────────────────────────

def run(client) -> None:
    # Initialize before try block to avoid unbound variable risk in outer except
    cli = None
    conversation = None

    try:
        cli = CLI()
        state = {"use_markdown": True}

        cli.print_banner()

        conversation = ConversationMemory()
        persistent = PersistentMemory()
        set_client(client)
        interaction_count = 0

        # Start session and load memory context
        on_session_start()
        memory_context = build_memory_context()
        if memory_context:
            cli.print_system("🧠 Memory loaded from previous sessions.")

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
                _graceful_exit(client, conversation, cli)
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            try:
                # ─── Quit ─────────────────────────────
                if user_input == "/quit":
                    _graceful_exit(client, conversation, cli)
                    break

                # ─── /skill <name> (prefix match) ─────
                if user_input.startswith("/skill "):
                    _cmd_skill(cli=cli, user_input=user_input)
                    continue

                # ─── /compact [instruction] (prefix match) ─────
                if user_input == "/compact" or user_input.startswith("/compact "):
                    _cmd_compact(
                        cli=cli, client=client, conversation=conversation,
                        user_input=user_input, state=state,
                    )
                    continue

                # ─── Dispatch table commands ──────────
                handler = COMMAND_TABLE.get(user_input)
                if handler:
                    handler(
                        cli=cli,
                        client=client,
                        conversation=conversation,
                        persistent=persistent,
                        state=state,
                        user_input=user_input,
                    )
                    continue

                # ─── Regular interaction ──────────────
                interaction_count += 1
                conversation.add_message("user", user_input)

                # Auto-compact check (token-based, with memory re-injection callback)
                def _on_compacted():
                    """Callback after auto-compaction: signal context rebuild."""
                    state["_force_rebuild_context"] = True

                summarize_if_needed(
                    conversation, client,
                    system_prompt_rebuilder=_on_compacted,
                )

                messages = conversation.get_history()

                # Build system prompt (re-reads memory & skills from disk each time)
                full_system_prompt = _build_full_system_prompt(user_input, skill_mgr)

                # Clear the rebuild flag if it was set
                state.pop("_force_rebuild_context", None)

                cli.print_separator()

                sys.stdout.write("\033[1;32mAgent ▶ \033[0m")
                sys.stdout.flush()

                try:
                    response = plan_next_action(client, messages, full_system_prompt)
                except KeyboardInterrupt:
                    cli.print_system("\n⚠ Interrupted — returning to prompt.")
                    response = "[Response interrupted by user]"

                conversation.add_message("assistant", response)

                # Render with rich markdown if enabled
                if state["use_markdown"] and response.strip() and not response.strip().startswith("["):
                    try:
                        from rich.markdown import Markdown
                        from rich.panel import Panel
                        md = Markdown(response, code_theme="monokai")
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
                _graceful_exit(client, conversation, cli)
            else:
                console.print("[system]👋 Goodbye![/system]")
        except Exception:
            console.print("\n[system]👋 Goodbye![/system]")
    except Exception as e:
        console.print(f"[error]Unexpected error in agent: {e}[/error]")
