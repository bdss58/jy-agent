# Agent — Main run loop. Top-level orchestrator.
#
# Command handlers live in jyagent.agent_commands (importing it wires the
# /help, /new, /continue, … handlers into the central command registry).
# System prompt assembly lives in jyagent.system_prompt.

import sys
from dataclasses import dataclass, field
from .runtime.tools.registry import get_registry
import jyagent.tools  # noqa: F401 — triggers tool registration
from .memory import (
    ConversationMemory, summarize_if_needed,
    has_saved_session,
    should_extract, extract_and_remember, extract_text,
)
from .ui.terminal import build_streaming_callbacks
from .ui.loop_result_presenter import present_loop_result
from .runtime.loop.engine import AgentLoop, LoopResult
from .runtime.loop.config import build_default_loop_config
from .ui.cli import CLI, console
from .ui.commands import find_command, get_handler
from .skills import init_skills
from .llm import LLMOwner
from .runtime.stats import get_stats
from .system_prompt import build_system_prompt, invalidate_memory_cache
from .durability import safe_checkpoint as _safe_checkpoint
import jyagent.agent_commands  # noqa: F401 — registers /commands via bind_handler at import time





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



def _checkpoint_turn(conversation) -> None:
    """Per-turn durability wrapper — kept for clarity at call sites."""
    _safe_checkpoint(conversation)


def _graceful_exit(cli, conversation=None):
    """Print goodbye, save session, and disconnect background services."""
    _safe_checkpoint(conversation)

    cli.goodbye()  # Say goodbye — user sees immediate response

    # Disconnect all MCP servers (kills Chrome, etc.) so they don't linger as
    # stale processes. ``disconnect_all()`` is a no-op when no clients are
    # registered — no need to peek at the private ``_clients`` dict.
    try:
        from .mcp import get_manager_if_exists
        mgr = get_manager_if_exists()
        if mgr is not None:
            mgr.disconnect_all()
    except Exception:
        pass



# ─── Run-loop state ──────────────────────────────────────────────────────────

@dataclass
class AgentRunState:
    """Mutable per-session state shared between the run loop and slash-command handlers.

    Replaced an ad-hoc ``{"use_markdown": True}`` dict during the
    LIGHT-CLEANUP follow-up so the contract is typed and the
    inter-turn "force context rebuild" signal isn't a magic string key.

    Fields:
      * ``use_markdown`` — render the final assistant text as Rich Markdown.
        Toggled by the ``/markdown`` slash command.
      * ``force_rebuild_context`` — set True by the auto-compaction callback;
        consumed (and reset) once at the start of the next turn so the system
        prompt is rebuilt against the freshly-compacted memory state.
      * ``last_reasoning_blocks`` — the reasoning/thinking blocks recorded
        during the most recent turn, in order.  Populated from the turn's
        StreamingUI when the engine returns; consumed by the ``/think``
        slash command to re-render the most recent reasoning expanded.
    """
    use_markdown: bool = True
    force_rebuild_context: bool = False
    # Default to empty so /think before any turn returns a clean message.
    # Stored as a list[ReasoningBlock] but typed loosely to avoid pulling
    # the UI module into the agent dataclass at import time.
    last_reasoning_blocks: list = field(default_factory=list)


# ─── Main agent loop ─────────────────────────────────────────────────────────

def run(runtime_owner: LLMOwner) -> None:
    # Initialize before try block to avoid unbound variable risk in outer except
    cli = None
    conversation = None

    try:
        cli = CLI()
        state = AgentRunState()
        stats = get_stats()
        stats.set_active_model(runtime_owner.model_spec.provider, runtime_owner.model_spec.model)

        cli.print_banner(runtime_owner.label())

        conversation = ConversationMemory()
        runtime_owner.set_session_id(conversation.session_id)
        from .tools.subagent import set_runtime_owner
        set_runtime_owner(runtime_owner)

        # Belt-and-braces: even if the process dies via uncaught exception,
        # SIGTERM, or sys.exit, the most recent turn is still on disk because
        # _checkpoint_turn() runs after every message. atexit catches the
        # rare case where a turn finishes but graceful_exit never runs.
        import atexit
        atexit.register(_checkpoint_turn, conversation)

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

        # G1-lite: announce auto-detected project context (AGENTS.md / CLAUDE.md)
        # so the user knows it's been ingested into the system prompt.
        try:
            from .system_prompt import project_context_source
            _proj_ctx_path = project_context_source()
            if _proj_ctx_path:
                cli.print_system(f"📄 Project context loaded: {_proj_ctx_path}")
        except Exception:
            pass

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
                # ─── Quit (handled out-of-band: must break the loop) ──
                if user_input == "/quit":
                    _graceful_exit(cli, conversation)
                    break

                # ─── Slash-command dispatch (registry-driven) ─────────
                cmd = find_command(user_input)
                if cmd is not None:
                    handler = get_handler(cmd.name)
                    if handler is None:
                        cli.print_error(f"Command {cmd.name} has no handler bound.")
                        continue
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
                _checkpoint_turn(conversation)  # durability: persist user input before engine runs

                # Auto-compact check (token-based, with memory re-injection callback)
                def _on_compacted():
                    """Callback after auto-compaction: signal context rebuild."""
                    state.force_rebuild_context = True

                # Pass the same base+memory prefix shape used for normal turns
                # so compaction remains cache-friendly without dropping rules.
                compact_sys_prompt = build_system_prompt(user_input, include_skills=False)
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
                force_rebuild = state.force_rebuild_context
                state.force_rebuild_context = False
                full_system_prompt = build_system_prompt(
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
                    # Build LoopConfig from app defaults (factory keeps wiring in one place)
                    config = build_default_loop_config()

                    # Build streaming callbacks
                    stats = get_stats()
                    stats.new_turn()
                    from .config import (
                        REASONING_SHOW,
                        REASONING_PREVIEW_LINES,
                    )
                    streaming_ui = build_streaming_callbacks(
                        stats,
                        runtime_owner,
                        reasoning_show=REASONING_SHOW,
                        reasoning_preview_lines=REASONING_PREVIEW_LINES,
                    )
                    callbacks = streaming_ui.callbacks
                    spinner = streaming_ui.spinner

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

                    # Map LoopResult → printed banner + persistence triple.
                    presented = present_loop_result(result, config, streaming_ui)
                    response = presented.response
                    final_text = presented.final_text
                    planner_messages = presented.planner_messages

                    # Stash this turn's reasoning blocks for /think to
                    # re-render later.  Always overwrite — /think always
                    # targets the MOST RECENT turn that produced reasoning.
                    state.last_reasoning_blocks = streaming_ui.reasoning_blocks

                except KeyboardInterrupt:
                    cli.print_system("\n⚠ Interrupted — returning to prompt.")
                    response = "[Response interrupted by user]"
                    final_text = ""
                    planner_messages = []
                    # On interrupt, still capture whatever reasoning got
                    # recorded before the cancel — useful for /think.
                    try:
                        state.last_reasoning_blocks = streaming_ui.reasoning_blocks
                    except Exception:
                        pass

                # Invalidate memory cache after planner runs (tools may have written memory)
                invalidate_memory_cache()

                # Preserve structured tool_use/tool_result messages from the planner loop.
                # loop.run() mutates messages in-place, so use the pre-loop snapshot.
                new_messages = planner_messages[history_len:]
                if new_messages:
                    conversation.messages.extend(new_messages)
                else:
                    conversation.add_message("assistant", response)
                _checkpoint_turn(conversation)  # durability: persist after assistant turn

                # Proactive memory extraction (background, non-blocking)
                if should_extract(user_input):
                    asst_text = extract_text(response) if response else ""
                    if asst_text:
                        extract_and_remember(runtime_owner, user_input, asst_text)

                # Render final LLM output (not intermediate tool-use text) with rich markdown
                from .ui.terminal import render_final_text
                render_final_text(final_text, markdown=state.use_markdown)

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
