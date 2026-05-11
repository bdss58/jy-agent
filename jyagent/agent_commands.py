# Slash-command handlers + dispatch wiring.
#
# Each ``_cmd_*`` function implements one ``/command`` from the REPL.  They
# share a ``(cli, runtime_owner, conversation, state, user_input, **_)``
# kwargs contract — callers (``agent.run``) pass everything; each handler
# pulls only what it needs.
#
# The ``bind_handler`` calls at the bottom register the implementations into
# ``jyagent.ui.commands`` (the central registry that also drives ``/help``
# rendering).  Importing this module is what wires them up — agent.py imports
# it once during ``run()`` startup so command dispatch works.
#
# This file also owns ``_safe_checkpoint`` because ``_cmd_new`` needs it; the
# top-level orchestrator (``agent.py``) imports the helper from here for the
# per-turn checkpoint and graceful-exit flows.

from .memory import (
    checkpoint_session, end_session, find_session, has_saved_session,
    list_sessions, load_session,
)
from .ui.commands import bind_handler
from .runtime.stats import get_stats
from .runtime.tools.registry import get_registry
from .skills import get_skill_manager
from .llm import LLMOwner
from .system_prompt import invalidate_memory_cache


# ─── Shared session-checkpoint helper ────────────────────────────────────────

def _safe_checkpoint(conversation, *, reason: str | None = None) -> None:
    """Flush pending events to the session log. Silent on failure.

    Single durability primitive used by per-turn checkpoints, /new, and
    graceful exit. Metadata always carries the active provider:model so
    /sessions can render it; ``reason`` is added only when the caller
    needs to tag the checkpoint (e.g. ``"new"``).
    """
    if not conversation or not conversation.messages:
        return
    try:
        stats = get_stats()
        metadata: dict = {
            "provider": stats.provider or "",
            "model": stats.model or "",
        }
        if reason:
            metadata["reason"] = reason
        checkpoint_session(conversation, metadata=metadata)
    except Exception:
        pass  # Never block a turn / exit on disk hiccup


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
    # End the current session (emit session_end into its log + clear the
    # latest pointer). Past sessions stay discoverable via /sessions and
    # resumable by id; bare /continue won't auto-resume them.
    if conversation.messages:
        try:
            # Flush pending events so the session_end event sits on top of
            # them in the log.
            _safe_checkpoint(conversation, reason="new")
            end_session(conversation, reason="new")
        except Exception:
            pass  # Don't let session-end failure block /new

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
    invalidate_memory_cache()

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
    lines.append(f"\n  Total: {len(catalog)} skills, {sum(1 for e in catalog if e['pinned'])} pinned")
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
        /continue                  resume the latest session (default)
        /continue latest           resume the latest session (explicit)
        /continue <session_id>     resume by session id (full or unique prefix)
        /continue <timestamp>      resume by saved_at prefix (e.g. 20260430_215012)
    """
    parts = user_input.split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""

    if conversation.messages:
        cli.print_system("⚠ Current conversation is not empty. Use /new first to clear, then /continue.")
        return

    if arg:
        entry = find_session(arg)
        if entry is None:
            cli.print_error(
                f"No session matched '{arg}'. Use /sessions to list available sessions."
            )
            return
        query: str | None = entry["session_id"]
    else:
        if not has_saved_session():
            cli.print_system("No saved session found. Start a new conversation.")
            return
        query = None  # use latest pointer

    result = load_session(conversation, query=query)
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


# ─── Dispatch wiring ─────────────────────────────────────────────────────────
#
# Importing this module is the ONLY thing required to make these handlers
# reachable from the REPL.  The registry (jyagent.ui.commands) is the single
# source of truth for both dispatch and help rendering.

bind_handler("/help",      _cmd_help)
bind_handler("/multi",     _cmd_multi)
bind_handler("/markdown",  _cmd_markdown)
bind_handler("/history",   _cmd_history)
bind_handler("/new",       _cmd_new)
bind_handler("/continue",  _cmd_continue)
bind_handler("/sessions",  _cmd_sessions)
bind_handler("/tools",     _cmd_tools)
bind_handler("/skills",    _cmd_skills)
bind_handler("/skill",     _cmd_skill)
bind_handler("/stats",     _cmd_stats)
bind_handler("/model",     _cmd_model)
