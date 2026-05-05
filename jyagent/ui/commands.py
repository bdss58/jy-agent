"""Slash-command registry — single source of truth for the agent's CLI commands.

Both the dispatcher in :mod:`jyagent.agent` and the help renderer in
:mod:`jyagent.ui.cli` consume this registry, so adding a new command
means editing exactly one place.

A :class:`Command` is a dataclass with a stable ``name`` (e.g. ``/help``),
a one-line ``summary`` shown in ``/help``, a ``group`` to bucket it
visually ("General" / "Memory" / "Skills" / ...), an optional
``aliases``, an optional ``prefix`` flag (``/skill foo`` is a prefix
match — anything else is exact), and a ``handler`` callable receiving
keyword args.

Handlers are looked up by name, so the registry can be defined here
*without* importing the heavy agent module — :mod:`jyagent.agent`
attaches handlers at import time via :func:`bind_handler`.

This decouples:
  * the *catalog* (this module)
  * the *implementation* (agent.py command handlers)
  * the *rendering* (cli.py print_help)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# Handler signature: ``handler(**kwargs) -> None``.  Common kwargs include
# ``cli``, ``runtime_owner``, ``conversation``, ``state``, ``user_input``.
# Handlers ignore unknown kwargs via ``**_``.
Handler = Callable[..., None]


@dataclass(frozen=True)
class Command:
    name: str                  # canonical, e.g. "/help"
    summary: str               # one-line description shown in /help
    group: str = "General"     # bucket header in /help
    prefix: bool = False       # True → matches "/name <args>" by prefix
    aliases: tuple[str, ...] = ()
    hidden: bool = False       # exclude from /help (e.g. /quit shown elsewhere)


# Catalog — order here determines /help order within each group.
COMMANDS: tuple[Command, ...] = (
    # General
    Command("/quit",      "Exit the agent",                         group="General", hidden=True),
    Command("/help",      "Show this help message",                 group="General"),
    Command("/history",   "Show last 10 messages",                  group="General"),
    Command("/new",       "Clear conversation and start fresh",     group="General"),
    Command("/continue",  "Resume last saved session (or by id)",   group="General", prefix=True),
    Command("/sessions",  "List saved sessions",                    group="General"),
    Command("/tools",     "List registered tools",                  group="General"),
    Command("/model",     "Show or switch provider/model",          group="General", prefix=True),
    Command("/multi",     "Toggle multi-line input mode",           group="General"),
    Command("/markdown",  "Toggle markdown rendering",              group="General"),
    Command("/stats",     "Show session statistics (tokens, cost)", group="General"),
    # Skills
    Command("/skills",    "List all available skills and status",   group="Skills"),
    Command("/skill",     "Activate (/skill X) or deactivate (/skill -X) a skill",
            group="Skills", prefix=True),
)


# ─── Handler binding ─────────────────────────────────────────────────────────
#
# Handlers are attached at agent import time so this module stays
# import-light.  ``HANDLERS`` is a name → callable map.

HANDLERS: dict[str, Handler] = {}


def bind_handler(name: str, handler: Handler) -> None:
    """Register a handler for command ``name``.  Idempotent overwrite."""
    HANDLERS[name] = handler


def get_handler(name: str) -> Handler | None:
    return HANDLERS.get(name)


# ─── Lookup helpers used by the dispatcher ──────────────────────────────────


def find_command(user_input: str) -> Command | None:
    """Match ``user_input`` against the registry.

    Tries exact match first (``user_input == cmd.name`` or alias).  Falls
    back to prefix match for commands flagged ``prefix=True`` (split on
    first whitespace).  Returns None if no command matches — the caller
    should treat the input as a regular chat turn.
    """
    if not user_input:
        return None
    head = user_input.split(None, 1)[0]

    for cmd in COMMANDS:
        if head == cmd.name or head in cmd.aliases:
            return cmd

    # Prefix-match fallback: user_input starts with "/name "
    for cmd in COMMANDS:
        if not cmd.prefix:
            continue
        if user_input.startswith(cmd.name + " "):
            return cmd

    return None


def commands_by_group() -> dict[str, list[Command]]:
    """Return commands bucketed by group, preserving registration order.

    ``hidden=True`` commands are excluded.  Used by ``CLI.print_help``.
    """
    out: dict[str, list[Command]] = {}
    for cmd in COMMANDS:
        if cmd.hidden:
            continue
        out.setdefault(cmd.group, []).append(cmd)
    return out


__all__ = [
    "Command",
    "COMMANDS",
    "HANDLERS",
    "bind_handler",
    "get_handler",
    "find_command",
    "commands_by_group",
]
