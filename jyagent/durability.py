"""Session durability helpers — kept separate from command registration.

``_safe_checkpoint`` is the single primitive that flushes the conversation
event log to disk.  It is used in three places:

  * ``agent.py`` per-turn (after every assistant message + on graceful exit)
  * ``agent_commands.py`` from the ``/new`` slash command
  * ``agent.py`` from ``atexit`` (belt-and-braces durability)

Previously this lived inside ``agent_commands.py``, which meant ``agent.py``
imported the commands module for the side effect of registering slash
commands AND for the checkpoint helper.  Pulling the helper into its own
module clarifies the dependency: durability does not depend on the command
registry.
"""

from __future__ import annotations

from .memory import checkpoint_session
from .runtime.stats import get_stats


def safe_checkpoint(conversation, *, reason: str | None = None) -> None:
    """Flush pending events to the session log.  Silent on failure.

    Single durability primitive used by per-turn checkpoints, ``/new`` and
    graceful exit.  Metadata always carries the active provider:model so
    ``/sessions`` can render it; ``reason`` is added only when the caller
    needs to tag the checkpoint (e.g. ``"new"``).

    Never raises — a disk hiccup must not block a turn or exit.
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


__all__ = ["safe_checkpoint"]
