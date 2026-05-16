"""Shared output primitives for the UI package.

Single source of truth for:

* ``console``        — the Rich :class:`~rich.console.Console` singleton
                       used by every UI module (theme-aware, markup-safe).
* ``CUSTOM_THEME``   — the agent's Rich theme (re-exported for callers
                       that need style names).
* ``_STDOUT_LOCK``   — module-shared :class:`threading.Lock` that ALL raw
                       ``sys.stdout.write`` paths in :mod:`jyagent.ui`
                       must acquire.  Rich's ``console.print`` has its
                       own internal lock and is excluded.

Why a dedicated module?
    Historically ``console`` lived in :mod:`jyagent.ui.terminal_renderer`
    and was re-exported from :mod:`jyagent.ui.cli`, so other UI modules
    imported the *output* singleton through the *input* module.  That
    drags the prompt_toolkit input layer into core output code and
    fragments lock discipline — the spinner in :mod:`jyagent.ui.terminal`
    used a private lock that the subagent status thread in
    :mod:`jyagent.ui.subagent_status` did not share, so raw ANSI from
    two daemon threads could interleave mid-line.

    Putting both the console and the lock here gives every UI writer a
    single import target and removes the input-module dependency.

Why is this file ``output.py`` and not ``console.py``?
    The public ``console`` singleton is re-exported from the package
    ``__init__`` (``from jyagent.ui import console``).  Naming the module
    ``console.py`` would shadow the singleton: ``import jyagent.ui.console``
    binds the package attribute (the Console object) on the parent
    package, NOT the submodule, so dotted access becomes confusing.
    Calling the module ``output`` removes the collision.
"""
from __future__ import annotations

import threading

from rich.console import Console
from rich.theme import Theme


# ─── Theme ───────────────────────────────────────────────────────────────────

CUSTOM_THEME = Theme({
    "agent":  "bold green",
    "user":   "bold cyan",
    "system": "yellow",
    "error":  "bold red",
    "dim":    "dim",
    "banner": "bold magenta",
    "tool":   "yellow",
    "info":   "dim cyan",
})


# ─── Singletons ──────────────────────────────────────────────────────────────

console = Console(theme=CUSTOM_THEME, highlight=False)
"""Rich console used by every UI module — themed, markup parsing off by default."""

_STDOUT_LOCK = threading.Lock()
"""Shared lock for raw ``sys.stdout`` writes from the UI package.

All non-Rich writers (spinner threads, char-level stream writers, raw ANSI
emitters) must take this lock so their output cannot interleave mid-line.
Rich's ``console.print`` already serializes internally and does NOT need
to take this lock — but it also must not be held while calling
``console.print``, to avoid double-locking via Rich's internals.
"""


__all__ = ["console", "CUSTOM_THEME", "_STDOUT_LOCK"]
