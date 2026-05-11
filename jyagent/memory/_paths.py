# Memory subsystem — filesystem path bootstrap.
#
# Was previously a single ``ensure_dirs`` function inside ``_index.py``. That
# made Tier-3 (``_journal``) import from Tier-1 just to create its own
# directory, and made every tier module depend on the index for a concern
# (mkdir) that has nothing to do with MEMORY.md.
#
# Now lives in its own neutral helper module so Tier-1/2/3 can each call
# ``ensure_dirs()`` without depending on each other.
#
# Path constants themselves still live in ``jyagent.config`` (late-bound so
# tests can monkey-patch them after import).

from __future__ import annotations

import os

from .. import config as _cfg


def ensure_dirs() -> None:
    """Create the memory / topics / journal directories on disk if missing.

    Idempotent. Safe to call from any tier module before reading or writing.
    """
    os.makedirs(os.path.dirname(_cfg.MEMORY_MD_FILE), exist_ok=True)
    os.makedirs(_cfg.TOPICS_DIR, exist_ok=True)
    os.makedirs(_cfg.JOURNAL_DIR, exist_ok=True)
