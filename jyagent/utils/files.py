"""Filesystem helpers shared across subpackages.

Currently hosts ``atomic_write`` — extracted from ``tools/core.py``
(2026-05-06) because both ``tools`` and ``memory`` need durable on-disk
writes for MEMORY.md / topics / journal / file edits.  Living in
``tools/core.py`` would have meant ``memory/`` reached upward into
``tools/`` for a non-tool-specific helper.
"""
from __future__ import annotations

import os
import stat as _stat
import tempfile

__all__ = ["atomic_write"]


def atomic_write(path: str, content: str, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``path`` atomically.

    Strategy: write to a sibling tempfile in the same directory (so
    ``os.replace`` is a same-filesystem rename), ``fsync``, then rename
    over the target.  Preserves the original file's permission bits when
    the target already exists (``mkstemp`` defaults to ``0o600``, which
    would otherwise silently tighten user-readable files).

    On any exception the tempfile is cleaned up so we don't leave
    ``.tmp_*.write`` debris behind.
    """
    dirname = os.path.dirname(path) or "."
    os.makedirs(dirname, exist_ok=True)

    # Preserve original file permissions (mkstemp defaults to 0o600).
    original_mode = None
    try:
        original_mode = os.stat(path).st_mode
    except FileNotFoundError:
        pass

    fd = None
    tmp_path = None
    try:
        fd_int, tmp_path = tempfile.mkstemp(dir=dirname, prefix=".tmp_", suffix=".write")
        fd = os.fdopen(fd_int, "w", encoding=encoding)
        fd.write(content)
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        fd = None
        if original_mode is not None:
            os.chmod(tmp_path, _stat.S_IMODE(original_mode))
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if fd is not None:
            try:
                fd.close()
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
