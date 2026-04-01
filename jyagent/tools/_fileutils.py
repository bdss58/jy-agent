# Shared file helpers: resolve_path, atomic_write, should_skip_dir, is_binary_ext
#
# Extracted to avoid duplicating path/IO logic across core.py and search.py.

import os
import fnmatch
import tempfile

from ..config import SKIP_DIRS, BINARY_EXTS

# Pre-split SKIP_DIRS into exact names and glob patterns (containing wildcards)
_SKIP_EXACT: set[str] = set()
_SKIP_PATTERNS: list[str] = []
for _entry in SKIP_DIRS:
    if any(c in _entry for c in ('*', '?', '[')):
        _SKIP_PATTERNS.append(_entry)
    else:
        _SKIP_EXACT.add(_entry)


def resolve_path(path: str, root: str | None = None) -> str:
    """Resolve a path to absolute, expanding ~ and relative segments.

    If *root* is given, relative paths are resolved against it;
    otherwise the current working directory is used.
    """
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        base = root if root else os.getcwd()
        path = os.path.join(base, path)
    return os.path.abspath(path)


def atomic_write(path: str, content: str, encoding: str = "utf-8") -> None:
    """Write *content* to *path* atomically.

    Creates parent directories as needed.  Writes to a temp file in the
    same directory, flushes, fsyncs, then os.replace()s into place.
    On failure the original file (if any) is left untouched.
    """
    dirname = os.path.dirname(path) or "."
    os.makedirs(dirname, exist_ok=True)

    fd = None
    tmp_path = None
    try:
        fd_int, tmp_path = tempfile.mkstemp(dir=dirname, prefix=".tmp_", suffix=".write")
        fd = os.fdopen(fd_int, "w", encoding=encoding)
        fd.write(content)
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        fd = None  # prevent double-close in finally
        os.replace(tmp_path, path)
        tmp_path = None  # prevent unlink in finally
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


def should_skip_dir(dirname: str) -> bool:
    """Return True if *dirname* should be skipped during traversal.

    Matches exact names (e.g. ``node_modules``) and glob patterns
    (e.g. ``*.egg-info``).  Hidden directories (starting with ``"."``)
    are also skipped.
    """
    if dirname.startswith('.'):
        return True
    if dirname in _SKIP_EXACT:
        return True
    return any(fnmatch.fnmatch(dirname, pat) for pat in _SKIP_PATTERNS)


def is_binary_ext(path: str) -> bool:
    """Return True if *path* has a known binary extension."""
    _, ext = os.path.splitext(path)
    return ext.lower() in BINARY_EXTS
