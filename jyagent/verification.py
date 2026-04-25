"""Backward-compat shim — module moved to ``jyagent.runtime.loop.verification``.

Imports re-export the live module so singletons (registries, stats counters,
etc.) stay singleton across both import paths.
"""
import warnings as _warnings

from jyagent.runtime.loop.verification import *  # noqa: F401,F403
from jyagent.runtime.loop.verification import __dict__ as _new_dict  # noqa: F401

# Re-export private names too (some callers reach in for `_lookup_pricing`,
# `_extract_text`, etc.). This mirrors the module wholesale.
globals().update({k: v for k, v in _new_dict.items() if not k.startswith("__")})

_warnings.warn(
    "jyagent.verification has moved to jyagent.runtime.loop.verification; update your imports.",
    DeprecationWarning,
    stacklevel=2,
)
