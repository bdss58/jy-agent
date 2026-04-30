"""Agent loop runtime: engine, callbacks, config, and supporting helpers.

Lazy engine load.
~~~~~~~~~~~~~~~~~

``AgentLoop`` and ``LoopConfig`` / ``LoopResult`` re-exports from
``.engine`` are now lazy.  Cheap leaf modules (``callbacks``, ``config``,
``llm_client``, ``llm_types``, ``cost``) are imported eagerly because
they have zero transitive engine cost.  Heavy modules (``engine``,
``tool_executor``, ``llm_runner``, ``compaction``, ``step``) are NOT
loaded by ``import jyagent.runtime.loop`` — they load only when a caller
actually touches ``AgentLoop`` (or imports a heavy submodule directly).

Phase modules (``phases``, ``reflection``, ``checkpoint``, ``todos``,
``verification``, ``remediation``, ``tracing``) are still pre-bound as
sub-module attributes via lazy ``__getattr__`` so existing
``from jyagent.runtime.loop import phases`` style imports keep working
without any runtime regression — but they are not eagerly imported any
more.  Each phase module IS a leaf (it does not import the engine), so
this is a no-op in terms of import-side-effects today, but stays
forward-compatible if a phase later grows engine-pulling code.
"""
from .callbacks import LoopCallbacks
from .config import LoopConfig, LoopResult

__all__ = [
    "AgentLoop",        # lazy from .engine
    "LoopConfig",
    "LoopResult",
    "LoopCallbacks",
    "phases",           # lazy
    "reflection",       # lazy
    "checkpoint",       # lazy
    "todos",            # lazy
    "verification",     # lazy
    "remediation",      # lazy
    "tracing",          # lazy
]


_LAZY_ATTRS = {
    "AgentLoop": (".engine", "AgentLoop"),
}

_LAZY_SUBMODULES = (
    "phases",
    "reflection",
    "checkpoint",
    "todos",
    "verification",
    "remediation",
    "tracing",
    # 'engine', 'tool_executor', 'llm_runner', 'compaction', 'step' are also
    # lazily importable via the standard package mechanism — explicit listing
    # is for the dir()/star-import convenience only.
)


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is not None:
        import importlib
        rel, attr = target
        value = getattr(importlib.import_module(rel, __name__), attr)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()))
