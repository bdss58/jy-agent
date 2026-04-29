"""Tool registry + per-step immutable snapshot (``ToolBatch``).

The registry is a long-lived, mutable singleton: tools register at startup and
MCP tools register/unregister dynamically as servers connect.  The dispatch
loop in ``runtime.loop.engine`` runs concurrently with potential registrations.

Historically the engine consulted ``ToolRegistry`` *live* throughout a step
(``is_parallel_safe``, ``get_timeout_hint``, ``get_schema``, ...), with a
plain ``dict.get`` and no lock.  Codex review 2026-04-25 (Part 1 findings #4,
#11, #12) flagged three resulting bugs:

  * The ``parallel_safe`` flag could flip mid-batch, so the same step could
    partition a tool as serial in one place and parallel in another.
  * ``get_schema`` was a live unlocked lookup, so validation could pair an
    *old* function (from the per-step ``functions`` snapshot) with a *new*
    schema, or no schema at all.
  * ``snapshot()`` returned the raw schema dicts by reference — a caller
    that retained a registered schema dict could mutate it post-freeze
    and the dispatch loop would see the change.

``ToolBatch`` (this module) is the fix: a frozen, immutable snapshot built
once per loop step, deep-copied under the registry lock.  All dispatch and
compaction code consumes the ``ToolBatch``; no helper reads the live
registry mid-step.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional


def _readonly(d: Mapping) -> Mapping:
    """Wrap ``d`` in a ``MappingProxyType`` for ToolBatch fields.

    B2 fix (codex review 2026-04-25): ``@dataclass(frozen=True)`` only
    freezes the *field references* on a ToolBatch instance — the dicts
    they point to are still mutable.  We deep-copy at freeze() time, so
    cross-batch mutation is already blocked, but a single batch's
    schema_map / functions / timeout_hints / etc. used to be writeable
    by any caller that held the batch reference.  ``MappingProxyType``
    is a zero-cost read-only view that raises ``TypeError`` on every
    mutating method (``__setitem__``, ``pop``, ``update``, ``clear``).

    Re-wrapping an existing ``MappingProxyType`` is fine and still O(1).
    """
    return MappingProxyType(d) if not isinstance(d, MappingProxyType) else d


@dataclass(frozen=True)
class ToolBatch:
    """Immutable per-step snapshot of all tool metadata.

    Built once per dispatch step (``ToolRegistry.freeze()``), consumed by the
    entire dispatch + compaction pipeline for that step.  Eliminates the
    live-registry-read race where a tool's ``parallel_safe`` flag, schema,
    or timeout hint could change between two reads in the same batch.

    Compaction and historical-message lookups (``get_compaction_priority``,
    ``get_large_input_keys``) gracefully degrade to safe defaults
    (``"standard"``, ``None``) for tool names not in the batch — e.g. tools
    that were unregistered between the step that called them and a later
    step that compacts their results.  This is acceptable degradation: the
    optimisation is lost but no behaviour is incorrect.
    """

    version: int  # registry version when frozen; -1 for ad-hoc batches (e.g. tool_source)
    schemas: tuple[dict, ...]  # deep-copied; safe to share across threads
    schema_map: Mapping[str, dict]  # name → schema (read-only view via MappingProxyType)
    functions: Mapping[str, Callable]  # name → callable (read-only view)
    parallel_safe: frozenset[str]  # tool names with parallel_safe=True
    timeout_hints: Mapping[str, int]  # name → timeout (seconds) — read-only view
    large_input_keys: Mapping[str, frozenset[str]]  # name → keys to truncate — read-only
    compaction_priority: Mapping[str, str]  # read-only: "ephemeral" | "standard" | "persistent"
    # Names of tools that perform externally-observable side effects (filesystem
    # writes, shell commands, sub-process spawns, sub-agent dispatches, MCP
    # calls).  Surfaced by ``is_mutating(name)`` so the dispatch loop can
    # classify timeouts: a mutating-tool timeout leaks a daemon thread whose
    # partial side effect is now invisible to the model, and the loop records
    # the name in ``LoopResult.partial_side_effects`` for outer layers to
    # reconcile.  A1 fix (codex review 2026-04-25): before this, every timeout
    # returned the same "consider smaller steps" error regardless of whether
    # the tool was read-only or mutating, so the model would happily retry a
    # half-completed edit / shell script.
    mutating: frozenset[str]  # tool names with mutating=True

    # ─── Convenience accessors (mirror the legacy ToolRegistry shape so
    #     callers can swap registry → batch with no method changes) ────────

    def is_parallel_safe(self, name: str) -> bool:
        return name in self.parallel_safe

    def is_mutating(self, name: str) -> bool:
        """Return True if the tool is flagged as having side effects.

        Used by the dispatch loop's timeout path: a mutating-tool timeout is
        treated as a potential-partial-effect event (warning logged, name
        appended to ``LoopResult.partial_side_effects``) because the daemon
        thread running the tool body keeps running in the background after
        we report the timeout — the operation may complete invisibly.

        Unknown names default to False: an overlaid or sub-source tool that
        was not registered through the canonical metadata pipeline is
        treated as read-only (safer to under-warn than to spam warnings on
        benign tools that happened to slip past metadata registration).
        """
        return name in self.mutating

    def get_timeout_hint(self, name: str) -> int | None:
        return self.timeout_hints.get(name)

    def get_large_input_keys(self, name: str) -> frozenset[str] | None:
        return self.large_input_keys.get(name)

    def get_compaction_priority(self, name: str) -> str:
        return self.compaction_priority.get(name, "standard")

    def get_function(self, name: str) -> Optional[Callable]:
        return self.functions.get(name)

    def get_schema(self, name: str) -> Optional[dict]:
        return self.schema_map.get(name)

    def list_tools(self) -> list[str]:
        return sorted(self.functions.keys())

    @classmethod
    def empty(cls) -> "ToolBatch":
        """An empty batch — useful for tests and edge cases (no tools available)."""
        return cls(
            version=-1,
            schemas=(),
            schema_map=_readonly({}),
            functions=_readonly({}),
            parallel_safe=frozenset(),
            timeout_hints=_readonly({}),
            large_input_keys=_readonly({}),
            compaction_priority=_readonly({}),
            mutating=frozenset(),
        )

    @classmethod
    def from_source(
        cls,
        src_schemas: list[dict],
        src_functions: dict[str, Callable],
        *,
        base: "ToolBatch",
    ) -> "ToolBatch":
        """Build a batch from an external tool source (e.g. sub-agent) while
        inheriting metadata classification (parallel_safe, mutating, timeout
        hints, large_input_keys, compaction_priority) from ``base``.

        Schemas are deep-copied; all maps are wrapped in MappingProxyType.
        Mirrors the safety properties of ``ToolRegistry.freeze()`` for the
        non-registry source path: callers that retain a reference to the
        passed-in ``src_schemas`` list element cannot mutate the snapshot
        through shared reference, and the resulting ``functions`` /
        ``schema_map`` views raise ``TypeError`` on assignment.

        ``base`` supplies the metadata maps verbatim — they are already
        ``MappingProxyType`` views (from ``freeze()``) so re-using them
        is O(1) and safe.  ``version=base.version`` so the engine's
        per-step ``registry.version != state.tools_batch.version`` cache
        check still semantically tracks "registry has changed since last
        freeze" (B-2 fix from 2026-04-29 review).
        """
        schema_map = {
            s.get("name"): copy.deepcopy(s)
            for s in src_schemas
            if s.get("name")
        }
        # Schemas tuple uses the deep-copied map values so mutating an
        # entry in src_schemas after this call has no effect on the
        # snapshot.
        schemas = tuple(schema_map[name] for name in schema_map)
        return cls(
            version=base.version,
            schemas=schemas,
            schema_map=_readonly(schema_map),
            functions=_readonly(dict(src_functions)),
            parallel_safe=base.parallel_safe,
            timeout_hints=base.timeout_hints,
            large_input_keys=base.large_input_keys,
            compaction_priority=base.compaction_priority,
            mutating=base.mutating,
        )

    def with_overlay(
        self,
        *,
        functions: dict[str, Callable] | None = None,
        schemas: list[dict] | None = None,
        parallel_safe: set[str] | None = None,
        mutating: set[str] | None = None,
    ) -> "ToolBatch":
        """Return a new batch with extra tools layered on top.

        Used by the engine to overlay the per-loop ``write_todos`` closure
        (which lives outside the registry to avoid ``ContextVar`` issues
        with the daemon-thread tool executor).  Overlaid tools are added
        with empty metadata unless explicitly specified — they default to
        non-parallel-safe, no timeout hint, standard compaction priority.

        ``mutating`` (M-5, codex review 2026-04-29): explicit set of tool
        names to flag as mutating in the new batch.  Used by tests and
        by future MCP integrations that need to surface a freshly-added
        side-effecting tool to the verification gate / partial-side-effects
        accumulator without going through the canonical registry path.
        """
        new_functions = dict(self.functions)
        if functions:
            new_functions.update(functions)

        new_schema_map = dict(self.schema_map)
        if schemas:
            for s in schemas:
                name = s.get("name")
                if name:
                    new_schema_map[name] = s

        new_schemas = tuple(self.schemas) + (tuple(schemas) if schemas else ())

        new_parallel = self.parallel_safe
        if parallel_safe:
            new_parallel = frozenset(self.parallel_safe | parallel_safe)

        new_mutating = self.mutating
        if mutating:
            # Union into the existing mutating set rather than replacing
            # it so a caller adding a new mutator doesn't accidentally
            # drop pre-existing mutating classifications.
            new_mutating = frozenset(self.mutating | mutating)

        return ToolBatch(
            version=self.version,
            schemas=new_schemas,
            schema_map=_readonly(new_schema_map),
            functions=_readonly(new_functions),
            parallel_safe=new_parallel,
            timeout_hints=self.timeout_hints,
            large_input_keys=self.large_input_keys,
            compaction_priority=self.compaction_priority,
            # Overlaid tools default to non-mutating unless ``mutating=``
            # is supplied (M-5).  The two in-tree overlays
            # (``write_todos``, verification-context injections) are
            # local scratchpad ops with no externally-observable side
            # effects, so the default is correct for every current
            # caller that doesn't pass ``mutating``.
            mutating=new_mutating,
        )


class ToolRegistry:
    """Mutable, thread-safe registry of agent tools.

    All public read/write methods take ``self._lock``.  In particular,
    metadata getters (``is_parallel_safe``, ``get_timeout_hint``, ...) lock
    even though they read a single dict entry: under concurrent
    ``register()``/``unregister()`` an unlocked read can briefly see a
    half-updated ``self._metadata`` dict.

    For per-step dispatch, prefer ``freeze()`` over per-call lookups —
    ``freeze()`` produces an immutable ``ToolBatch`` consumed by the entire
    step, eliminating mid-step races on metadata (Codex review 2026-04-25).

    The per-call metadata getters that used to live here (``snapshot``,
    ``is_parallel_safe``, ``is_mutating``, ``get_timeout_hint``,
    ``get_large_input_keys``, ``get_compaction_priority``) were deprecated
    in 2026-04-27 (Codex review Part 1 #11) and removed in 2026-04-28 once
    all in-tree callers had migrated to ``ToolBatch``.  Single-lookup
    methods (:meth:`get_function`, :meth:`get_schema`, :meth:`get_schemas`,
    :meth:`get_functions`, :meth:`list_tools`, and :attr:`version`) remain
    — they are fine for one-off live reads where cross-call atomicity
    does not matter.
    """

    def __init__(self):
        self._functions: dict[str, Callable] = {}
        self._schemas: list[dict] = []
        self._schema_map: dict[str, dict] = {}
        self._metadata: dict[str, dict] = {}
        self._version: int = 0
        self._lock = threading.Lock()

    def register(self, name: str, fn: Callable, schema: dict, *,
                 parallel_safe: bool = False,
                 timeout_hint: int | None = None,
                 large_input_keys: set[str] | None = None,
                 compaction_priority: str | None = None,
                 mutating: bool = False) -> None:
        with self._lock:
            self._functions[name] = fn
            self._schema_map[name] = schema
            self._schemas = list(self._schema_map.values())
            meta: dict[str, Any] = {"parallel_safe": parallel_safe}
            if timeout_hint is not None:
                meta["timeout_hint"] = timeout_hint
            if large_input_keys:
                meta["large_input_keys"] = large_input_keys
            if compaction_priority:
                meta["compaction_priority"] = compaction_priority
            # ``mutating`` is stored only when True: keeps the metadata
            # dict compact (most tools are read-only) and matches the
            # "present means yes" convention used by the freeze() path
            # for every other boolean-like flag.
            if mutating:
                meta["mutating"] = True
            self._metadata[name] = meta
            self._version += 1

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name in self._functions:
                del self._functions[name]
                del self._schema_map[name]
                self._metadata.pop(name, None)
                self._schemas = list(self._schema_map.values())
                self._version += 1
                return True
            return False

    # ─── Atomic snapshot ─────────────────────────────────────────────────

    def freeze(self) -> ToolBatch:
        """Return an immutable, deep-copied snapshot of the entire registry.

        This is the **canonical** read API for the dispatch loop — call it
        once per step and pass the resulting ``ToolBatch`` to all helpers.

        Schemas are deep-copied so a caller that retained a schema dict
        from ``register()`` cannot mutate the snapshot through shared
        reference (Codex Part 1 #12).  This is more expensive than the
        old shallow ``snapshot()``, but tool registration is rare and
        per-step freeze cost is negligible compared to the LLM call.
        """
        with self._lock:
            schema_map = {name: copy.deepcopy(s) for name, s in self._schema_map.items()}
            schemas = tuple(schema_map[name] for name in self._schema_map)
            parallel_safe = frozenset(
                name for name, meta in self._metadata.items()
                if meta.get("parallel_safe", False)
            )
            timeout_hints = {
                name: meta["timeout_hint"]
                for name, meta in self._metadata.items()
                if "timeout_hint" in meta
            }
            large_input_keys = {
                name: frozenset(meta["large_input_keys"])
                for name, meta in self._metadata.items()
                if "large_input_keys" in meta
            }
            compaction_priority = {
                name: meta["compaction_priority"]
                for name, meta in self._metadata.items()
                if "compaction_priority" in meta
            }
            mutating = frozenset(
                name for name, meta in self._metadata.items()
                if meta.get("mutating", False)
            )
            return ToolBatch(
                version=self._version,
                schemas=schemas,
                schema_map=_readonly(schema_map),
                functions=_readonly(dict(self._functions)),
                parallel_safe=parallel_safe,
                timeout_hints=_readonly(timeout_hints),
                large_input_keys=_readonly(large_input_keys),
                compaction_priority=_readonly(compaction_priority),
                mutating=mutating,
            )

    # ─── Single-lookup live reads (NOT batch-atomic; only safe when
    #     atomicity across multiple reads doesn't matter — for cross-call
    #     atomicity, use ``freeze()`` and read from the ToolBatch) ──────

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def get_function(self, name: str) -> Optional[Callable]:
        with self._lock:
            return self._functions.get(name)

    def get_schemas(self) -> list[dict]:
        with self._lock:
            return self._schemas.copy()

    def get_schema(self, name: str) -> Optional[dict]:
        with self._lock:
            return self._schema_map.get(name)

    def get_functions(self) -> dict[str, Callable]:
        with self._lock:
            return self._functions.copy()

    def list_tools(self) -> list[str]:
        with self._lock:
            return sorted(self._functions.keys())


_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _registry


__all__ = ["ToolRegistry", "ToolBatch", "get_registry"]
