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
from typing import Any, Callable, Optional


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
    schema_map: dict[str, dict]  # name → schema (also deep-copied)
    functions: dict[str, Callable]  # name → callable
    parallel_safe: frozenset[str]  # tool names with parallel_safe=True
    timeout_hints: dict[str, int]  # name → timeout (seconds)
    large_input_keys: dict[str, frozenset[str]]  # name → keys whose values to truncate
    compaction_priority: dict[str, str]  # name → "ephemeral" | "standard" | "persistent"

    # ─── Convenience accessors (mirror the legacy ToolRegistry shape so
    #     callers can swap registry → batch with no method changes) ────────

    def is_parallel_safe(self, name: str) -> bool:
        return name in self.parallel_safe

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
            schema_map={},
            functions={},
            parallel_safe=frozenset(),
            timeout_hints={},
            large_input_keys={},
            compaction_priority={},
        )

    def with_overlay(
        self,
        *,
        functions: dict[str, Callable] | None = None,
        schemas: list[dict] | None = None,
        parallel_safe: set[str] | None = None,
    ) -> "ToolBatch":
        """Return a new batch with extra tools layered on top.

        Used by the engine to overlay the per-loop ``write_todos`` closure
        (which lives outside the registry to avoid ``ContextVar`` issues
        with the daemon-thread tool executor).  Overlaid tools are added
        with empty metadata unless explicitly specified — they default to
        non-parallel-safe, no timeout hint, standard compaction priority.
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

        return ToolBatch(
            version=self.version,
            schemas=new_schemas,
            schema_map=new_schema_map,
            functions=new_functions,
            parallel_safe=new_parallel,
            timeout_hints=self.timeout_hints,
            large_input_keys=self.large_input_keys,
            compaction_priority=self.compaction_priority,
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
                 compaction_priority: str | None = None) -> None:
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
            return ToolBatch(
                version=self._version,
                schemas=schemas,
                schema_map=schema_map,
                functions=dict(self._functions),
                parallel_safe=parallel_safe,
                timeout_hints=timeout_hints,
                large_input_keys=large_input_keys,
                compaction_priority=compaction_priority,
            )

    def snapshot(self) -> tuple[int, list[dict], dict[str, Callable]]:
        """Legacy shallow snapshot — kept for callers that haven't migrated.

        Returns ``(version, schemas_copy, functions_copy)``.  Prefer
        ``freeze()`` which also captures metadata atomically and deep-copies
        schemas to prevent post-freeze mutation.
        """
        with self._lock:
            return (self._version, list(self._schemas), dict(self._functions))

    # ─── Locked metadata getters (defense-in-depth) ─────────────────────
    #
    # These remain for callers that legitimately need a single live
    # lookup (e.g. ``mcp_manager`` reflection).  All take ``self._lock``
    # so a concurrent ``register()``/``unregister()`` cannot expose a
    # half-updated metadata dict.  The dispatch loop must NOT use these
    # — it uses ``freeze()`` once per step.

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def is_parallel_safe(self, name: str) -> bool:
        with self._lock:
            return self._metadata.get(name, {}).get("parallel_safe", False)

    def get_timeout_hint(self, name: str) -> int | None:
        """Return the tool's preferred timeout in seconds, or None for default."""
        with self._lock:
            return self._metadata.get(name, {}).get("timeout_hint")

    def get_large_input_keys(self, name: str) -> set[str] | None:
        """Return keys whose values should be truncated in working messages, or None."""
        with self._lock:
            return self._metadata.get(name, {}).get("large_input_keys")

    def get_compaction_priority(self, name: str) -> str:
        """Return tool's compaction priority: 'ephemeral', 'standard', or 'persistent'."""
        with self._lock:
            return self._metadata.get(name, {}).get("compaction_priority", "standard")

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
