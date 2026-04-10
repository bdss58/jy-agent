import threading
from typing import Any, Callable, Optional


class ToolRegistry:
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
                 dedup_exempt: bool = False) -> None:
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
            if dedup_exempt:
                meta["dedup_exempt"] = True
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

    def snapshot(self) -> tuple[int, list[dict], dict[str, Callable]]:
        """Return (version, schemas_copy, functions_copy) atomically."""
        with self._lock:
            return (self._version, list(self._schemas), dict(self._functions))

    @property
    def version(self) -> int:
        return self._version

    def is_parallel_safe(self, name: str) -> bool:
        return self._metadata.get(name, {}).get("parallel_safe", False)

    def get_timeout_hint(self, name: str) -> int | None:
        """Return the tool's preferred timeout in seconds, or None for default."""
        return self._metadata.get(name, {}).get("timeout_hint")

    def get_large_input_keys(self, name: str) -> set[str] | None:
        """Return keys whose values should be truncated in working messages, or None."""
        return self._metadata.get(name, {}).get("large_input_keys")

    def get_compaction_priority(self, name: str) -> str:
        """Return tool's compaction priority: 'ephemeral', 'standard', or 'persistent'."""
        return self._metadata.get(name, {}).get("compaction_priority", "standard")

    def is_dedup_exempt(self, name: str) -> bool:
        """Return True if the tool should be excluded from duplicate-call detection."""
        return self._metadata.get(name, {}).get("dedup_exempt", False)

    def get_function(self, name: str) -> Optional[Callable]:
        return self._functions.get(name)

    def get_schemas(self) -> list[dict]:
        return self._schemas.copy()

    def get_schema(self, name: str) -> Optional[dict]:
        return self._schema_map.get(name)

    def get_functions(self) -> dict[str, Callable]:
        return self._functions.copy()

    def list_tools(self) -> list[str]:
        return sorted(self._functions.keys())


_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _registry
