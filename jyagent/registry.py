from typing import Any, Callable, Optional


class ToolRegistry:
    def __init__(self):
        self._functions: dict[str, Callable] = {}
        self._schemas: list[dict] = []
        self._schema_map: dict[str, dict] = {}

    def register(self, name: str, fn: Callable, schema: dict) -> None:
        self._functions[name] = fn
        self._schema_map[name] = schema
        self._schemas = list(self._schema_map.values())

    def unregister(self, name: str) -> bool:
        if name in self._functions:
            del self._functions[name]
            del self._schema_map[name]
            self._schemas = list(self._schema_map.values())
            return True
        return False

    def get_function(self, name: str) -> Optional[Callable]:
        return self._functions.get(name)

    def get_schemas(self) -> list[dict]:
        return self._schemas.copy()

    def get_functions(self) -> dict[str, Callable]:
        return self._functions.copy()

    def list_tools(self) -> list[str]:
        return sorted(self._functions.keys())


_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _registry
