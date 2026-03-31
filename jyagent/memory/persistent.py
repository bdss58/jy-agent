# File-backed key-value store with atomic writes.

import os
from typing import Any, Optional

from ..config import MEMORY_DIR
from .utils import atomic_write, load_json


class PersistentMemory:
    """File-backed key-value store with atomic writes."""

    def __init__(self, store_dir: str = MEMORY_DIR):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def save(self, key: str, data: Any) -> None:
        filepath = os.path.join(self.store_dir, f"{key}.json")
        atomic_write(filepath, data)

    def load(self, key: str) -> Optional[Any]:
        filepath = os.path.join(self.store_dir, f"{key}.json")
        return load_json(filepath, default=None)

    def list_keys(self) -> list[str]:
        if not os.path.exists(self.store_dir):
            return []
        keys = []
        for filename in os.listdir(self.store_dir):
            if filename.endswith('.json') and not filename.startswith('_'):
                keys.append(filename[:-5])
        return sorted(keys)

    def delete(self, key: str) -> bool:
        filepath = os.path.join(self.store_dir, f"{key}.json")
        try:
            os.remove(filepath)
            return True
        except FileNotFoundError:
            return False
