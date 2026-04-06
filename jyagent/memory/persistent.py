# File-backed key-value store with JSON persistence helpers.

import json
import os
import tempfile
from typing import Any, Optional

from ..config import MEMORY_DIR


def atomic_write(filepath: str, data: Any) -> None:
    """Atomically write JSON data to disk."""
    dir_for_tmp = os.path.dirname(filepath) or MEMORY_DIR
    os.makedirs(dir_for_tmp, exist_ok=True)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_for_tmp, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
            os.replace(tmp_path, filepath)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(filepath: str, default=None):
    """Load JSON data from disk, returning default on missing/corrupt files."""
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


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
