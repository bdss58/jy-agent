# Tests for memory subsystem

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.memory.utils import (
    estimate_tokens, estimate_message_tokens, estimate_conversation_tokens,
    atomic_write, load_json,
)
from jyagent.memory.conversation import ConversationMemory
from jyagent.memory.persistent import PersistentMemory


class TestTokenEstimation:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_basic(self):
        # "hello" = 5 chars → 5/4 = 1 token
        assert estimate_tokens("hello") == 1

    def test_longer(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_message_string(self):
        msg = {"role": "user", "content": "hello world"}
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_message_list_content(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "name": "run_shell", "input": {"command": "ls"}},
            ]
        }
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_conversation_tokens(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        tokens = estimate_conversation_tokens(messages)
        assert tokens > 0


class TestConversationMemory:
    def test_add_and_get(self):
        conv = ConversationMemory()
        conv.add_message("user", "hello")
        conv.add_message("assistant", "hi")
        assert len(conv) == 2
        history = conv.get_history()
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "hi"

    def test_get_recent(self):
        conv = ConversationMemory()
        for i in range(20):
            conv.add_message("user", f"msg {i}")
        recent = conv.get_recent(5)
        assert len(recent) == 5
        assert recent[0]["content"] == "msg 15"

    def test_clear(self):
        conv = ConversationMemory()
        conv.add_message("user", "hello")
        conv.clear()
        assert len(conv) == 0

    def test_estimated_tokens(self):
        conv = ConversationMemory()
        conv.add_message("user", "a" * 1000)
        tokens = conv.estimated_tokens()
        assert tokens > 200  # ~1000/4 = 250


class TestAtomicWrite:
    def test_write_and_read(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            data = {"key": "value", "list": [1, 2, 3]}
            atomic_write(path, data)
            loaded = load_json(path)
            assert loaded == data
        finally:
            os.unlink(path)

    def test_load_missing(self):
        result = load_json("/nonexistent/file.json", default=[])
        assert result == []

    def test_load_corrupt(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not json{{{")
            path = f.name
        try:
            result = load_json(path, default={"fallback": True})
            assert result == {"fallback": True}
        finally:
            os.unlink(path)


class TestPersistentMemory:
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistentMemory(store_dir=tmpdir)
            pm.save("test_key", {"data": 42})
            loaded = pm.load("test_key")
            assert loaded == {"data": 42}

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistentMemory(store_dir=tmpdir)
            # load_json returns {} by default when file not found
            assert pm.load("nonexistent") is None or pm.load("nonexistent") == {}

    def test_list_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistentMemory(store_dir=tmpdir)
            pm.save("alpha", {})
            pm.save("beta", {})
            keys = pm.list_keys()
            assert "alpha" in keys
            assert "beta" in keys

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistentMemory(store_dir=tmpdir)
            pm.save("to_delete", {"data": 1})
            assert pm.delete("to_delete") is True
            # After delete, load returns default (None or {})
            result = pm.load("to_delete")
            assert result is None or result == {}
            assert pm.delete("to_delete") is False
