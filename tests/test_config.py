# Tests for central configuration

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.config import (
    SKIP_DIRS, BINARY_EXTS,
    DEFAULT_MAX_TOKENS, MAX_TOKENS_CAP, DEFAULT_MAX_STEPS,
    MAX_TOOL_RESULT_CHARS, DEFAULT_TOOL_TIMEOUT,
    MEMORY_DIR, TOPICS_DIR, MEMORY_MD_FILE, SESSIONS_FILE,
    CHARS_PER_TOKEN,
)


class TestConfig:
    def test_skip_dirs_is_set(self):
        assert isinstance(SKIP_DIRS, set)
        assert '.git' in SKIP_DIRS
        assert 'node_modules' in SKIP_DIRS

    def test_binary_exts_is_set(self):
        assert isinstance(BINARY_EXTS, set)
        assert '.png' in BINARY_EXTS
        assert '.exe' in BINARY_EXTS

    def test_numeric_constants(self):
        assert isinstance(DEFAULT_MAX_TOKENS, int)
        assert DEFAULT_MAX_TOKENS > 0
        assert MAX_TOKENS_CAP > DEFAULT_MAX_TOKENS
        assert DEFAULT_MAX_STEPS > 0
        assert MAX_TOOL_RESULT_CHARS > 0
        assert DEFAULT_TOOL_TIMEOUT > 0

    def test_paths(self):
        assert "memory" in MEMORY_DIR
        assert "topics" in TOPICS_DIR
        assert MEMORY_MD_FILE.endswith("MEMORY.md")
        assert SESSIONS_FILE.endswith(".json")

    def test_chars_per_token(self):
        assert CHARS_PER_TOKEN == 4
