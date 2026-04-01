# Tests for jyagent/tools/_fileutils.py

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.tools._fileutils import resolve_path, atomic_write, should_skip_dir, is_binary_ext


class TestResolvePath:
    def test_absolute_unchanged(self):
        assert resolve_path("/tmp/foo.txt") == "/tmp/foo.txt"

    def test_relative_uses_cwd(self):
        result = resolve_path("foo.txt")
        assert os.path.isabs(result)
        assert result == os.path.join(os.getcwd(), "foo.txt")

    def test_relative_uses_root(self):
        result = resolve_path("bar.py", root="/some/project")
        assert result == "/some/project/bar.py"

    def test_tilde_expanded(self):
        result = resolve_path("~/notes.txt")
        assert not result.startswith("~")
        assert os.path.isabs(result)

    def test_dot_segments_resolved(self):
        result = resolve_path("/a/b/../c")
        assert result == "/a/c"


class TestAtomicWrite:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "new.txt")
            atomic_write(path, "hello")
            with open(path) as f:
                assert f.read() == "hello"

    def test_overwrites_existing(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("old content")
            path = f.name
        try:
            atomic_write(path, "new content")
            with open(path) as f:
                assert f.read() == "new content"
        finally:
            os.unlink(path)

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deep", "sub", "file.txt")
            atomic_write(path, "nested")
            with open(path) as f:
                assert f.read() == "nested"

    def test_failure_preserves_original(self, monkeypatch):
        """If os.replace fails, the original file should be untouched."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("original")
            path = f.name
        try:
            real_replace = os.replace
            def failing_replace(src, dst):
                raise OSError("simulated failure")

            monkeypatch.setattr(os, "replace", failing_replace)
            with pytest.raises(OSError, match="simulated failure"):
                atomic_write(path, "should not appear")

            # Original content preserved
            with open(path) as f:
                assert f.read() == "original"
        finally:
            os.unlink(path)

    def test_no_temp_file_left_on_failure(self, monkeypatch):
        """Temp file should be cleaned up on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "target.txt")
            def failing_replace(src, dst):
                raise OSError("boom")
            monkeypatch.setattr(os, "replace", failing_replace)

            with pytest.raises(OSError):
                atomic_write(path, "content")

            # No temp files left behind
            remaining = os.listdir(tmpdir)
            assert not any(f.startswith(".tmp_") for f in remaining)


class TestShouldSkipDir:
    def test_hidden_dir(self):
        assert should_skip_dir(".hidden") is True

    def test_exact_match(self):
        assert should_skip_dir("node_modules") is True
        assert should_skip_dir("__pycache__") is True

    def test_egg_info_pattern(self):
        assert should_skip_dir("mypackage.egg-info") is True
        assert should_skip_dir("foo.egg-info") is True

    def test_normal_dir_allowed(self):
        assert should_skip_dir("src") is False
        assert should_skip_dir("tests") is False

    def test_git_skipped(self):
        # .git starts with dot, so it's skipped
        assert should_skip_dir(".git") is True


class TestIsBinaryExt:
    def test_binary_extensions(self):
        assert is_binary_ext("image.png") is True
        assert is_binary_ext("archive.zip") is True
        assert is_binary_ext("lib.so") is True

    def test_text_extensions(self):
        assert is_binary_ext("code.py") is False
        assert is_binary_ext("readme.md") is False
        assert is_binary_ext("config.json") is False

    def test_case_insensitive(self):
        assert is_binary_ext("IMAGE.PNG") is True
        assert is_binary_ext("file.JPEG") is True
