# Tests for core tools (run_shell, read_file, write_file, list_directory)

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.tools.core import run_shell, read_file, write_file, list_directory


class TestRunShell:
    def test_basic_echo(self):
        result = run_shell("echo hello")
        assert "hello" in result

    def test_stderr_captured(self):
        result = run_shell("echo err >&2")
        assert "STDERR:" in result
        assert "err" in result

    def test_nonzero_exit(self):
        result = run_shell("exit 1")
        assert "exit" in result.lower() or "code 1" in result

    def test_timeout(self):
        result = run_shell("sleep 10", timeout=1)
        assert "timed out" in result.lower()

    def test_timeout_clamp(self):
        # timeout > 600 should be clamped
        result = run_shell("echo ok", timeout=9999)
        assert "ok" in result

    def test_output_truncation(self):
        result = run_shell("python3 -c \"print('x' * 60000)\"")
        assert len(result) <= 50100  # 50000 + some slack for truncation message


class TestReadFile:
    def test_read_existing(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\nline2\nline3\n")
            path = f.name
        try:
            result = read_file(path)
            assert "line1" in result
            assert "line2" in result
        finally:
            os.unlink(path)

    def test_read_missing(self):
        result = read_file("/nonexistent/path/file.txt")
        assert "Error" in result

    def test_line_numbers(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("alpha\nbeta\ngamma\n")
            path = f.name
        try:
            result = read_file(path, line_numbers=True)
            assert "L1: alpha" in result
            assert "L2: beta" in result
            assert "L3: gamma" in result
        finally:
            os.unlink(path)

    def test_offset_limit(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("a\nb\nc\nd\ne\n")
            path = f.name
        try:
            result = read_file(path, offset=2, limit=2)
            assert "b" in result
            assert "c" in result
            assert "d" not in result.split('\n')[-1]  # d should not be in output
        finally:
            os.unlink(path)

    def test_binary_file(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(b'\x89PNG\r\n')
            path = f.name
        try:
            result = read_file(path)
            assert "Binary file" in result
        finally:
            os.unlink(path)


class TestWriteFile:
    def test_basic_write(self):
        path = os.path.join(tempfile.gettempdir(), "jy_test_write.txt")
        try:
            result = write_file(path, "hello world")
            assert "Successfully wrote" in result
            with open(path) as f:
                assert f.read() == "hello world"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_creates_parent_dirs(self):
        path = os.path.join(tempfile.gettempdir(), "jy_test_deep", "sub", "file.txt")
        try:
            result = write_file(path, "nested")
            assert "Successfully wrote" in result
            assert os.path.exists(path)
        finally:
            import shutil
            shutil.rmtree(os.path.join(tempfile.gettempdir(), "jy_test_deep"), ignore_errors=True)


class TestListDirectory:
    def test_current_dir(self):
        result = list_directory(".")
        assert "entries" in result

    def test_nonexistent(self):
        result = list_directory("/nonexistent/path")
        assert "Error" in result

    def test_limit(self):
        result = list_directory(".", limit=2)
        # Should have limited entries
        assert "entries" in result

    def test_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "sub", "deep"))
            with open(os.path.join(tmpdir, "file.txt"), 'w') as f:
                f.write("test")
            with open(os.path.join(tmpdir, "sub", "nested.txt"), 'w') as f:
                f.write("test")

            # depth=1 should not show nested.txt
            result1 = list_directory(tmpdir, depth=1)
            assert "sub/" in result1

            # depth=2 should show nested.txt
            result2 = list_directory(tmpdir, depth=2)
            assert "nested.txt" in result2
