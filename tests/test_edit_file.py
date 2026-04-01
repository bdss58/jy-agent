# Tests for edit_file

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.tools.core import edit_file


class TestEditFileReplace:
    def test_exact_replace(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello world\nfoo bar\n")
            path = f.name
        try:
            result = edit_file(path, new_text="hello universe", old_text="hello world")
            assert "Edited" in result
            with open(path) as f:
                assert "hello universe" in f.read()
        finally:
            os.unlink(path)

    def test_no_match_diagnostic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello world\n")
            path = f.name
        try:
            result = edit_file(path, new_text="x", old_text="does not exist")
            assert "Error" in result
            assert "not found" in result.lower() or "old_text" in result
        finally:
            os.unlink(path)

    def test_whitespace_mismatch_diagnostic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("    indented line\n    more indented\n")
            path = f.name
        try:
            # Full-line old_text without indentation triggers a diagnostic
            result = edit_file(path, new_text="x", old_text="indented line\nmore indented")
            assert "Error" in result
            # Should show a helpful diagnostic (fuzzy match or whitespace hint)
            assert "match" in result.lower() or "hint" in result.lower()
        finally:
            os.unlink(path)

    def test_multi_match_diagnostic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("dup\nother\ndup\n")
            path = f.name
        try:
            result = edit_file(path, new_text="x", old_text="dup")
            assert "Error" in result
            assert "2 times" in result
        finally:
            os.unlink(path)

    def test_dry_run_replace(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("alpha\nbeta\n")
            path = f.name
        try:
            result = edit_file(path, new_text="gamma", old_text="alpha", dry_run=True)
            assert "DRY RUN" in result
            with open(path) as f:
                assert f.read() == "alpha\nbeta\n"  # unchanged
        finally:
            os.unlink(path)


class TestEditFileInsert:
    def test_insert_at_beginning(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\nline2\n")
            path = f.name
        try:
            result = edit_file(path, new_text="header", insert_at_line=1)
            assert "Inserted" in result
            with open(path) as f:
                content = f.read()
            assert content.startswith("header\n")
        finally:
            os.unlink(path)

    def test_insert_at_middle(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("a\nb\nc\n")
            path = f.name
        try:
            edit_file(path, new_text="X", insert_at_line=2)
            with open(path) as f:
                lines = f.read().split('\n')
            assert lines[0] == "a"
            assert lines[1] == "X"
            assert lines[2] == "b"
        finally:
            os.unlink(path)

    def test_insert_beyond_end(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("only\n")
            path = f.name
        try:
            result = edit_file(path, new_text="extra", insert_at_line=999)
            assert "Inserted" in result
        finally:
            os.unlink(path)


class TestEditFileAppend:
    def test_append(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("start")
            path = f.name
        try:
            result = edit_file(path, new_text="\nend")
            assert "Appended" in result
            with open(path) as f:
                assert f.read() == "start\nend"
        finally:
            os.unlink(path)


class TestEditFileCreate:
    def test_create_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "new_file.txt")
            result = edit_file(path, new_text="content", create_if_missing=True)
            assert "Created" in result
            with open(path) as f:
                assert f.read() == "content"

    def test_create_missing_without_flag(self):
        result = edit_file("/tmp/nonexistent_xyz.txt", new_text="x")
        assert "Error" in result
        assert "create_if_missing" in result


class TestEditFileBinaryGuard:
    def test_rejects_binary(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(b'\x89PNG\r\n')
            path = f.name
        try:
            result = edit_file(path, new_text="x", old_text="y")
            assert "Error" in result
            assert "binary" in result.lower()
        finally:
            os.unlink(path)


class TestEditFileOperation:
    def test_operation_replace_validates(self):
        result = edit_file("/tmp/x.txt", new_text="y", operation="replace")
        assert "Error" in result
        assert "old_text" in result

    def test_operation_insert_validates(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test\n")
            path = f.name
        try:
            result = edit_file(path, new_text="y", operation="insert")
            assert "Error" in result
            assert "insert_at_line" in result
        finally:
            os.unlink(path)

    def test_operation_create(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "op_create.txt")
            result = edit_file(path, new_text="hello", operation="create")
            assert "Created" in result
            with open(path) as f:
                assert f.read() == "hello"

    def test_operation_append(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("base")
            path = f.name
        try:
            result = edit_file(path, new_text="+more", operation="append")
            assert "Appended" in result
            with open(path) as f:
                assert f.read() == "base+more"
        finally:
            os.unlink(path)

    def test_operation_invalid(self):
        result = edit_file("/tmp/x.txt", new_text="y", operation="bogus")
        assert "Error" in result
        assert "bogus" in result


class TestEditFileEncoding:
    def test_non_utf8_error(self):
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'\xff\xfe invalid utf-8 \x80\x81')
            path = f.name
        try:
            result = edit_file(path, new_text="x", old_text="y")
            assert "Error" in result
            assert "UTF-8" in result or "non-UTF-8" in result
        finally:
            os.unlink(path)
