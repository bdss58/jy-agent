# Tests for glob_files and grep_files

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.tools.search import glob_files, grep_files


class TestGlobFiles:
    def test_star_py_in_root(self):
        """*.py matches Python files in the root directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "main.py"), 'w').close()
            open(os.path.join(tmpdir, "readme.md"), 'w').close()
            result = glob_files("*.py", tmpdir)
            assert "main.py" in result
            assert "readme.md" not in result

    def test_path_qualified_pattern(self):
        """src/*.py matches files inside src/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "src"))
            open(os.path.join(tmpdir, "src", "app.py"), 'w').close()
            open(os.path.join(tmpdir, "root.py"), 'w').close()
            result = glob_files("src/*.py", tmpdir)
            assert "app.py" in result
            assert "root.py" not in result

    def test_recursive_double_star(self):
        """**/*.py matches Python files in any subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "a", "b"))
            open(os.path.join(tmpdir, "top.py"), 'w').close()
            open(os.path.join(tmpdir, "a", "mid.py"), 'w').close()
            open(os.path.join(tmpdir, "a", "b", "deep.py"), 'w').close()
            result = glob_files("**/*.py", tmpdir)
            assert "mid.py" in result
            assert "deep.py" in result

    def test_skips_egg_info(self):
        """Directories matching *.egg-info are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            egg_dir = os.path.join(tmpdir, "foo.egg-info")
            os.makedirs(egg_dir)
            open(os.path.join(egg_dir, "PKG-INFO"), 'w').close()
            open(os.path.join(tmpdir, "setup.py"), 'w').close()
            result = glob_files("**/*", tmpdir)
            assert "PKG-INFO" not in result
            assert "setup.py" in result

    def test_skips_hidden_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, ".hidden"))
            open(os.path.join(tmpdir, ".hidden", "secret.py"), 'w').close()
            open(os.path.join(tmpdir, "visible.py"), 'w').close()
            result = glob_files("**/*.py", tmpdir)
            assert "secret.py" not in result
            assert "visible.py" in result

    def test_skips_binary_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "image.png"), 'w').close()
            open(os.path.join(tmpdir, "code.py"), 'w').close()
            result = glob_files("*", tmpdir)
            assert "image.png" not in result
            assert "code.py" in result

    def test_no_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = glob_files("*.xyz", tmpdir)
            assert "No files matching" in result

    def test_max_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                open(os.path.join(tmpdir, f"file{i}.txt"), 'w').close()
            result = glob_files("*.txt", tmpdir, max_results=3)
            assert "truncated" in result


class TestGrepFiles:
    def test_basic_content_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.txt"), 'w') as f:
                f.write("hello world\nfoo bar\nhello again\n")
            result = grep_files("hello", tmpdir)
            assert "2 matches" in result
            assert "hello world" in result
            assert "hello again" in result

    def test_files_only_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.txt"), 'w') as f:
                f.write("match here\n")
            with open(os.path.join(tmpdir, "b.txt"), 'w') as f:
                f.write("no match\n")
            result = grep_files("match here", tmpdir, output_mode="files_only")
            assert "a.txt" in result
            assert "b.txt" not in result

    def test_count_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.txt"), 'w') as f:
                f.write("x\nx\nx\n")
            result = grep_files("x", tmpdir, output_mode="count")
            assert "3 matches" in result

    def test_context_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.txt"), 'w') as f:
                f.write("before\ntarget\nafter\n")
            result = grep_files("target", tmpdir, context_lines=1)
            assert "before" in result
            assert "after" in result

    def test_ignore_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.txt"), 'w') as f:
                f.write("Hello World\n")
            result = grep_files("hello", tmpdir, ignore_case=True)
            assert "Hello World" in result

    def test_file_pattern_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.py"), 'w') as f:
                f.write("match\n")
            with open(os.path.join(tmpdir, "b.txt"), 'w') as f:
                f.write("match\n")
            result = grep_files("match", tmpdir, file_pattern="*.py")
            assert "a.py" in result
            assert "b.txt" not in result

    def test_invalid_regex_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.txt"), 'w') as f:
                f.write("hello (world\n")
            # Invalid regex should fall back to literal search
            result = grep_files("(world", tmpdir)
            assert "hello (world" in result

    def test_single_file_search(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line one\nline two\nline three\n")
            path = f.name
        try:
            result = grep_files("two", path)
            assert "line two" in result
        finally:
            os.unlink(path)

    def test_no_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.txt"), 'w') as f:
                f.write("nothing here\n")
            result = grep_files("xyznonexistent", tmpdir)
            assert "No matches" in result

    def test_max_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.txt"), 'w') as f:
                f.write("match\n" * 100)
            result = grep_files("match", tmpdir, max_results=5)
            assert "truncated" in result

    def test_skips_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "data.png"), 'w') as f:
                f.write("match\n")
            result = grep_files("match", tmpdir)
            assert "No matches" in result
