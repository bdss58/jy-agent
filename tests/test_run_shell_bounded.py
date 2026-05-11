"""Memory-safety tests for ``run_shell``.

These exist because the previous implementation buffered the entire
child stdout+stderr in RAM before truncating to 50 000 chars. A runaway
command (recursive find, ``yes`` pipe, infinite log loop) could push the
parent agent into multi-GB territory and trip macOS jetsam — see
journal 2026-04-30 (~74 GiB resident → SIGKILL in the wild).

The contract these tests pin down:

1. Normal commands still work (exit code, stdout, stderr, timeout).
2. A child that emits hundreds of MB does NOT inflate the parent's RSS
   beyond a small bound.
3. A child that exceeds ``_RUN_SHELL_HARD_KILL_BYTES`` on a single
   stream gets SIGKILLed and the result is tagged with an overflow
   marker.
4. The existing 50 000-char user-visible cap is still applied.
"""

from __future__ import annotations

import os
import resource
import threading
import time

import pytest

from jyagent.tools import shell as tools_core
from jyagent.tools.shell import run_shell


def _rss_kb() -> int:
    """Self RSS in KB. macOS reports bytes, Linux KB — normalize to KB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return rss // 1024
    return rss


# ---------------------------------------------------------------------------
# Baseline: behaviour preserved
# ---------------------------------------------------------------------------


def test_simple_stdout():
    result = run_shell("echo hello")
    assert not result.is_error
    assert "hello" in result.content


def test_nonzero_exit():
    result = run_shell("false")
    assert result.is_error
    # Empty-output non-zero path emits a synthetic message.
    assert "exit" in result.content.lower() or result.content.strip() != ""


def test_stderr_captured():
    result = run_shell("echo oops 1>&2; exit 3")
    assert result.is_error
    assert "oops" in result.content
    assert "STDERR" in result.content


def test_timeout_kills_child():
    t0 = time.monotonic()
    result = run_shell("sleep 30", timeout=1)
    elapsed = time.monotonic() - t0
    assert result.is_error
    assert "timed out" in result.content.lower()
    assert elapsed < 5, f"timeout should fire promptly, took {elapsed:.1f}s"


def test_user_visible_char_cap():
    # 200 KB of 'x' — well under the hard kill but well over the 50k char cap.
    result = run_shell("python3 -c 'print(\"x\"*200000)'")
    assert not result.is_error
    assert len(result.content) <= tools_core._RUN_SHELL_OUTPUT_CHAR_CAP + 200
    assert "truncated at 50000 chars" in result.content


# ---------------------------------------------------------------------------
# Memory safety — the actual fix
# ---------------------------------------------------------------------------


def test_huge_output_does_not_inflate_rss():
    """500 MB of stdout must NOT cost the parent 500 MB of RSS.

    Pre-fix, ``capture_output=True`` would have grown RSS by ≥500 MB
    (often more, due to the ``str`` decode step). Post-fix the bounded
    drain caps RAM at head + tail (~136 KB) per stream.
    """
    rss_before = _rss_kb()

    # Stream 500 MB to stdout in 1 MB chunks. ``dd`` with /dev/zero is
    # the cleanest portable way to get high-throughput output.
    # The hard kill is 32 MB → child will be SIGKILLed long before
    # 500 MB lands. That is itself part of the contract: the parent
    # never has to absorb the full stream.
    result = run_shell(
        "dd if=/dev/zero bs=1048576 count=500 2>/dev/null", timeout=30,
    )

    rss_after = _rss_kb()
    growth_mb = (rss_after - rss_before) / 1024
    # Generous bound — Python interpreter wiggle, GC pressure, test
    # framework overhead. The pre-fix code would blow past 500 MB here;
    # 100 MB is comfortably below that and well above the bounded
    # buffer's true cost.
    assert growth_mb < 100, (
        f"RSS grew {growth_mb:.1f} MB during 500 MB stdout — "
        "bounded drain is leaking"
    )

    # Either: child got SIGKILLed via overflow path, OR finished
    # cleanly with a fully-bounded captured output. Both are acceptable.
    if "output exceeded" in result.content:
        assert result.is_error  # overflow tagged as error
    # Output, regardless of path, must be bounded by the user-visible cap.
    assert len(result.content) <= tools_core._RUN_SHELL_OUTPUT_CHAR_CAP + 500


def test_overflow_kills_child_and_tags_output():
    """A stream that crosses the 32 MB threshold must SIGKILL the child."""
    t0 = time.monotonic()
    # 64 MB > _RUN_SHELL_HARD_KILL_BYTES (32 MB).
    result = run_shell(
        "dd if=/dev/zero bs=1048576 count=64 2>/dev/null", timeout=30,
    )
    elapsed = time.monotonic() - t0

    assert result.is_error
    assert "output exceeded" in result.content
    # Should kill quickly, well before timeout.
    assert elapsed < 15, f"overflow kill was slow: {elapsed:.1f}s"


def test_overflow_on_stderr_also_triggers():
    """stderr is on its own bounded buffer; overflow there also kills."""
    result = run_shell(
        "dd if=/dev/zero bs=1048576 count=64 1>&2 2>/dev/null; "
        "true",  # ensure shell sees dd's exit code via stderr only
        timeout=30,
    )
    # dd was killed → parent shell may or may not exit clean; what we
    # assert is that the marker appeared.
    assert "output exceeded" in result.content


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cooperative_cancel():
    cancel = threading.Event()

    def fire_cancel_soon():
        time.sleep(0.5)
        cancel.set()

    threading.Thread(target=fire_cancel_soon, daemon=True).start()

    t0 = time.monotonic()
    result = run_shell("sleep 30", timeout=30, _cancel_event=cancel)
    elapsed = time.monotonic() - t0

    assert result.is_error
    assert "cancel" in result.content.lower()
    assert elapsed < 5, f"cancel was slow: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Drain helper unit tests
# ---------------------------------------------------------------------------


def test_bounded_reader_keeps_head_and_tail():
    """Direct test of the buffer logic with a fake stream."""
    import io

    payload = b"H" * 100 + b"M" * 100_000 + b"T" * 100
    stream = io.BytesIO(payload)

    fired = []
    reader = tools_core._BoundedStreamReader(
        stream,
        head_bytes=50,
        tail_bytes=50,
        hard_kill_bytes=10**9,
        on_overflow=lambda: fired.append(True),
    )
    reader.start()
    reader.join(timeout=5)

    text, overflowed = reader.collect()
    assert not overflowed
    # First 50 bytes are 'H', last 50 bytes are 'T'.
    assert text.startswith("H" * 50)
    assert text.endswith("T" * 50)
    assert "elided" in text


def test_bounded_reader_fires_overflow():
    import io

    payload = b"x" * 5000
    stream = io.BytesIO(payload)

    fired = []
    reader = tools_core._BoundedStreamReader(
        stream,
        head_bytes=100,
        tail_bytes=100,
        hard_kill_bytes=1000,  # well below total → must fire
        on_overflow=lambda: fired.append(True),
    )
    reader.start()
    reader.join(timeout=5)

    _, overflowed = reader.collect()
    assert overflowed
    assert fired == [True]  # fired exactly once



# ---------------------------------------------------------------------------
# Three-regime collect() correctness — regression test for the boundary bug
# where ``tail_max < total <= head_max + tail_max`` was silently returning
# only the tail (dropping head bytes that tail had overflowed past).
# ---------------------------------------------------------------------------


def _run_reader(payload: bytes, head_bytes: int, tail_bytes: int):
    import io
    reader = tools_core._BoundedStreamReader(
        io.BytesIO(payload),
        head_bytes=head_bytes,
        tail_bytes=tail_bytes,
        hard_kill_bytes=10**9,
        on_overflow=lambda: None,
    )
    reader.start()
    reader.join(timeout=5)
    return reader.collect()


def test_collect_regime1_tail_holds_everything():
    """total <= tail_max: full output, no elision marker, no data loss."""
    payload = b"A" * 30 + b"B" * 40  # total=70, tail_max=100 → tail has all
    text, overflowed = _run_reader(payload, head_bytes=20, tail_bytes=100)
    assert not overflowed
    assert text == payload.decode()
    assert "elided" not in text


def test_collect_regime2_overlap_no_gap_no_loss():
    """tail_max < total <= head_max + tail_max: dedupe, no marker, no loss.

    Pre-fix this regime silently returned tail-only, dropping the early
    bytes that head had but tail had overflowed past.
    """
    # head_max=20, tail_max=50, total=60 → in (50, 70] → regime 2.
    # Bytes 0..19 are 'H', bytes 20..59 are 'T'. tail will hold the
    # last 50 bytes = bytes 10..59 → tail has 'H'*10 + 'T'*40.
    # head has 'H'*20.  Dedupe: head[:60-50] + tail = 'H'*10 + tail.
    # Final reconstruction must equal the full 60-byte payload.
    payload = b"H" * 20 + b"T" * 40
    text, overflowed = _run_reader(payload, head_bytes=20, tail_bytes=50)
    assert not overflowed
    assert text == payload.decode(), (
        f"regime 2 lost data: expected {len(payload)} bytes, got {len(text)}"
    )
    assert "elided" not in text  # no gap, no marker


def test_collect_regime2_exact_boundary():
    """total == head_max + tail_max: still no gap, still no marker."""
    # head_max=20, tail_max=50, total=70 → tail holds last 50 (bytes 20..69),
    # head holds first 20 (bytes 0..19). They are adjacent, no overlap, no
    # gap. head[:total-tail_max] = head[:20] = full head.
    payload = b"H" * 20 + b"T" * 50
    text, overflowed = _run_reader(payload, head_bytes=20, tail_bytes=50)
    assert not overflowed
    assert text == payload.decode()
    assert "elided" not in text


def test_collect_regime3_real_gap_marker_present():
    """total > head_max + tail_max: marker present, head + tail visible."""
    # head_max=20, tail_max=50, total=200 → 130 bytes elided.
    payload = b"H" * 100 + b"M" * 50 + b"T" * 50
    text, overflowed = _run_reader(payload, head_bytes=20, tail_bytes=50)
    assert not overflowed
    assert text.startswith("H" * 20)
    assert text.endswith("T" * 50)
    assert "[... 130 bytes elided ...]" in text


def test_collect_regime_default_config_boundary_band():
    """In default config (head=8 KB, tail=128 KB), the buggy band was
    (128 KB, 136 KB]. Smoke-test a payload there."""
    H = tools_core._RUN_SHELL_HEAD_BYTES   # 8 KB
    T = tools_core._RUN_SHELL_TAIL_BYTES   # 128 KB
    total = T + (H // 2)                    # 132 KB → middle of the band
    payload = b"S" * (H // 2) + b"x" * (total - (H // 2))  # distinct head prefix
    assert len(payload) == total            # arithmetic sanity
    text, overflowed = _run_reader(payload, head_bytes=H, tail_bytes=T)
    assert not overflowed
    # The original head bytes ('S'*4096) MUST appear at the start —
    # pre-fix they were silently dropped.
    assert text.startswith("S" * (H // 2))
    assert len(text) == total
    assert "elided" not in text



# ---------------------------------------------------------------------------
# Spill-to-disk: large outputs land in /tmp and are recoverable by the agent
# ---------------------------------------------------------------------------


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_no_spill_for_small_output(tmp_path):
    """Output that fits in the tail buffer creates no spill file."""
    # Snapshot files in spill dir before/after.
    before = set(os.listdir("/tmp"))
    result = run_shell("echo hello")
    after = set(os.listdir("/tmp"))
    new_files = after - before
    spill_files = {f for f in new_files if f.startswith("jyagent_runshell_")}
    assert not spill_files, (
        f"small output should not spill, got: {spill_files}"
    )
    assert "spilled" not in result.content


def test_spill_activates_when_tail_overflows():
    """Output > tail_max bytes triggers spill, file contains full output."""
    # Default tail_max = 128 KB. Emit 256 KB → must spill.
    payload_size = 256 * 1024
    result = run_shell(
        f"python3 -c 'import sys; sys.stdout.write(\"x\"*{payload_size})'",
        timeout=15,
    )
    assert not result.is_error
    assert "spilled to /tmp/jyagent_runshell_out_" in result.content

    # Extract the spill path from the marker.
    import re
    m = re.search(r"spilled to (/tmp/jyagent_runshell_out_\S+\.out)", result.content)
    assert m, f"no spill path in output: {result.content[-500:]}"
    spill_path = m.group(1)
    try:
        data = _read_bytes(spill_path)
        # Spill file should contain the FULL payload (recoverable).
        assert len(data) == payload_size, (
            f"spill incomplete: {len(data)} bytes, expected {payload_size}"
        )
        assert data == b"x" * payload_size
    finally:
        os.unlink(spill_path)


def test_spill_captures_data_before_hard_kill():
    """When child is SIGKILLed for overflow, spill still has up to that point."""
    # Default hard kill = 32 MB. Emit 64 MB → child killed, spill has ~32 MB.
    result = run_shell(
        "dd if=/dev/zero bs=1048576 count=64 2>/dev/null", timeout=30,
    )
    assert result.is_error
    assert "output exceeded" in result.content
    assert "spilled to /tmp/jyagent_runshell_out_" in result.content

    import re
    m = re.search(r"spilled to (/tmp/jyagent_runshell_out_\S+\.out)", result.content)
    assert m, f"no spill path in output: {result.content[-500:]}"
    spill_path = m.group(1)
    try:
        size = os.path.getsize(spill_path)
        # We started spilling at tail_max=128 KB and kept teeing until kill
        # at 32 MB. So spill file is at least ~30 MB.  Allow slack for
        # buffering / chunk timing.
        assert size > 20 * 1024 * 1024, (
            f"spill file should have most of pre-kill bytes, got {size}"
        )
        assert size < 64 * 1024 * 1024, (
            f"spill file should be bounded by kill threshold, got {size}"
        )
    finally:
        os.unlink(spill_path)


def test_spill_disabled_via_env(monkeypatch):
    """JYAGENT_RUN_SHELL_SPILL=0 disables spill entirely."""
    monkeypatch.setattr(tools_core, "_RUN_SHELL_SPILL_ENABLED", False)
    before = set(os.listdir("/tmp"))
    result = run_shell(
        "python3 -c 'import sys; sys.stdout.write(\"x\"*200000)'", timeout=10,
    )
    after = set(os.listdir("/tmp"))
    new_files = after - before
    spill_files = {f for f in new_files if f.startswith("jyagent_runshell_")}
    assert not spill_files
    assert "spilled to" not in result.content
    # Inline output should still be capped (50 K chars).
    assert "truncated at 50000 chars" in result.content


def test_spill_path_factory_failure_is_silent_fallback():
    """If the spill factory raises (e.g. /tmp is RO), we fall back to in-memory only."""
    import io
    payload = b"x" * 200_000

    def boom():
        raise OSError("read-only filesystem (simulated)")

    reader = tools_core._BoundedStreamReader(
        io.BytesIO(payload),
        head_bytes=8 * 1024,
        tail_bytes=128 * 1024,
        hard_kill_bytes=10**9,
        on_overflow=lambda: None,
        spill_path_factory=boom,
    )
    reader.start()
    reader.join(timeout=5)
    text, overflowed = reader.collect()
    assert reader.spill_path is None       # spill never created
    assert not overflowed                   # below hard kill
    assert text.startswith("x" * 100)       # inline buffer still works


def test_spill_round_trip_via_unit():
    """Direct unit test of the reader: spill activates, file has full payload."""
    import io, tempfile
    payload = b"H" * 100 + b"M" * 200_000 + b"T" * 100
    spill_holder = {}

    def factory():
        fd, path = tempfile.mkstemp(prefix="jyagent_test_", suffix=".out")
        spill_holder["path"] = path
        return path, os.fdopen(fd, "ab")

    reader = tools_core._BoundedStreamReader(
        io.BytesIO(payload),
        head_bytes=8 * 1024,
        tail_bytes=128 * 1024,
        hard_kill_bytes=10**9,
        on_overflow=lambda: None,
        spill_path_factory=factory,
    )
    reader.start()
    reader.join(timeout=5)

    assert reader.spill_path == spill_holder["path"]
    try:
        data = _read_bytes(spill_holder["path"])
        assert data == payload, (
            f"spill file content mismatch: {len(data)} vs {len(payload)}"
        )
    finally:
        os.unlink(spill_holder["path"])


# ---------------------------------------------------------------------------
# Env-var configuration: malformed values must warn loudly, not silently fall
# back (the regression Claude Code shipped in v2.1.2 with BASH_MAX_OUTPUT_LENGTH).
# ---------------------------------------------------------------------------


def test_env_int_valid(monkeypatch):
    monkeypatch.setenv("FOO_BAR", "12345")
    assert tools_core._env_int("FOO_BAR", 99) == 12345


def test_env_int_unset_uses_default(monkeypatch):
    monkeypatch.delenv("FOO_BAR", raising=False)
    assert tools_core._env_int("FOO_BAR", 99) == 99


def test_env_int_malformed_warns_and_uses_default(monkeypatch, capsys):
    monkeypatch.setenv("FOO_BAR", "not-an-int")
    val = tools_core._env_int("FOO_BAR", 99)
    assert val == 99
    captured = capsys.readouterr()
    # Must NOT be silent — claude-code's #17944 was a silent ignore.
    assert "FOO_BAR" in captured.err
    assert "not an int" in captured.err


def test_env_int_below_minimum_warns(monkeypatch, capsys):
    monkeypatch.setenv("FOO_BAR", "5")
    val = tools_core._env_int("FOO_BAR", 99, minimum=10)
    assert val == 99
    assert "below minimum" in capsys.readouterr().err


def test_env_bool_variants(monkeypatch):
    for raw, expected in [
        ("1", True), ("true", True), ("yes", True), ("on", True), ("TRUE", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
    ]:
        monkeypatch.setenv("FOO_BOOL", raw)
        assert tools_core._env_bool("FOO_BOOL", default=not expected) == expected


def test_env_bool_malformed_warns(monkeypatch, capsys):
    monkeypatch.setenv("FOO_BOOL", "maybe")
    val = tools_core._env_bool("FOO_BOOL", default=True)
    assert val is True  # default
    assert "FOO_BOOL" in capsys.readouterr().err
