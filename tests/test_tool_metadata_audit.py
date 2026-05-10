"""Regression tests for tool metadata drift.

These tests pin down each registered tool's side-effect profile so that
adding a new tool — or changing an existing tool's behavior — without
updating ``_TOOL_METADATA`` fails CI loudly instead of silently breaking
the loop engine's timeout/cancel/partial-side-effect handling.

Background: Codex review flagged that ``manage_memory`` and ``manage_skills``
were marked ``mutating=False`` even though both write to disk. The bug had
been latent for ~weeks because no test asserted the contract between
"writes the filesystem" and "the metadata says so". This file is that test.
"""
from __future__ import annotations
import inspect

import pytest

import jyagent.tools  # registers everything as a side effect
from jyagent.tools import _TOOL_FN_MAP, _TOOL_METADATA


# ─── Source-of-truth: every tool's expected side-effect profile ──────────────
#
# Adding a tool? Add an entry here. The audit will fail until you do.
# Changing an existing tool's behavior (e.g., it now spawns a subprocess)?
# Update the entry here AND _TOOL_METADATA in tools/__init__.py — the audit
# will keep them in sync.

# Each entry: tool_name -> (mutating, parallel_safe, rationale)
EXPECTED_PROFILES: dict[str, tuple[bool, bool, str]] = {
    # Read-only file tools — pure queries, safe to retry, safe in parallel.
    "read_file":         (False, True,  "pure read"),
    "list_directory":    (False, True,  "pure read"),
    "glob_files":        (False, True,  "pure read"),
    "grep_files":        (False, True,  "pure read"),

    # Filesystem mutators — atomic_write or os.* mutation.
    "write_file":        (True,  False, "atomic_write to disk"),
    "edit_file":         (True,  False, "atomic_write to disk"),

    # Shell / subprocess — arbitrary side effects, never parallel-safe.
    "run_shell":         (True,  False, "spawns subprocess"),
    "run_background":    (True,  False, "spawns long-lived subprocess"),
    # check_background is mutating ONLY because action='kill' SIGTERMs the pgroup.
    # Reading status is read-only but the kill branch makes the whole tool unsafe
    # to silently retry on timeout.
    "check_background":  (True,  True,  "action=kill SIGTERMs pgroup"),

    # Memory / skills — both write to data/memory/ or skills dir on most actions.
    "manage_memory":     (True,  False, "writes MEMORY.md / topics / journal"),
    "manage_skills":     (True,  False, "writes/deletes skill files; mutates pin set"),

    # Sub-agent dispatch — spawns a whole AgentLoop with its own tool calls.
    "dispatch_agent":    (True,  False, "spawns sub-agent (transitive side effects)"),
    "check_agent":       (False, True,  "pure status read; cancel is cooperative"),

    # Network — currently treated as read-only (no caches/cookies persisted).
    # parallel_safe=True for web_search (engine cascade is reentrant);
    # web_fetch=False because Chrome MCP path mutates browser state.
    "web_search":        (False, True,  "stateless HTTP queries"),
    "web_fetch":         (False, False, "Chrome path mutates browser tab state"),

    # MCP control plane — connect/disconnect mutates the tool registry.
    "mcp":               (True,  False, "mutates MCP server connections + registry"),
}


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_every_registered_tool_has_an_expected_profile():
    """A new tool added to _TOOL_FN_MAP must have an EXPECTED_PROFILES entry.

    This catches the "I added a tool but forgot the metadata" case at test
    time instead of in production. Update both _TOOL_METADATA and the
    EXPECTED_PROFILES dict above when adding a tool.
    """
    registered = set(_TOOL_FN_MAP.keys())
    expected = set(EXPECTED_PROFILES.keys())
    missing = registered - expected
    extra = expected - registered
    assert not missing, (
        f"New tool(s) without an EXPECTED_PROFILES entry: {sorted(missing)}. "
        f"Add an entry to tests/test_tool_metadata_audit.py::EXPECTED_PROFILES."
    )
    assert not extra, (
        f"EXPECTED_PROFILES references tool(s) that aren't registered: "
        f"{sorted(extra)}. Either register them or remove from the audit."
    )


def test_metadata_matches_expected_profile():
    """_TOOL_METADATA must match EXPECTED_PROFILES for every tool.

    Catches drift: e.g., the original Codex finding that manage_memory was
    marked mutating=False despite writing MEMORY.md.
    """
    failures: list[str] = []
    for name, (want_mutating, want_parallel, rationale) in EXPECTED_PROFILES.items():
        meta = _TOOL_METADATA.get(name, {})
        got_mutating = meta.get("mutating", False)
        got_parallel = meta.get("parallel_safe", False)
        if got_mutating != want_mutating:
            failures.append(
                f"  {name}: mutating={got_mutating} but expected {want_mutating} "
                f"({rationale})"
            )
        if got_parallel != want_parallel:
            failures.append(
                f"  {name}: parallel_safe={got_parallel} but expected {want_parallel} "
                f"({rationale})"
            )
    assert not failures, "Metadata drift detected:\n" + "\n".join(failures)


def test_mutating_tools_are_not_parallel_safe_by_default():
    """A mutating tool that's also parallel_safe is suspect — it means the
    tool can race with itself when batched. The only legitimate exception
    is check_background, whose 'kill' branch is the mutating path but
    polling status (the common case) is naturally idempotent.
    """
    EXPLICIT_EXCEPTIONS = {"check_background"}
    for name, meta in _TOOL_METADATA.items():
        if meta.get("mutating") and meta.get("parallel_safe"):
            assert name in EXPLICIT_EXCEPTIONS, (
                f"{name} is marked mutating=True AND parallel_safe=True. "
                f"This usually means concurrent invocations can race. "
                f"If this is intentional, add {name!r} to EXPLICIT_EXCEPTIONS "
                f"in this test with a comment explaining why."
            )


def test_tools_using_subprocess_are_marked_mutating():
    """Static-source heuristic: any tool whose function body mentions
    subprocess.Popen (directly or via a helper) must be marked mutating.
    Catches the case where a future tool adds a shell call without
    updating its metadata.

    Inspects the FUNCTION source (not the whole module) so a read-only
    tool that happens to live in the same file as run_shell isn't
    falsely flagged.
    """
    SUBPROCESS_SIGNALS = ("subprocess.Popen", "subprocess.run", "os.system",
                          "os.execv", "os.spawn")
    failures: list[str] = []
    for name, fn in _TOOL_FN_MAP.items():
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        if any(sig in src for sig in SUBPROCESS_SIGNALS):
            meta = _TOOL_METADATA.get(name, {})
            if not meta.get("mutating"):
                failures.append(
                    f"  {name}: function source uses subprocess but "
                    f"mutating={meta.get('mutating')}"
                )
    assert not failures, (
        "Tools whose source uses subprocess but aren't marked mutating:\n"
        + "\n".join(failures)
    )


def test_tools_using_atomic_write_are_marked_mutating():
    """Same idea for filesystem writers. atomic_write is jyagent's only
    blessed disk-mutation path, so any tool whose source imports it must
    be marked mutating.
    """
    failures: list[str] = []
    # Map module name -> set of tool names that live in it
    by_module: dict[str, list[str]] = {}
    for name, fn in _TOOL_FN_MAP.items():
        mod = inspect.getmodule(fn)
        if mod is None:
            continue
        by_module.setdefault(mod.__name__, []).append(name)

    for mod_name, tool_names in by_module.items():
        try:
            mod = __import__(mod_name, fromlist=["*"])
            src = inspect.getsource(mod)
        except (OSError, TypeError, ImportError):
            continue
        if "atomic_write" not in src:
            continue
        # The module writes to disk. Check that AT LEAST ONE tool in it is
        # marked mutating — coarse but catches the obvious drift case.
        any_mutating = any(
            _TOOL_METADATA.get(n, {}).get("mutating") for n in tool_names
        )
        if not any_mutating:
            failures.append(
                f"  module {mod_name} uses atomic_write but none of its "
                f"registered tools {tool_names} is marked mutating"
            )
    assert not failures, "\n".join(failures)


def test_no_orphan_metadata_entries():
    """Every key in _TOOL_METADATA must correspond to a registered tool.

    Catches the case where a tool gets removed but its metadata entry stays,
    silently misleading future readers about what's registered.
    """
    registered = set(_TOOL_FN_MAP.keys())
    metadata_keys = set(_TOOL_METADATA.keys())
    orphans = metadata_keys - registered
    assert not orphans, (
        f"_TOOL_METADATA has entries for unregistered tools: {sorted(orphans)}"
    )
