# tests/test_context_mgmt_fixes.py — Tests for the May 2026 context-management
# fixes (codex self-review follow-ups):
#
#   Fix #1: structural-boundary-aware /compact (no orphan tool_results)
#   Fix #2: dehydration-aware tool-result clearing (spill-path preserved)
#   Fix #3: SKILL_PRE_ROUTER wired to main loop via pre_route_for_turn()

from unittest.mock import MagicMock, patch

import pytest


# ─── Fix #1: tool-pair boundary safety in memory/compaction.py ───────────────

class TestSplitPointBoundary:
    """`_choose_split_point` must not leave a tool_result as the first kept msg."""

    def test_plain_boundary_no_shift(self):
        from jyagent.memory.compaction import _choose_split_point
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
        # keep 3 → split at index 3 (assistant), no shift needed
        assert _choose_split_point(messages, keep_recent=3) == 3

    def test_shifts_past_top_level_tool_result(self):
        """If split would land on a top-level tool_result, walk left to the
        preceding assistant(tool_use) so the pair stays intact."""
        from jyagent.memory.compaction import _choose_split_point
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": [
                {"type": "tool_call", "id": "t0", "name": "run_shell",
                 "arguments": {"command": "ls"}},
            ]},
            {"role": "tool_result", "tool_call_id": "t0",
             "tool_name": "run_shell", "content": "out"},
            # ↑ splitting here (index 2) would orphan the tool_result
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        # keep_recent=3 → nominal split=2 (tool_result). Must walk back to 1.
        assert _choose_split_point(messages, keep_recent=3) == 1

    def test_shifts_past_anthropic_tool_result_block(self):
        """Anthropic convention: tool_result rides inside a user message's
        content list. Must still walk back past it."""
        from jyagent.memory.compaction import _choose_split_point
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t0", "name": "f", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t0", "content": "out"},
            ]},
            {"role": "assistant", "content": "a2"},
        ]
        assert _choose_split_point(messages, keep_recent=2) == 1

    def test_pathological_all_tool_results_lands_at_zero(self):
        """If the whole window up to the boundary is tool_results, we land at
        0 — the caller is expected to treat this as 'skip compaction'."""
        from jyagent.memory.compaction import _choose_split_point
        messages = [
            {"role": "tool_result", "tool_name": "f", "content": "x"},
            {"role": "tool_result", "tool_name": "f", "content": "x"},
            {"role": "tool_result", "tool_name": "f", "content": "x"},
        ]
        assert _choose_split_point(messages, keep_recent=2) == 0


class TestCompactConversationBoundary:
    """`compact_conversation` must skip when no safe split exists."""

    def test_no_safe_split_returns_compacted_false(self):
        """Pathological all-tool-result conversation → no compaction (vs the
        old behaviour of summarising an empty prefix)."""
        from jyagent.memory.conversation import ConversationMemory
        from jyagent.memory.compaction import compact_conversation

        conv = ConversationMemory()
        for _ in range(8):
            conv.add_message("tool_result", "x" * 100)

        runtime_owner = MagicMock()
        runtime_owner.complete_text.return_value = "SUMMARY"

        res = compact_conversation(
            conv, runtime_owner, keep_recent=2, system_prompt="SYS",
        )
        assert res["compacted"] is False
        # runtime_owner must not have been called — no summary requested
        runtime_owner.complete_text.assert_not_called()


# ─── Fix #2: dehydration pointer preservation ────────────────────────────────

class TestDehydrationPlaceholder:
    """Ephemeral/far-away tool results keep spill-file paths for rehydration."""

    def test_plain_clear_unchanged(self):
        """Results with no spill path fall back to the legacy marker so
        callers / existing tests that match on it still work."""
        from jyagent.runtime.loop.compaction import _dehydration_placeholder
        assert _dehydration_placeholder("hello world") == "[Tool result cleared]"
        assert _dehydration_placeholder("") == "[Tool result cleared]"

    def test_runshell_spill_preserved(self):
        from jyagent.runtime.loop.compaction import _dehydration_placeholder
        result = (
            "some output\n[stdout spilled to /tmp/jyagent_runshell_out_abc123]\n"
            "more output"
        )
        out = _dehydration_placeholder(result)
        assert "/tmp/jyagent_runshell_out_abc123" in out
        assert "recover via run_shell" in out

    def test_bg_spill_preserved(self):
        from jyagent.runtime.loop.compaction import _dehydration_placeholder
        result = '{"pid": 123, "output_file": "/tmp/jyagent_bg_xyz.out"}'
        out = _dehydration_placeholder(result)
        assert "/tmp/jyagent_bg_xyz.out" in out

    def test_multiple_paths_deduped_and_capped(self):
        from jyagent.runtime.loop.compaction import _dehydration_placeholder
        result = (
            "/tmp/jyagent_runshell_out_a "
            "/tmp/jyagent_runshell_out_a "  # duplicate
            "/tmp/jyagent_runshell_out_b "
            "/tmp/jyagent_runshell_out_c "
            "/tmp/jyagent_runshell_out_d "
        )
        out = _dehydration_placeholder(result)
        # Dedup: 'a' appears once
        assert out.count("jyagent_runshell_out_a") == 1
        # Cap at 3: 'd' dropped
        assert "jyagent_runshell_out_d" not in out
        assert "jyagent_runshell_out_a" in out
        assert "jyagent_runshell_out_b" in out
        assert "jyagent_runshell_out_c" in out

    def test_compact_messages_preserves_spill_path_top_level(self):
        """End-to-end: compact_messages on an ephemeral tool_result whose body
        references a spill path keeps the path in the cleared placeholder."""
        from jyagent.runtime.tools.registry import get_registry
        from jyagent.runtime.loop.compaction import compact_messages
        import jyagent.tools  # noqa: F401

        messages = []
        for i in range(10):
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": f"step {i}"}],
            })
            messages.append({
                "role": "tool_result",
                "tool_call_id": f"tc_{i}",
                "tool_name": "run_shell",
                "content": (
                    f"output of step {i} " + "padding " * 500
                    + f"[stdout spilled to /tmp/jyagent_runshell_out_step{i}]"
                ),
            })

        result = compact_messages(
            messages, max_tokens=500, compact_chars=2000,
            batch=get_registry().freeze(),
        )
        # Early results are both far-away AND ephemeral → cleared.
        # Their placeholder must reference the spill path.
        early_tr = result[1]["content"]
        assert "jyagent_runshell_out_step0" in early_tr, early_tr
        assert "recover via run_shell" in early_tr


# ─── Fix #3: SKILL_PRE_ROUTER actually runs ──────────────────────────────────

class TestSkillPreRouterWiring:
    """`pre_route_for_turn` must no-op when disabled and invoke the router
    when enabled."""

    def test_disabled_by_default_no_side_effect(self, tmp_path):
        from jyagent.skills import SkillManager
        mgr = SkillManager(skills_dir=str(tmp_path))
        mgr._auto_activate = False
        # Should return current active set (empty) without calling anything.
        assert mgr.pre_router_enabled is False
        with patch.object(mgr, "auto_activate_for_query") as spy:
            result = mgr.pre_route_for_turn("do something")
            spy.assert_not_called()
        assert result == []

    def test_enabled_calls_router(self, tmp_path):
        from jyagent.skills import SkillManager
        mgr = SkillManager(skills_dir=str(tmp_path))
        mgr._auto_activate = True
        # Need at least one skill for routing to trigger
        mgr._skills = {"test-skill": {"name": "test-skill", "description": "x", "path": "/tmp/x"}}

        assert mgr.pre_router_enabled is True
        with patch.object(mgr, "auto_activate_for_query",
                          return_value=["test-skill"]) as spy:
            result = mgr.pre_route_for_turn("test it", runtime_owner=None,
                                            recent_messages=[])
            spy.assert_called_once()
        assert result == ["test-skill"]

    def test_empty_query_no_op_even_when_enabled(self, tmp_path):
        """Router should not run on empty query (e.g. slash commands handled
        upstream returning early)."""
        from jyagent.skills import SkillManager
        mgr = SkillManager(skills_dir=str(tmp_path))
        mgr._auto_activate = True
        mgr._skills = {"s": {"name": "s", "description": "x", "path": "/tmp/s"}}

        with patch.object(mgr, "auto_activate_for_query") as spy:
            mgr.pre_route_for_turn("")
            spy.assert_not_called()
