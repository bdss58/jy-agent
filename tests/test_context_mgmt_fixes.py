# tests/test_context_mgmt_fixes.py — Tests for the May 2026 context-management
# fixes (codex self-review follow-ups):
#
#   Fix #1: structural-boundary-aware /compact (no orphan tool_results)
#   Fix #2: dehydration-aware tool-result clearing (spill-path preserved)
#   Fix #3: (REMOVED 2026-05) SKILL_PRE_ROUTER deleted — per-turn auto-router gone

import os
import tempfile
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


# ─── Fix #3 (REMOVED): SKILL_PRE_ROUTER + the whole routing mechanism were
# removed 2026-05.  Per-turn auto-routing is gone; skills activate only via
# manage_skills tool / `/skill` command.  Eval tooling for "would query X
# trigger skill Y?" lives self-contained in
# skills/create-skill/scripts/test_trigger.py.


# ─── Fix #4: sub-agent memory scoping ────────────────────────────────────────

class TestSubagentMemoryScoping:
    """`_get_memory_context(query, mode)` must respect the mode gate."""

    def test_default_mode_is_none_no_leak(self, monkeypatch, tmp_path):
        """Default memory mode is 'none' — no MEMORY.md leaks into sub-agents."""
        from jyagent.tools import subagent

        # Even with a populated MEMORY.md, default mode returns empty.
        fake_md = tmp_path / "MEMORY.md"
        fake_md.write_text("# sensitive user memory\n- rule 1\n")
        monkeypatch.setattr("jyagent.config.MEMORY_MD_FILE", str(fake_md))

        assert subagent._get_memory_context() == ""
        assert subagent._get_memory_context(query="anything", mode="none") == ""

    def test_matched_mode_requires_query(self, monkeypatch, tmp_path):
        """mode='matched' with empty query → empty (no BM25 query possible)."""
        from jyagent.tools import subagent
        assert subagent._get_memory_context(query="", mode="matched") == ""

    def test_matched_mode_uses_bm25_search(self, monkeypatch):
        """mode='matched' delegates to search_memory; empty hits → empty output."""
        from jyagent.tools import subagent

        called = {}
        def fake_search(query, top_k=5):
            called["query"] = query
            called["top_k"] = top_k
            return []  # no hits
        monkeypatch.setattr("jyagent.memory.search.search_memory", fake_search)

        out = subagent._get_memory_context(query="install nginx", mode="matched")
        assert out == ""  # empty hits → empty context
        assert called["query"] == "install nginx"

    def test_invalid_mode_silently_empty(self):
        """Unknown modes must not raise — they degrade to empty."""
        from jyagent.tools import subagent
        assert subagent._get_memory_context(query="x", mode="bogus") == ""

    def test_dispatch_schema_advertises_memory_mode(self):
        """The public schema must expose memory_mode so the LLM can set it."""
        from jyagent.tools.schemas import SUBAGENT_SCHEMA as TOOL_SCHEMA
        props = TOOL_SCHEMA["input_schema"]["properties"]
        assert "memory_mode" in props
        assert set(props["memory_mode"]["enum"]) == {"none", "matched"}


# ─── Fix #5a: pricing coverage for the default model ────────────────────────

class TestPricingCoverage:
    """The default model must have a pricing entry so session cost isn't
    silently reported as 'unknown' on every default run."""

    def test_default_sonnet_has_pricing(self):
        from jyagent.runtime.stats import _lookup_pricing
        from jyagent.config import DEFAULT_ANTHROPIC_MODEL

        pricing = _lookup_pricing("anthropic", DEFAULT_ANTHROPIC_MODEL)
        assert pricing is not None, (
            f"No pricing for default model {DEFAULT_ANTHROPIC_MODEL!r} — "
            f"session cost would show 'unknown'"
        )
        # Sonnet tier: $3 input / $15 output (verified against
        # platform.claude.com pricing page, 2026-05-01).
        assert pricing.input_per_million == 3.0
        assert pricing.output_per_million == 15.0
        # Cache ratios must be correct or cache savings are mis-reported.
        assert pricing.cache_creation_per_million == 3.75   # 1.25x input
        assert pricing.cache_read_per_million == 0.30       # 0.1x input


# ─── Fix #5b: sub-agent API-call + cache accounting ─────────────────────────

class TestSubagentAccounting:
    """`record_subagent_usage` must honour the child's real api_calls and
    cache tokens instead of the legacy '1 per dispatch, 0 cache' roll-up."""

    def test_api_calls_uses_child_count_when_provided(self):
        from jyagent.runtime.stats import SessionStats

        s = SessionStats()
        before = s.api_calls
        s.record_subagent_usage(
            input_tokens=100, output_tokens=50,
            provider="anthropic", model="claude-sonnet-4-6",
            api_calls=7,  # child ran 7 API calls
        )
        assert s.api_calls == before + 7

    def test_api_calls_falls_back_to_one_when_zero(self):
        """Legacy callers that don't plumb api_calls still see the sub-agent
        counted as one unit of API activity (safer than +0)."""
        from jyagent.runtime.stats import SessionStats

        s = SessionStats()
        before = s.api_calls
        s.record_subagent_usage(
            input_tokens=100, output_tokens=50,
            provider="anthropic", model="claude-sonnet-4-6",
            # api_calls omitted → default 0 → fallback +1
        )
        assert s.api_calls == before + 1

    def test_cache_tokens_accumulate(self):
        from jyagent.runtime.stats import SessionStats

        s = SessionStats()
        s.record_subagent_usage(
            input_tokens=100, output_tokens=50,
            provider="anthropic", model="claude-sonnet-4-6",
            cache_creation_tokens=1000,
            cache_read_tokens=5000,
            api_calls=3,
        )
        assert s.total_cache_creation_tokens == 1000
        assert s.total_cache_read_tokens == 5000

    def test_subagent_run_records_new_fields(self):
        from jyagent.runtime.stats import SessionStats

        s = SessionStats()
        s.record_subagent_usage(
            input_tokens=100, output_tokens=50,
            provider="anthropic", model="claude-sonnet-4-6",
            task_preview="hello",
            steps=4, tool_calls=6,
            cache_creation_tokens=200, cache_read_tokens=800,
            api_calls=5,
        )
        assert len(s.subagent_runs) == 1
        rec = s.subagent_runs[0]
        assert rec["cache_creation_tokens"] == 200
        assert rec["cache_read_tokens"] == 800
        assert rec["api_calls"] == 5


# ─── Fix #5c: LoopResult carries cache tokens + api_calls ───────────────────

class TestLoopResultCarriesCacheAndApiCalls:
    """LoopResult must expose total_cache_creation_tokens / total_cache_read_tokens /
    api_calls so the sub-agent accounting path can read them."""

    def test_loopresult_has_new_fields(self):
        from jyagent.runtime.loop.config import LoopResult
        r = LoopResult(
            status="completed", text="ok", final_text="ok",
            messages=[], steps=1,
            total_input_tokens=10, total_output_tokens=5,
            tool_calls_count=0,
            total_cache_creation_tokens=100,
            total_cache_read_tokens=200,
            api_calls=3,
        )
        assert r.total_cache_creation_tokens == 100
        assert r.total_cache_read_tokens == 200
        assert r.api_calls == 3

    def test_loopresult_new_fields_default_zero(self):
        from jyagent.runtime.loop.config import LoopResult
        r = LoopResult(
            status="completed", text="ok", final_text="ok",
            messages=[], steps=1,
            total_input_tokens=10, total_output_tokens=5,
            tool_calls_count=0,
        )
        assert r.total_cache_creation_tokens == 0
        assert r.total_cache_read_tokens == 0
        assert r.api_calls == 0
