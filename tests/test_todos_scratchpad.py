# tests/test_todos_scratchpad.py — Persistent TODO scratchpad regression tests.
#
# Validates the design decisions from the 2026-04-18 review:
#   * replace-all semantics matching Claude Code's TodoWrite
#   * 3-state enum (pending / in_progress / completed)
#   * closure-scoped tool factory (no ContextVar, no globals)
#   * todos live OUTSIDE messages — compaction never touches them
#   * todos injected as a <system-reminder> text block on the tail user msg
#   * base system_prompt stays untouched (Anthropic prefix-cache preservation)
#   * initial_todos on run(), not in LoopConfig

from __future__ import annotations

from dataclasses import dataclass

import pytest

from jyagent.runtime.loop import engine as le
from jyagent.runtime.loop.todos import (
    TodoItem,
    WRITE_TODOS_SCHEMA,
    build_write_todos_tool,
    format_todos_block,
    inject_todos_into_messages,
    normalize_todo,
    todo_to_dict,
)


# ─── Data model ──────────────────────────────────────────────────────────────


class TestTodoItemValidation:
    def test_valid_item(self):
        t = TodoItem(content="Run tests", status="pending")
        assert t.validate() is None

    def test_empty_content_invalid(self):
        t = TodoItem(content="", status="pending")
        assert "non-empty" in (t.validate() or "")

    def test_whitespace_content_invalid(self):
        t = TodoItem(content="   ", status="pending")
        assert t.validate() is not None

    def test_bad_status_invalid(self):
        t = TodoItem(content="x", status="bogus")
        err = t.validate()
        assert err and "pending" in err and "in_progress" in err and "completed" in err

    def test_all_three_statuses_accepted(self):
        for s in ("pending", "in_progress", "completed"):
            assert TodoItem(content="x", status=s).validate() is None


class TestNormalizeTodo:
    def test_from_dict_snake_case(self):
        t = normalize_todo({"content": "Do X", "status": "in_progress", "active_form": "Doing X"})
        assert t.content == "Do X"
        assert t.status == "in_progress"
        assert t.active_form == "Doing X"

    def test_from_dict_camel_case(self):
        """`activeForm` (Claude Code's schema) must also work."""
        t = normalize_todo({"content": "Y", "status": "pending", "activeForm": "Doing Y"})
        assert t.active_form == "Doing Y"

    def test_from_existing_todoitem_passthrough(self):
        orig = TodoItem(content="X", status="completed")
        assert normalize_todo(orig) is orig

    def test_non_dict_rejected(self):
        with pytest.raises(TypeError):
            normalize_todo("just a string")


# ─── Rendering & injection ───────────────────────────────────────────────────


class TestFormatTodosBlock:
    def test_empty_list_returns_empty_string(self):
        assert format_todos_block([]) == ""

    def test_renders_all_statuses_with_markers(self):
        block = format_todos_block([
            TodoItem(content="A", status="pending"),
            TodoItem(content="B", status="in_progress", active_form="Doing B"),
            TodoItem(content="C", status="completed"),
        ])
        assert "<system-reminder>" in block
        assert "</system-reminder>" in block
        assert "[ ] A" in block
        # in_progress prefers active_form
        assert "[>] Doing B" in block
        assert "[x] C" in block

    def test_in_progress_without_active_form_falls_back_to_content(self):
        block = format_todos_block([TodoItem(content="Task X", status="in_progress", active_form="")])
        assert "[>] Task X" in block


class TestInjectTodosIntoMessages:
    def test_empty_todos_returns_same_messages(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = inject_todos_into_messages(msgs, [])
        assert out is msgs  # same reference — no-op short circuit

    def test_injects_into_tail_user_string_content(self):
        msgs = [{"role": "user", "content": "what's up"}]
        todos = [TodoItem(content="Task", status="pending")]
        out = inject_todos_into_messages(msgs, todos)
        # Original not mutated
        assert msgs[0]["content"] == "what's up"
        # Output has the user message converted to a list with two text blocks
        assert out[-1]["role"] == "user"
        assert isinstance(out[-1]["content"], list)
        assert out[-1]["content"][0]["text"] == "what's up"
        assert "<system-reminder>" in out[-1]["content"][1]["text"]
        assert "[ ] Task" in out[-1]["content"][1]["text"]

    def test_injects_into_tail_user_list_content(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "content": "ok", "tool_use_id": "t1"},
            ],
        }]
        todos = [TodoItem(content="Task", status="pending")]
        out = inject_todos_into_messages(msgs, todos)
        # Original list unchanged
        assert len(msgs[0]["content"]) == 1
        # Output has one more block
        assert len(out[-1]["content"]) == 2
        assert out[-1]["content"][-1]["type"] == "text"
        assert "[ ] Task" in out[-1]["content"][-1]["text"]

    def test_tail_non_user_appends_new_message(self):
        """If the tail is an assistant msg (edge case), a fresh user message
        is appended so the reminder is definitely the last thing the model
        sees."""
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        todos = [TodoItem(content="X", status="pending")]
        out = inject_todos_into_messages(msgs, todos)
        assert len(out) == 3
        assert out[-1]["role"] == "user"
        assert "<system-reminder>" in out[-1]["content"]


# ─── write_todos tool factory ────────────────────────────────────────────────


class TestWriteTodosTool:
    def _make_tool(self):
        """Spin up a fresh closure-scoped tool over a local store."""
        store: list = []

        def get_store():
            return store

        def set_store(new):
            store[:] = new

        tool = build_write_todos_tool(get_store, set_store)
        return tool, lambda: store

    def test_replaces_whole_list(self):
        tool, get_store = self._make_tool()
        result = tool(todos=[
            {"content": "A", "status": "pending"},
            {"content": "B", "status": "in_progress", "activeForm": "Doing B"},
        ])
        assert not result.is_error
        assert len(get_store()) == 2
        assert get_store()[0].content == "A"
        assert get_store()[1].status == "in_progress"

    def test_subsequent_call_replaces_not_appends(self):
        tool, get_store = self._make_tool()
        tool(todos=[{"content": "first", "status": "pending"}])
        tool(todos=[{"content": "second", "status": "pending"}])
        assert len(get_store()) == 1
        assert get_store()[0].content == "second"

    def test_empty_list_clears(self):
        tool, get_store = self._make_tool()
        tool(todos=[{"content": "x", "status": "pending"}])
        result = tool(todos=[])
        assert not result.is_error
        assert get_store() == []

    def test_invalid_status_rejected(self):
        tool, get_store = self._make_tool()
        result = tool(todos=[{"content": "x", "status": "bogus"}])
        assert result.is_error
        assert "bogus" in result.content
        # Store unchanged on failure
        assert get_store() == []

    def test_non_list_input_rejected(self):
        tool, _ = self._make_tool()
        result = tool(todos="not a list")  # type: ignore[arg-type]
        assert result.is_error
        assert "array" in result.content.lower()

    def test_multiple_in_progress_warned_but_allowed(self):
        tool, get_store = self._make_tool()
        result = tool(todos=[
            {"content": "A", "status": "in_progress"},
            {"content": "B", "status": "in_progress"},
        ])
        # Soft guardrail: warning appears in summary, but store is updated.
        assert not result.is_error
        assert "in_progress simultaneously" in result.content
        assert len(get_store()) == 2


# ─── Isolation between closures ──────────────────────────────────────────────


class TestClosureIsolation:
    """Two AgentLoops running concurrently must have independent todo stores.
    ContextVar would be fragile across our daemon-thread tool executor; the
    closure-factory design avoids the issue entirely."""

    def test_two_independent_stores(self):
        store_a: list = []
        store_b: list = []
        tool_a = build_write_todos_tool(lambda: store_a, lambda v: store_a.__iadd__(v[:]) or store_a.__setitem__(slice(None), v))
        tool_b = build_write_todos_tool(lambda: store_b, lambda v: store_b.__setitem__(slice(None), v))

        tool_a(todos=[{"content": "only-a", "status": "pending"}])
        tool_b(todos=[{"content": "only-b", "status": "pending"}])

        assert len(store_a) == 1 and store_a[0].content == "only-a"
        assert len(store_b) == 1 and store_b[0].content == "only-b"


# ─── AgentLoop integration (no live LLM) ─────────────────────────────────────


class TestAgentLoopTodosWiring:
    """Verify run() plumbs todos correctly without actually calling an LLM.

    We poke the engine's internal state directly because spinning up a
    full streaming runtime for these specific assertions adds no signal."""

    def _make_loop(self, *, enabled: bool):
        class _Owner:
            class model_spec:
                provider = "anthropic"
                model = "claude-opus-4-6"

                @staticmethod
                def label():
                    return "anthropic:claude-opus-4-6"

        loop = le.AgentLoop.__new__(le.AgentLoop)
        loop._runtime_owner = _Owner()
        loop._config = le.LoopConfig(todos_enabled=enabled, max_steps=0)
        loop._callbacks = le.LoopCallbacks()
        loop._tool_source = None
        loop._model_spec = None
        loop._cancel_event = None
        loop._executor = le._tool_dispatch_executor
        loop._todos = []
        return loop

    def test_loopconfig_has_todos_enabled_field(self):
        cfg = le.LoopConfig()
        assert cfg.todos_enabled is False  # default off

    def test_loopresult_has_todos_field(self):
        r = le.LoopResult(status="completed", text="", final_text="", messages=[], steps=0)
        assert r.todos == []

    def test_run_without_todos_enabled_leaves_store_empty(self):
        """When disabled, initial_todos is silently ignored."""
        loop = self._make_loop(enabled=False)
        # max_steps=0 → loop runs zero iterations and exits immediately.
        result = loop.run("system", [], initial_todos=[{"content": "X", "status": "pending"}])
        # Because todos_enabled=False, no serialization happens.
        assert result.todos == []
        assert loop._todos == []  # never seeded

    def test_run_with_todos_enabled_seeds_initial(self):
        loop = self._make_loop(enabled=True)
        result = loop.run(
            "system", [],
            initial_todos=[{"content": "Step 1", "status": "pending"}],
        )
        # Todo was seeded; final snapshot returned as dicts (JSON-safe).
        assert len(loop._todos) == 1
        assert loop._todos[0].content == "Step 1"
        assert result.todos == [{"content": "Step 1", "status": "pending", "active_form": ""}]

    def test_run_ignores_malformed_initial_todos_with_warning(self):
        loop = self._make_loop(enabled=True)
        warnings: list[str] = []
        loop._callbacks = le.LoopCallbacks(on_warning=warnings.append)
        # A non-dict entry should trigger the graceful fallback.
        result = loop.run("system", [], initial_todos=[{"content": "ok", "status": "pending"}, 42])
        assert loop._todos == []
        assert any("initial_todos" in w for w in warnings)
        assert result.todos == []

    def test_initial_todos_none_preserves_cross_turn_state(self):
        """When ``initial_todos=None`` is passed (the default), any todos
        already on ``loop._todos`` from a prior ``run()`` MUST be preserved.
        This lets an outer session chain multiple turns on the same AgentLoop
        instance without restating the plan.
        """
        loop = self._make_loop(enabled=True)
        # First run seeds the todo store.
        loop.run("system", [], initial_todos=[{"content": "Step 1", "status": "pending"}])
        assert len(loop._todos) == 1
        first_snapshot = list(loop._todos)

        # Second run with initial_todos=None must NOT clear the store.
        result = loop.run("system", [], initial_todos=None)
        assert loop._todos == first_snapshot, (
            "initial_todos=None must preserve loop._todos across runs; "
            f"got {loop._todos!r} (was {first_snapshot!r})"
        )
        # And the result snapshot reflects the preserved store.
        assert result.todos == [
            {"content": "Step 1", "status": "pending", "active_form": ""}
        ]

    def test_initial_todos_empty_list_clears_store(self):
        """An explicit empty list (``initial_todos=[]``) MUST clear the store
        — that's the caller's escape hatch for "fresh start" when ``None``
        would otherwise preserve state.
        """
        loop = self._make_loop(enabled=True)
        loop.run("system", [], initial_todos=[{"content": "Step 1", "status": "pending"}])
        assert len(loop._todos) == 1

        result = loop.run("system", [], initial_todos=[])
        assert loop._todos == [], (
            "initial_todos=[] must clear loop._todos; "
            f"got {loop._todos!r}"
        )
        assert result.todos == []


# ─── Schema ──────────────────────────────────────────────────────────────────


class TestWriteTodosSchema:
    def test_has_required_shape(self):
        assert WRITE_TODOS_SCHEMA["name"] == "write_todos"
        schema = WRITE_TODOS_SCHEMA["input_schema"]
        assert schema["type"] == "object"
        assert schema["required"] == ["todos"]
        props = schema["properties"]
        assert props["todos"]["type"] == "array"
        item = props["todos"]["items"]
        assert item["required"] == ["content", "status"]
        assert set(item["properties"]["status"]["enum"]) == {"pending", "in_progress", "completed"}
