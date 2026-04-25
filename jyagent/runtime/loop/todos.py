# jyagent/todos.py — Persistent TODO / plan scratchpad for the agent loop.
#
# Gives the model a structured task list that SURVIVES context compaction.
# The list is NOT stored in `messages` — it lives on the AgentLoop instance
# and is re-rendered into a tail user-message text block immediately before
# each LLM call.  That way:
#
#   * Compaction (which operates on `messages`) never touches it.
#   * The base system_prompt stays stable across todo updates, which
#     preserves Anthropic's `tools → system → messages` prefix cache.
#   * Only the final (already-uncached) user message is modified, which is
#     the lowest-cost injection point.
#
# Mirrors Claude Code's TodoWrite: a single `write_todos(todos)` tool with
# replace-all semantics and a 3-state status enum.  The tool is produced by
# `build_write_todos_tool(agent_loop)` so each live AgentLoop gets its own
# closure — no ContextVar, no thread-local state, no global mutable singleton.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..tools.result import ToolResult


# ─── Types ───────────────────────────────────────────────────────────────────

_ALLOWED_STATUS = ("pending", "in_progress", "completed")


@dataclass
class TodoItem:
    """Single entry in the agent's task plan.

    Fields match Claude Code's TodoWrite schema:
      content     — imperative description ("Refactor the auth module")
      status      — one of pending | in_progress | completed
      active_form — present-continuous phrasing shown while status=in_progress
                    ("Refactoring the auth module")
    """

    content: str
    status: str = "pending"
    active_form: str = ""

    def validate(self) -> str | None:
        """Return an error string if invalid, None if OK."""
        if not isinstance(self.content, str) or not self.content.strip():
            return "todo.content must be a non-empty string"
        if self.status not in _ALLOWED_STATUS:
            return (
                f"todo.status must be one of {list(_ALLOWED_STATUS)}, "
                f"got {self.status!r}"
            )
        if self.active_form and not isinstance(self.active_form, str):
            return "todo.active_form must be a string when provided"
        return None


def normalize_todo(raw: dict | TodoItem) -> TodoItem:
    """Coerce a dict (from tool input) or existing TodoItem into TodoItem."""
    if isinstance(raw, TodoItem):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"todo must be a dict or TodoItem, got {type(raw).__name__}")
    return TodoItem(
        content=str(raw.get("content", "")).strip(),
        status=str(raw.get("status", "pending")),
        active_form=str(raw.get("activeForm", raw.get("active_form", ""))),
    )


def todo_to_dict(t: TodoItem) -> dict[str, Any]:
    """Serialize for persistence.  Uses snake_case internally."""
    return asdict(t)


# ─── Rendering ───────────────────────────────────────────────────────────────

_STATUS_MARKER = {
    "completed": "[x]",
    "in_progress": "[>]",
    "pending": "[ ]",
}


def format_todos_block(todos: list[TodoItem]) -> str:
    """Render todos as a human-readable Markdown block for the model.

    Wrapped in ``<system-reminder>`` sentinels so the model treats it as
    an authoritative status reminder, not user content.
    """
    if not todos:
        return ""
    lines = ["<system-reminder>", "Current task plan:"]
    for t in todos:
        marker = _STATUS_MARKER.get(t.status, "[?]")
        label = t.active_form if (t.status == "in_progress" and t.active_form) else t.content
        lines.append(f"  {marker} {label}")
    lines.append(
        "Keep this plan accurate.  Use the `write_todos` tool to mark items "
        "in_progress / completed as you make progress.  Only one item should "
        "be in_progress at a time."
    )
    lines.append("</system-reminder>")
    return "\n".join(lines)


def inject_todos_into_messages(messages: list, todos: list[TodoItem]) -> list:
    """Return a shallow-copied messages list with the todos reminder
    appended as a text block to the final user message.

    Does NOT mutate the caller's list or the original message dict.  When
    the tail is an assistant message (rare edge case) or the list is empty,
    a fresh user message is appended.
    """
    if not todos:
        return messages
    block = format_todos_block(todos)
    if not block:
        return messages

    out = list(messages)

    if not out or out[-1].get("role") != "user":
        # Edge case: append a standalone user message so the reminder is
        # the last thing the model sees.
        out.append({"role": "user", "content": block})
        return out

    # Clone the tail user message and append a text block to its content.
    tail = dict(out[-1])
    content = tail.get("content", "")
    if isinstance(content, str):
        tail["content"] = [
            {"type": "text", "text": content},
            {"type": "text", "text": block},
        ]
    elif isinstance(content, list):
        tail["content"] = list(content) + [{"type": "text", "text": block}]
    else:
        # Unknown shape — fall back to appending a separate user message.
        out.append({"role": "user", "content": block})
        return out
    out[-1] = tail
    return out


# ─── Tool ────────────────────────────────────────────────────────────────────

WRITE_TODOS_SCHEMA: dict[str, Any] = {
    "name": "write_todos",
    "description": (
        "Replace the agent's current task plan with the given list of todo "
        "items.  Use this tool to (1) draft a plan at the start of a complex "
        "task, (2) mark items in_progress as you start them, and (3) mark "
        "items completed as you finish them.  The plan is injected into "
        "every subsequent step and SURVIVES context compaction, so use it "
        "to keep track of long-horizon work.  Only one item should be "
        "in_progress at a time.  Replace-all semantics: emit the FULL list "
        "on every call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The complete, current task plan (replaces the previous list).",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Imperative description of the task.",
                        },
                        "status": {
                            "type": "string",
                            "enum": list(_ALLOWED_STATUS),
                            "description": "Task status.",
                        },
                        "activeForm": {
                            "type": "string",
                            "description": (
                                "Present-continuous phrasing shown while "
                                "status=in_progress (e.g. 'Running tests')."
                            ),
                        },
                    },
                    "required": ["content", "status"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["todos"],
        "additionalProperties": False,
    },
}


def build_write_todos_tool(
    get_store: Callable[[], list[TodoItem]],
    set_store: Callable[[list[TodoItem]], None],
) -> Callable[..., ToolResult]:
    """Build a closure-scoped ``write_todos`` tool for a specific AgentLoop.

    Passing explicit get/set lambdas rather than a shared object keeps this
    trivially thread-safe (the caller controls the store's lifetime) and
    avoids any module-level mutable state.
    """

    def write_todos(todos: list[dict]) -> ToolResult:
        if not isinstance(todos, list):
            return ToolResult(
                "Error: `todos` must be an array of todo objects.", is_error=True,
            )
        try:
            parsed = [normalize_todo(t) for t in todos]
        except TypeError as e:
            return ToolResult(f"Error: {e}", is_error=True)

        errors = []
        for idx, t in enumerate(parsed):
            err = t.validate()
            if err:
                errors.append(f"  todos[{idx}]: {err}")
        if errors:
            return ToolResult(
                "Error: invalid todos:\n" + "\n".join(errors), is_error=True,
            )

        # Soft guardrail: warn if multiple in_progress.  Not a hard error —
        # models sometimes legitimately track concurrent work.
        in_prog = sum(1 for t in parsed if t.status == "in_progress")
        set_store(parsed)

        summary_lines = [f"Task plan updated ({len(parsed)} item(s)):"]
        for t in parsed:
            marker = _STATUS_MARKER.get(t.status, "[?]")
            summary_lines.append(f"  {marker} {t.content}")
        if in_prog > 1:
            summary_lines.append(
                f"Warning: {in_prog} items are in_progress simultaneously. "
                "Prefer one in-flight task at a time."
            )
        return ToolResult("\n".join(summary_lines))

    return write_todos


__all__ = [
    "TodoItem",
    "WRITE_TODOS_SCHEMA",
    "build_write_todos_tool",
    "format_todos_block",
    "inject_todos_into_messages",
    "normalize_todo",
    "todo_to_dict",
]
