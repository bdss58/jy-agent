# Structured tool result — shared by planner and tool implementations.


class ToolResult:
    """Structured tool result with explicit error flag.

    Benefits over raw strings:
    - Anthropic API supports `is_error: true` in tool_result blocks, which helps
      Claude reason better about failures vs. successful results containing "Error"
    - Error results are never truncated
    - Clear programmatic distinction between success and failure

    String-like interface: delegates common str methods to self.content so that
    existing code using ``"text" in result`` or ``result.startswith(...)`` keeps working.
    """
    __slots__ = ('content', 'is_error', 'duration_ms')

    def __init__(self, content: str, is_error: bool = False, duration_ms: float | None = None):
        self.content = content
        self.is_error = is_error
        # Wall-clock duration of the tool call in milliseconds. Stamped by the
        # tool executor at the 4 dispatch sites in ``execute_tools`` so the UI
        # can surface per-call timings without re-measuring (which would be
        # wrong for parallel batches whose UI ``on_tool_end`` callbacks fire
        # serially in submission order after all bodies have completed).
        # ``None`` = not measured (direct caller or legacy construction).
        self.duration_ms = duration_ms

    def __str__(self):
        return self.content

    def __contains__(self, item):
        return item in self.content

    def __repr__(self):
        return f"ToolResult(is_error={self.is_error}, content={self.content[:80]!r}...)"

    # Delegate common str methods for backward compatibility
    def startswith(self, *args):
        return self.content.startswith(*args)

    def endswith(self, *args):
        return self.content.endswith(*args)

    def lower(self):
        return self.content.lower()

    def upper(self):
        return self.content.upper()

    def strip(self, *args):
        return self.content.strip(*args)

    def split(self, *args):
        return self.content.split(*args)

    def replace(self, *args):
        return self.content.replace(*args)

    def count(self, *args):
        return self.content.count(*args)

    def __len__(self):
        return len(self.content)

    def __add__(self, other):
        return self.content + other

    def __radd__(self, other):
        return other + self.content

    def __getitem__(self, key):
        return self.content[key]
