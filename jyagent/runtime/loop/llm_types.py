"""Runtime-owned LLM value types: ``LLMOptions`` and ``ModelSpec``.

Historical note
---------------
These two dataclasses used to live in ``jyagent.llm.types``.  The runtime
engine imported them from there, which inverted the intended dependency
direction: the runtime defines *what it needs from an LLM*; provider
packages *implement* that contract.

Codex review 2026-04-25 Part 3 #5 flagged the runtime → llm import as a
reversed dependency.  A first pass (commit ``4046792``) extracted the
*behavioural* contract into the ``LLMClient`` Protocol
(``runtime/loop/llm_client.py``).  This module closes the other half:
the *value types* that the engine constructs (``LLMOptions(...)``) and
threads through sub-agent tier swaps (``ModelSpec``) now live under the
runtime package itself.

``jyagent.llm.types`` re-exports the names so existing imports keep
working — this is a pure reorganisation, not an API break.  After this
move, ``jyagent.runtime`` has **zero** runtime-import of
``jyagent.llm``.

Why these two and not the whole Message/StreamEvent bestiary
------------------------------------------------------------
``LLMOptions`` and ``ModelSpec`` are *inputs* the runtime produces and
hands to an LLM client — they're the runtime's contract.  ``Message``,
``StreamEvent``, ``Context``, ``Usage``, etc. are the shape of *data
exchanged* with the client; they're owned equally by both sides and
moving them adds churn without clarifying ownership.  Keep them in
``jyagent.llm.types`` for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only imports: ``ReasoningConfig`` and ``ToolChoice`` are
    # TypedDict unions — at runtime they're just ``dict``, so the
    # dataclass annotations (stored as strings under ``from __future__
    # import annotations``) are never resolved during normal execution.
    # This keeps the TYPE_CHECKING block free of runtime cost and
    # avoids a circular import back into ``jyagent.llm``.
    from ...llm.types import ReasoningConfig, ToolChoice


@dataclass(frozen=True)
class ModelSpec:
    """Identifies an LLM: ``(provider, model)`` pair.

    Frozen + hashable so it can key pricing lookups, sub-agent tier
    caches, etc.
    """

    provider: str
    model: str

    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class LLMOptions:
    """Per-call knobs the engine passes to an ``LLMClient``.

    All fields optional; the engine fills in per-call defaults based on
    ``LoopConfig`` and the active ``ModelSpec``.
    """

    max_output_tokens: int | None = None
    timeout: float | None = None
    reasoning: "ReasoningConfig | None" = None
    metadata: dict[str, Any] | None = None
    tool_choice: "ToolChoice | None" = None


__all__ = ["LLMOptions", "ModelSpec"]
