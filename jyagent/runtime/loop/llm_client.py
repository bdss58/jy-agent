"""LLMClient Protocol — the runtime's view of an LLM provider.

The runtime depends on exactly three things from an LLM client:
``model_spec``, ``complete()``, ``stream()``.  Any object that supplies
them (concrete provider, test fake, recorded-replay client) can drive
the loop — no inheritance required, structural typing suffices.

The detailed wire-format / normalized-message-shape contract that
providers must honour lives in two places:

  * ``jyagent.llm.types`` — the canonical TypedDict definitions
    (``Context``, ``Message``, ``AssistantMessage``, ``StreamEvent``,
    ``Usage``, etc.).
  * ``jyagent.llm.validation`` — runtime validators used when
    ``LoopConfig.validate_provider_output`` (or the
    ``JYAGENT_VALIDATE_PROVIDER_OUTPUT`` env var) is on.

Look there for the truth; this file deliberately keeps the Protocol
surface small and unburdened by prose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .llm_types import LLMOptions, ModelSpec
    from ...llm.types import AssistantMessage, Context, StreamEvent


@runtime_checkable
class LLMClient(Protocol):
    """Minimal LLM-call surface required by ``AgentLoop``."""

    @property
    def model_spec(self) -> "ModelSpec":
        ...

    def complete(
        self,
        context: "Context",
        *,
        options: "LLMOptions",
        model_spec: "ModelSpec | None" = None,
    ) -> "AssistantMessage":
        ...

    def stream(
        self,
        context: "Context",
        *,
        options: "LLMOptions",
        model_spec: "ModelSpec | None" = None,
    ) -> "Iterator[StreamEvent]":
        ...


__all__ = ["LLMClient"]
