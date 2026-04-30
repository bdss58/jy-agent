"""LLMClient Protocol — the runtime's view of an LLM provider.

This module defines the **minimal contract** the agent loop engine depends on
from an LLM client.  Concrete provider classes (e.g. ``jyagent.llm.LLMOwner``)
satisfy this Protocol structurally — they need not subclass anything.

Why this exists
---------------
``jyagent/runtime/loop/engine.py`` used to import ``LLMOwner``,
``LLMOptions``, and ``ModelSpec`` directly from ``jyagent.llm`` — a
runtime-→-implementation dependency that reverses the intended direction
(the runtime should declare *what it needs*; provider packages should
*implement* that contract).

This Protocol fixes the **behavioural** half of that coupling:

  * Engine code now type-annotates with ``LLMClient`` instead of ``LLMOwner``.
  * Anyone (test fakes, alternative providers) can satisfy the contract
    without importing from ``jyagent.llm`` at all.
  * The engine's actual API surface is now self-documenting in one place.

The **value-type** half (``LLMOptions``, ``ModelSpec``) is still imported
from ``jyagent.llm.types`` — these are bag-of-fields data classes shared
across provider implementations.  Moving them into a neutral types
package is a separate, larger refactor; the import here is annotated as
"types only, no behavioural dependency".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Type-only imports: kept under TYPE_CHECKING so importing this module
    # does NOT eagerly import the LLM provider package.  Concrete providers
    # supply the real classes at runtime; the Protocol just describes the
    # shape.
    from .llm_types import LLMOptions, ModelSpec


@runtime_checkable
class LLMClient(Protocol):
    """Minimal LLM-call surface required by ``AgentLoop``.

    The runtime depends on **exactly three** things from the active LLM
    client.  Any object that supplies them — concrete provider, test
    fake, recorded-replay client — can drive the loop.

    Attributes
    ----------
    model_spec
        The active model identification (provider, model name, options).
        Read by the engine for tracing, cost accounting (so sub-agents on
        a different model tier bill against the correct pricing), and
        building reasoning-config defaults.

    Methods
    -------
    complete(context, *, options, model_spec=None) -> dict
        Non-streaming completion.  Returns the final assistant message in
        the runtime's normalized message-dict form.  ``model_spec`` may
        override the client's default for one call (sub-agent tier swap).

    stream(context, *, options, model_spec=None) -> Iterator
        Streaming completion.  Returns an iterator that yields provider-
        normalized stream events; the engine consumes these to update
        UI callbacks and assemble the final message.
    """

    @property
    def model_spec(self) -> "ModelSpec":
        ...

    def complete(
        self,
        context: dict,
        *,
        options: "LLMOptions",
        model_spec: "ModelSpec | None" = None,
    ) -> dict:
        ...

    def stream(
        self,
        context: dict,
        *,
        options: "LLMOptions",
        model_spec: "ModelSpec | None" = None,
    ) -> Iterator[Any]:
        ...


__all__ = ["LLMClient"]
