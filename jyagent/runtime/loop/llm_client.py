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
    # Normalized message / context / stream types live in jyagent.llm.types
    # (TypedDicts — see jyagent/llm/validation.py for the runtime validators).
    # Imported under TYPE_CHECKING to keep the Protocol import cheap.
    from ...llm.types import AssistantMessage, Context, StreamEvent


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

    Normalized data contract (the runtime depends on these shapes)
    --------------------------------------------------------------
    Provider adapters (``jyagent.llm.providers.*``) MUST decode their
    wire format into these normalized shapes — the engine
    (``runtime/loop/llm_runner.py``, ``runtime/loop/engine.py``) reads
    them directly.  Diverging from these shapes will silently break the
    loop with no type error (everything is plain ``dict``).

    1. **Request context** (``complete()`` / ``stream()`` arg)::

           {
               "system_prompt": str,        # MUST stay byte-stable across a
                                            #   session — see MEMORY.md prompt-
                                            #   caching rule.  Dynamic context
                                            #   goes in tail user messages.
               "messages": list[dict],      # ordered turns; see (2).
           }

    2. **Message** (entries in ``context["messages"]``, and the dict
       returned by ``complete()`` / yielded as ``done.message`` by
       ``stream()``)::

           {
               "role": "user" | "assistant",
               "content": list[ContentBlock],
               "stop_reason": str,          # assistant only; see (4).
               "usage": dict,               # assistant only; see (5).
           }

    3. **Content blocks** (entries in ``message["content"]``).  Discriminated
       by ``type``::

           {"type": "text",        "text": str}
           {"type": "thinking",    "thinking": str, ...}    # provider-specific
                                                            #   extras allowed;
                                                            #   compaction
                                                            #   strips these
                                                            #   from old turns.
           {"type": "tool_call",   "id": str,
                                   "name": str,
                                   "arguments": dict}       # the runtime's
                                                            #   normalized form
                                                            #   of an Anthropic
                                                            #   ``tool_use`` /
                                                            #   OpenAI
                                                            #   ``tool_calls``
                                                            #   entry.
           {"type": "tool_result", "tool_use_id": str,
                                   "content": str | list,
                                   "is_error": bool}        # appended by the
                                                            #   engine after
                                                            #   tool dispatch.

    4. **stop_reason** values the engine special-cases (provider adapters
       MUST normalize to these strings)::

           "stop"        — natural completion (terminal turn condition).
           "tool_use"    — assistant emitted tool_call blocks; loop continues.
           "length"      — output truncated by max_tokens.  Triggers the
                           truncation-retry / token-scale path in step.py.
           "error"       — streaming adapter only; final_message carries an
                           ``error_message`` field that becomes the retry
                           layer's exception text.

    5. **Usage** (``message["usage"]``).  All keys optional; missing keys
       are treated as zero by the cost tracker::

           {
               "input_tokens":         int,
               "output_tokens":        int,
               "cache_creation_input_tokens": int,  # Anthropic prompt-cache
               "cache_read_input_tokens":     int,  # Anthropic prompt-cache
           }

    6. **Stream events** (``stream()`` iterator yields these dicts).  The
       engine in ``llm_runner._stream_loop`` switches on ``event["type"]``::

           {"type": "text_delta",        "text": str}
           {"type": "thinking_start"}
           {"type": "thinking_delta",    "text": str}
           {"type": "thinking_end"}
           {"type": "tool_call_start",   ...}    # used only to drop thinking
           {"type": "tool_call_delta",   ...}    # used only to drop thinking
           {"type": "done",  "message":  Message}     # see (2).
           {"type": "error", "message":  Message}     # final_message with
                                                      # stop_reason="error".

       The stream object MAY also expose ``get_final_message()`` for the
       fallback-flush path; this is optional but the Anthropic adapter
       provides it.

    Why the runtime owns these shapes
    ---------------------------------
    Provider adapters are the natural home for wire-format quirks
    (Anthropic's ``tool_use``, OpenAI's ``tool_calls``, signed
    ``thinking`` blocks, reasoning items).  Concentrating wire-format
    knowledge in the adapters lets the engine reason about a single
    stable shape — tested via the ``test_step_runner.py`` and
    ``test_loop_edge_cases.py`` suites which build messages by hand
    against this contract.
    """

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
