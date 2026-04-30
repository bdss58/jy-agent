"""LLM call stack — streaming, complete, retry, fallback, cancellation.

Extracted from ``engine.py`` to keep the loop controller focused on
orchestration.  This module owns everything between "AgentLoop decided it
wants another inference" and "we have an AssistantMessage + tool calls back":

    * Streaming event consumption (``LLMRunner.call_streaming``) —
      includes the C1 cancellation watcher that closes a network-stuck
      stream's underlying httpx response.
    * Non-streaming (``LLMRunner.call_complete``) — includes the C1
      daemon-worker + 100 ms poll pattern for prompt cancel latency.
    * Transient-error retry with jittered exponential backoff
      (``LLMRunner.call_with_retry``), including the on_stream_retry
      partial-text propagation.
    * Free-function helpers: ``extract_text``, ``extract_tool_calls``,
      ``is_transient_error``, ``build_runtime_options``.

Unlike the plain-alias helper modules, this module uses a ``LLMRunner`` class
because the call stack has too much injected state (runtime owner, config,
callbacks, cancel_event, model_spec) for a flat-function translation to be
clean.  ``AgentLoop`` constructs one per instance and keeps thin delegation
methods for back-compat with callers (and tests that mock ``_call_complete``
etc.).

The ``_is_cancelled`` / ``_cancellable_sleep`` / ``_fire`` helpers live on
the ``LoopThreadHelper`` mixin (``_thread_helpers.py``) — both ``LLMRunner``
and ``AgentLoop`` inherit them and override the helper's two class-level
attribute-name strings to point at their respective instance attributes.
This was previously ~30 lines of cut-and-paste between the two classes.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from typing import TYPE_CHECKING

from .callbacks import LoopCallbacks
from .config import LoopConfig
from .llm_client import LLMClient
from .llm_types import LLMOptions, ModelSpec, ToolCallRequest
from ._thread_helpers import LoopThreadHelper
from ...config import get_reasoning_config_for_provider, STREAM_TIMEOUT


_logger = logging.getLogger(__name__)


# ─── Free-function helpers ──────────────────────────────────────────────────


def extract_text(message: dict) -> str:
    """Extract concatenated text blocks from an AssistantMessage."""
    return "".join(
        block.get("text", "")
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


def extract_tool_calls(message: dict) -> list[ToolCallRequest]:
    """Extract tool_call blocks from an AssistantMessage."""
    return [
        ToolCallRequest(
            id=block["id"],
            name=block["name"],
            input=block.get("arguments", {}),
        )
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_call"
    ]


def is_transient_error(error: BaseException) -> bool:
    """Return True if the error is likely transient and worth retrying.

    Checks concrete exception types first to avoid false positives from
    keyword-matching against arbitrary error messages.
    """
    # --- Network / transport layer (always transient) ---
    import httpx  # local import to avoid hard dependency at module level
    if isinstance(error, (
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        httpx.RemoteProtocolError,
        ConnectionResetError,
        BrokenPipeError,
        ConnectionAbortedError,
    )):
        return True

    # --- Provider SDK errors (transient if server-side) ---
    try:
        import anthropic as _anth
        # 424 = proxy/gateway envelope error (e.g. domestic relay wrapping
        # upstream flakes). Treat as transient; if the inner cause is a
        # deterministic 4xx (quota/billing/bad-request) it will simply
        # re-raise after burning the retry budget — acceptable tradeoff
        # since most 424s we see in practice are upstream transients.
        if isinstance(error, _anth.APIStatusError) and error.status_code in (424, 429, 500, 502, 503, 529):
            return True
        if isinstance(error, (_anth.APIConnectionError, _anth.APITimeoutError)):
            return True
    except ImportError:
        pass
    try:
        import openai as _oai
        # See comment above on 424 for rationale (symmetric treatment for
        # OpenAI-compatible proxies). OpenAI-canonical services never emit
        # 424, so this only fires for proxy/relay envelopes.
        if isinstance(error, _oai.APIStatusError) and error.status_code in (424, 429, 500, 502, 503):
            return True
        if isinstance(error, (_oai.APIConnectionError, _oai.APITimeoutError)):
            return True
    except ImportError:
        pass

    # --- JSON decode failure (often a truncated stream response) ---
    if isinstance(error, json.JSONDecodeError):
        return True

    # --- Fallback: keyword match, but only for generic / unknown types ---
    msg = str(error).lower()
    transient_keywords = [
        "overloaded", "server_error", "peer closed",
        "connection reset", "broken pipe",
    ]
    return any(kw in msg for kw in transient_keywords)


def build_runtime_options(
    runtime_owner: LLMClient,
    max_output_tokens: int,
    model_spec: ModelSpec | None = None,
    metadata: dict | None = None,
    session_id: str | None = None,
) -> LLMOptions:
    """Build LLMOptions with reasoning config for the active provider."""
    spec = model_spec or runtime_owner.model_spec
    merged_metadata = dict(metadata or {})
    if session_id:
        merged_metadata["session_id"] = session_id
    return LLMOptions(
        max_output_tokens=max_output_tokens,
        timeout=STREAM_TIMEOUT,
        reasoning=get_reasoning_config_for_provider(
            spec.provider,
            max_output_tokens=max_output_tokens,
            model=spec.model,
        ),
        metadata=merged_metadata,
    )


# ─── LLMRunner ──────────────────────────────────────────────────────────────


class LLMRunner(LoopThreadHelper):
    """Per-AgentLoop LLM I/O orchestrator.

    Owns the three entry points (``call_complete``, ``call_streaming``,
    ``call_with_retry``) plus the C1 cancellation watchers and the retry
    loop.  AgentLoop constructs one per instance and delegates its
    ``_call_complete`` / ``_call_streaming`` / ``_call_llm_with_retry``
    methods to it.

    State (all injected by AgentLoop, never mutated post-__init__):
      * runtime_owner — the LLMOwner (provider-neutral façade)
      * config — the loop's LoopConfig (for streaming flag, retry count,
        buffered_streaming, etc.)
      * callbacks — LoopCallbacks for on_text_delta, on_thinking_*,
        on_retry, on_stream_retry, on_usage
      * cancel_event — optional threading.Event for cooperative cancel
      * model_spec — optional override (sub-agent tier)

    Inherits ``_is_cancelled`` / ``_cancellable_sleep`` / ``_fire`` from
    ``LoopThreadHelper``.  Overrides the
    helper's two attribute-name class-vars because LLMRunner uses
    un-prefixed instance attribute names (``cancel_event`` / ``callbacks``)
    while AgentLoop uses underscore-prefixed names.
    """

    # Tell the LoopThreadHelper mixin which instance attributes hold
    # the cancel event and callbacks dataclass.
    _helper_cancel_event_attr = "cancel_event"
    _helper_callbacks_attr = "callbacks"

    def __init__(
        self,
        runtime_owner: LLMClient,
        config: LoopConfig,
        callbacks: LoopCallbacks,
        cancel_event: threading.Event | None = None,
        model_spec: ModelSpec | None = None,
    ):
        self.runtime_owner = runtime_owner
        self.config = config
        self.callbacks = callbacks
        self.cancel_event = cancel_event
        self.model_spec = model_spec

    # ── cancellation + callback helpers from LoopThreadHelper ───────────────
    # ``_is_cancelled``, ``_cancellable_sleep``, ``_fire`` live on the
    # ``LoopThreadHelper`` mixin (see ``_thread_helpers.py``).  Class-var
    # overrides at the top of this class point the helper's attribute
    # lookups at LLMRunner's un-prefixed ``cancel_event`` / ``callbacks``
    # instance attributes.

    # ── non-streaming call ───────────────────────────────────────────────

    def call_complete(
        self,
        context: dict,
        options: LLMOptions,
    ) -> tuple[str, list[ToolCallRequest], str, dict]:
        """Non-streaming: runtime_owner.complete() -> extract text/tool_calls.

        The sync SDK call is a single
        blocking network operation with zero yield points, so cooperative
        cancellation between yields (which the streaming path has) isn't
        possible here.  Run the call in a daemon worker thread and poll
        ``cancel_event`` from the main thread every 100 ms.  On cancel we
        raise ``KeyboardInterrupt`` immediately; the worker thread keeps
        running in the background (same trade-off as the mutating-tool
        timeout path — Python threads aren't cancellable, so the network
        request leaks until it completes).  Without this, a cancel during
        a slow ``complete()`` call would wait up to the provider HTTP
        timeout (60-300 s typical).

        If no cancel_event is attached (the common non-CLI path), the
        call runs synchronously in the current thread — no worker-thread
        overhead.
        """
        # Pre-call check: no point issuing the request if already cancelled.
        if self._is_cancelled():
            raise KeyboardInterrupt("cancelled before complete()")

        if self.cancel_event is None:
            # Fast path: no cancel machinery needed.
            final_message = self.runtime_owner.complete(
                context, options=options, model_spec=self.model_spec,
            )
        else:
            # Cancellable path: daemon worker + 100 ms poll.
            final_holder: list[dict | None] = [None]
            exc_holder: list[BaseException | None] = [None]
            done = threading.Event()

            def _worker() -> None:
                try:
                    final_holder[0] = self.runtime_owner.complete(
                        context, options=options, model_spec=self.model_spec,
                    )
                except BaseException as e:  # noqa: BLE001
                    exc_holder[0] = e
                finally:
                    done.set()

            t = threading.Thread(
                target=_worker,
                name="jyagent-llm-complete",
                daemon=True,
            )
            t.start()
            while not done.wait(0.1):
                if self._is_cancelled():
                    _logger.info(
                        "cancel signalled during complete(); "
                        "worker thread will drain in background",
                    )
                    raise KeyboardInterrupt("cancelled during complete()")
            if exc_holder[0] is not None:
                raise exc_holder[0]
            assert final_holder[0] is not None
            final_message = final_holder[0]

        stop_reason = final_message.get("stop_reason", "stop")

        if stop_reason == "error":
            error_msg = final_message.get("error_message", "Unknown error")
            raise RuntimeError(error_msg)

        step_text = extract_text(final_message)
        if step_text:
            self._fire("on_text_delta", step_text)

        tool_calls = extract_tool_calls(final_message)
        return step_text, tool_calls, stop_reason, final_message

    # ── streaming call ───────────────────────────────────────────────────

    def call_streaming(
        self,
        context: dict,
        options: LLMOptions,
    ) -> tuple[str, list[ToolCallRequest], str, dict]:
        """Streaming: consume LLMStream events and fire callbacks.

        Delta-emission policy is controlled by ``cfg.buffered_streaming``:

        * ``False`` (default) — fire ``on_text_delta`` live as tokens arrive.
          On transient error mid-stream, the user sees partial output and the
          retry replays it, producing visible duplication.  The engine fires
          ``on_stream_retry`` before the retry so UIs can mark/clear the
          duplicated region.
        * ``True`` — buffer deltas locally and only flush them to
          ``on_text_delta`` after a clean ``done`` event.  A failed attempt
          discards its buffer silently.  Eliminates duplication at the cost
          of losing live-token UX.

        The buffered partial text is stashed on the raised exception as
        ``err.partial_stream_text`` so ``call_with_retry`` can pass it
        to ``on_stream_retry``.
        """
        cfg = self.config
        text_parts: list[str] = []
        # Tracks how many characters have already been flushed to
        # on_text_delta — used for buffered mode and for partial-text
        # reporting on error.
        emitted_len = 0
        final_message: dict | None = None
        thinking_active = False
        stream = None

        def _flush_pending() -> None:
            """Emit un-flushed buffered deltas (buffered mode only)."""
            nonlocal emitted_len
            if emitted_len >= sum(len(p) for p in text_parts):
                return
            pending = "".join(text_parts)[emitted_len:]
            if pending:
                self._fire("on_text_delta", pending)
                emitted_len += len(pending)

        # Watcher-thread state.  Declared before the try so the finally
        # block can always see them, even on exceptions raised mid-setup.
        watcher_stop: threading.Event | None = None
        watcher_thread: threading.Thread | None = None

        try:
            stream = self.runtime_owner.stream(
                context, options=options, model_spec=self.model_spec,
            )
            # The between-yields cancel
            # check at the top of the loop only fires if the iterator is
            # yielding.  When the provider hangs mid-chunk (network stuck
            # waiting for bytes), the `for event in stream` call blocks
            # indefinitely without ever returning to the Python level.
            # Spawn a daemon watcher that waits on cancel_event and calls
            # ``stream.close()`` when it fires — closing the underlying
            # httpx response unblocks the SDK iterator with an exception
            # the except-clause below catches and normalises.
            #
            # Returns early (no watcher spawned) when there's no
            # cancel_event — avoids the tiny thread overhead on the common
            # non-CLI path where cancellation isn't wired up.
            if self.cancel_event is not None:
                watcher_stop = threading.Event()
                _cancel_ev = self.cancel_event  # local capture for closure

                def _stream_cancel_watcher() -> None:
                    # Wake on EITHER the cancel signal OR the stop signal
                    # (which fires from the finally clause when the call
                    # completes normally).  We poll both because
                    # threading.Event doesn't support wait_for_any.
                    while not watcher_stop.is_set():
                        if _cancel_ev.wait(0.05):
                            if watcher_stop.is_set():
                                return
                            try:
                                stream.close()
                            except Exception:  # noqa: BLE001 — best-effort
                                pass
                            return

                watcher_thread = threading.Thread(
                    target=_stream_cancel_watcher,
                    name="jyagent-llm-stream-cancel-watcher",
                    daemon=True,
                )
                watcher_thread.start()
            for event in stream:
                # Cancellation check inside the stream loop so Ctrl-C
                # doesn't wait for the provider to close — latency-sensitive.
                if self._is_cancelled():
                    raise KeyboardInterrupt("cancelled during stream")
                etype = event.get("type")

                if etype == "text_delta":
                    text = event.get("text", "")
                    if thinking_active:
                        thinking_active = False
                        self._fire("on_thinking_stop")
                    text_parts.append(text)
                    if not cfg.buffered_streaming:
                        # Live mode: emit now.
                        self._fire("on_text_delta", text)
                        emitted_len += len(text)
                    # Buffered mode: accumulate, flush on `done`.

                elif etype == "thinking_start":
                    if not thinking_active:
                        thinking_active = True
                        self._fire("on_thinking_start")

                elif etype == "thinking_delta":
                    if not thinking_active:
                        thinking_active = True
                        self._fire("on_thinking_start")

                elif etype in ("tool_call_start", "tool_call_delta"):
                    if thinking_active:
                        thinking_active = False
                        self._fire("on_thinking_stop")

                elif etype == "thinking_end":
                    if thinking_active:
                        thinking_active = False
                        self._fire("on_thinking_stop")

                elif etype == "done":
                    final_message = event["message"]
                    # Buffered mode: flush the accumulated text now that we
                    # know the stream completed cleanly.
                    if cfg.buffered_streaming:
                        _flush_pending()

                elif etype == "error":
                    final_message = event["message"]

            if final_message is None:
                final_message = stream.get_final_message()

            stop_reason = final_message.get("stop_reason", "stop")
            if stop_reason == "error":
                error_msg = final_message.get("error_message", "Unknown streaming error")
                # Attach partial text so the retry layer can report it to
                # on_stream_retry.  Accumulated even in buffered mode — the
                # caller decides what (if anything) to do with it.
                err = RuntimeError(error_msg)
                err.partial_stream_text = "".join(text_parts)  # type: ignore[attr-defined]
                raise err

            # Successful completion: in live mode emitted_len already equals
            # the full text length; in buffered mode the done-handler above
            # flushed it.  Nothing more to do.
            final_text = extract_text(final_message)
            if final_text and not text_parts:
                text_parts.append(final_text)
                self._fire("on_text_delta", final_text)
            elif final_text and final_text != "".join(text_parts):
                _logger.debug(
                    "Stream delta text differed from final message text; using final message text for loop result."
                )
            tool_calls = extract_tool_calls(final_message)
            return final_text or "".join(text_parts), tool_calls, stop_reason, final_message

        except BaseException as err:
            # Stash partial text on every exception path (transient network
            # errors, cancellations, etc.) so the retry layer can pass it to
            # on_stream_retry.  Intentionally set on the raised exception
            # rather than returned — the call site re-raises and catches it
            # at a different layer.
            if not hasattr(err, "partial_stream_text"):
                err.partial_stream_text = "".join(text_parts)  # type: ignore[attr-defined]
            raise

        finally:
            # Signal the cancel watcher to exit BEFORE closing the
            # stream ourselves, so it doesn't race with our own close().
            if watcher_stop is not None:
                watcher_stop.set()
                # Nudge the watcher: it's waiting on either cancel_event
                # or its own polling timeout.  Don't join — it's a daemon
                # and will exit within its 50 ms poll cycle; joining would
                # add needless latency to every stream call.
            if thinking_active:
                self._fire("on_thinking_stop")
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    # ── retry wrapper ────────────────────────────────────────────────────

    def call_with_retry(
        self,
        context: dict,
        options: LLMOptions,
        step: int = 0,
    ) -> tuple[str, list[ToolCallRequest], str, dict]:
        """Call the LLM (streaming or complete) with transient-error retry.

        Returns (step_text, tool_call_blocks, stop_reason, final_message).

        ``step`` is accepted for signature compatibility with the old
        ``AgentLoop._call_llm_with_retry`` but is currently unused; kept
        so callers that pass it positionally don't break.
        """
        cfg = self.config
        last_error: BaseException | None = None

        for attempt in range(cfg.retry_attempts + 1):
            try:
                if cfg.streaming:
                    return self.call_streaming(context, options)
                else:
                    return self.call_complete(context, options)
            except KeyboardInterrupt:
                raise
            except Exception as err:
                last_error = err
                if is_transient_error(err) and attempt < cfg.retry_attempts:
                    if self._is_cancelled():
                        raise
                    # Exponential backoff with "equal jitter" (AWS architecture
                    # recommendation) to avoid thundering-herd when multiple
                    # parallel sub-agents all retry a 529 at the same moment.
                    #   half the delay is deterministic exponential,
                    #   half is uniform random in [0, base * 2^attempt / 2].
                    base = cfg.retry_base_delay * (2 ** attempt)
                    delay = base / 2 + random.uniform(0, base / 2)
                    self._fire("on_retry", attempt + 1, err)
                    # Signal UI that any partial output from the failed
                    # attempt will be replayed on retry (visual de-duplication
                    # hook).  ``partial_stream_text`` is stashed by
                    # ``call_streaming`` on the exception; missing for
                    # non-streaming path.
                    partial_text = getattr(err, "partial_stream_text", "")
                    self._fire("on_stream_retry", "transient_error", partial_text)
                    # Cancel-aware backoff: wake immediately on Ctrl-C so we
                    # don't burn through a long retry window after cancel.
                    if self._cancellable_sleep(delay):
                        raise KeyboardInterrupt("cancelled during retry backoff")
                    continue
                raise

        # Should not reach here, but just in case:
        raise last_error  # type: ignore[misc]


__all__ = [
    "LLMRunner",
    "extract_text",
    "extract_tool_calls",
    "is_transient_error",
    "build_runtime_options",
]
