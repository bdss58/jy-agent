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
and ``AgentLoop`` inherit them and expose ``_cancel_event`` / ``_callbacks``
instance attributes for the helper to read.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import random
import threading
from typing import TYPE_CHECKING, Any, Callable

from .callbacks import LoopCallbacks
from .config import LoopConfig
from .llm_client import LLMClient
from .llm_types import LLMOptions, ModelSpec, ToolCallRequest
from ._thread_helpers import LoopThreadHelper
from ...config import get_reasoning_config_for_provider, STREAM_TIMEOUT


_logger = logging.getLogger(__name__)


# Type alias for the (text, tool_calls, stop_reason, message) tuple every
# LLM call returns.  Used by the retry helper below and the public-ish
# call sites on ``AgentLoop`` and ``LLMRunner``.
_LLMResult = "tuple[str, list[ToolCallRequest], str, dict]"


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


# ─── Shared retry helper ────────────────────────────────────────────────────


def _retry_llm_call(
    *,
    config: LoopConfig,
    context: dict,
    options: LLMOptions,
    call_streaming: Callable[[dict, LLMOptions], Any],
    call_complete: Callable[[dict, LLMOptions], Any],
    fire: Callable[..., None],
    is_cancelled: Callable[[], bool],
    cancellable_sleep: Callable[[float], bool],
    is_transient: Callable[[BaseException], bool] | None = None,
):
    """Shared retry loop with cancel-aware backoff + on_stream_retry signaling.

    The retry-loop body was previously duplicated byte-for-byte between
    ``AgentLoop._call_llm_with_retry`` and ``LLMRunner.call_with_retry``;
    only the dispatch callables and helper hooks differed:

    +----------+-------------------------+------------------------+--------------------+
    | Caller   | streaming arg           | complete arg           | helper hooks       |
    +==========+=========================+========================+====================+
    | AgentLoop| self._call_streaming    | self._call_complete    | self._fire / etc.  |
    +----------+-------------------------+------------------------+--------------------+
    | LLMRunner| self.call_streaming     | self.call_complete     | self._fire / etc.  |
    +----------+-------------------------+------------------------+--------------------+

    AgentLoop's variant routes through ``self._call_streaming`` /
    ``self._call_complete`` (NOT ``LLMRunner.call_*`` directly) so subclass
    overrides + per-instance monkeypatches for transient-failure injection
    stay in effect.  See ``tests/test_loop_edge_cases.py::test_retry_loop_
    invokes_subclass_overrides`` for the exact contract.

    ``is_transient`` lets each shim pass its own module's
    ``is_transient_error`` reference so existing tests that monkeypatch
    ``engine.is_transient_error`` (or ``llm_runner.is_transient_error``)
    continue to work — the patched alias is resolved at call time, not
    captured here.  Defaults to ``llm_runner.is_transient_error``.

    Returns whatever the underlying call returns — by convention an
    ``(text, tool_calls, stop_reason, final_message)`` tuple.

    Backoff is "equal jitter" (AWS architecture recommendation): half the
    delay is deterministic exponential, half is uniform random in
    ``[0, base * 2^attempt / 2]``.  This avoids a thundering-herd retry
    when many parallel sub-agents all hit the same 529 simultaneously.

    Cancellation: ``is_cancelled()`` is checked before each retry sleep,
    and ``cancellable_sleep(delay)`` returns ``True`` if cancel fired
    during the sleep — both paths raise ``KeyboardInterrupt`` so the
    outer engine path runs the cancelled-exit handler.
    """
    if is_transient is None:
        is_transient = is_transient_error
    last_error: BaseException | None = None
    for attempt in range(config.retry_attempts + 1):
        try:
            if config.streaming:
                return call_streaming(context, options)
            return call_complete(context, options)
        except KeyboardInterrupt:
            raise
        except Exception as err:
            last_error = err
            transient = is_transient(err)
            should_retry = (
                (transient or config.retry_on_all_errors)
                and attempt < config.retry_attempts
            )
            if not should_retry:
                raise
            if is_cancelled():
                raise
            base = config.retry_base_delay * (2 ** attempt)
            delay = base / 2 + random.uniform(0, base / 2)
            fire("on_retry", attempt + 1, err)
            # ``partial_stream_text`` is stashed on the exception by
            # ``LLMRunner.call_streaming`` so the UI can replace any
            # partially-emitted text before the retry attempt re-issues
            # it.  Missing on the non-streaming path; default to "".
            partial_text = getattr(err, "partial_stream_text", "")
            reason = "transient_error" if transient else "error"
            fire("on_stream_retry", reason, partial_text)
            # Cancel-aware backoff: wake immediately on Ctrl-C so we don't
            # burn through a long retry window after cancel.
            if cancellable_sleep(delay):
                raise KeyboardInterrupt("cancelled during retry backoff")
            continue

    # Unreachable — the loop either returns or raises every iteration.
    raise last_error  # type: ignore[misc]


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
      * _callbacks — LoopCallbacks for on_text_delta, on_thinking_*,
        on_retry, on_stream_retry, on_usage
      * _cancel_event — optional threading.Event for cooperative cancel
      * model_spec — optional override (sub-agent tier)

    Inherits ``_is_cancelled`` / ``_cancellable_sleep`` / ``_fire`` from
    ``LoopThreadHelper`` (canonical ``_callbacks`` / ``_cancel_event``
    instance-attribute names; same convention AgentLoop uses).
    """

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
        self._callbacks = callbacks
        self._cancel_event = cancel_event
        self.model_spec = model_spec

    # ── provider-output validation gate ──────────────────────────────────────

    def _should_validate_provider_output(self) -> bool:
        """Return True iff provider output should be runtime-validated.

        Two equally-weighted triggers (boolean OR):
          * ``LoopConfig.validate_provider_output = True`` — caller-set per
            loop instance (e.g. test harness, dev sessions).
          * ``JYAGENT_VALIDATE_PROVIDER_OUTPUT`` env var set to a truthy
            value — global override that doesn't require touching every
            test fixture.  Truthy: ``1``, ``true``, ``yes``, ``on``
            (case-insensitive).

        Resolved on every call so a test can flip the env var inside one
        case without rebuilding the runner.  Microsecond-cheap; no caching.
        """
        if self.config.validate_provider_output:
            return True
        env_val = os.environ.get("JYAGENT_VALIDATE_PROVIDER_OUTPUT", "").strip().lower()
        return env_val in {"1", "true", "yes", "on"}

    def _validate_assistant_message_or_warn(self, msg: dict, *, source: str) -> None:
        """Validate ``msg`` as an ``AssistantMessage``; raise on failure.

        ``source`` is a short human-readable label that disambiguates
        which boundary failed in the error path: ``"complete"`` for
        ``LLMClient.complete`` returns, ``"stream:done"`` /
        ``"stream:error"`` for terminal stream events.

        Failures raise ``MessageValidationError`` (TypeError subclass) so
        the loop's existing retry/finalize machinery gets a structured
        signal — the validator path never silently logs-and-continues
        because that defeats the purpose (catching adapter drift before
        it corrupts loop state).
        """
        # Lazy import: keeps the runtime engine import-graph free of the
        # llm.validation module unless the gate is on.  Validators are
        # pure-Python and depend only on jyagent.llm.types, so the import
        # is cheap when it does happen.
        from ...llm.validation import (
            MessageValidationError,
            validate_assistant_message,
        )
        try:
            validate_assistant_message(msg, path=f"provider({source}).message")
        except MessageValidationError:
            # Re-raise unchanged — the path already names the boundary.
            # Caller's existing exception handlers (retry layer in
            # call_with_retry) will surface this as
            # "non-transient_error" and end the run with a precise error.
            raise

    # ── cancellation + callback helpers from LoopThreadHelper ───────────────
    # ``_is_cancelled``, ``_cancellable_sleep``, ``_fire`` live on the
    # ``LoopThreadHelper`` mixin (see ``_thread_helpers.py``).  The mixin
    # reads ``_cancel_event`` / ``_callbacks`` directly off ``self``.

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

        if self._cancel_event is None:
            # Fast path: no cancel machinery needed.
            final_message = self.runtime_owner.complete(
                context, options=options, model_spec=self.model_spec,
            )
        else:
            # Cancellable path: daemon worker + 100 ms poll.
            final_holder: list[dict | None] = [None]
            exc_holder: list[BaseException | None] = [None]
            done = threading.Event()

            # ContextVar propagation: ``threading.Thread`` does not
            # auto-inherit the parent's context, so any session- /
            # tracing-scoped CV consumed by ``runtime_owner.complete``
            # (or its provider SDK) would silently disappear on the
            # daemon side.  Snapshot the calling context and route the
            # worker through ``ctx.run`` so the SDK call sees the same
            # CV state as if it had run inline on the calling thread.
            ctx_complete = contextvars.copy_context()

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
                target=ctx_complete.run,
                args=(_worker,),
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

        # Provider boundary: validate the assistant message shape before
        # the loop trusts ``stop_reason`` / ``usage`` / ``content`` fields.
        # Gated by ``LoopConfig.validate_provider_output`` (or
        # ``JYAGENT_VALIDATE_PROVIDER_OUTPUT`` env var) — off by default.
        if self._should_validate_provider_output():
            self._validate_assistant_message_or_warn(final_message, source="complete")

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
        # Track which terminal event produced ``final_message`` so the
        # provider-output validator can name the exact stream boundary
        # ("stream:done" vs "stream:error" vs "stream:fallback") in any
        # error message.  The previous heuristic of inferring from
        # ``final_message["stop_reason"] == "error"`` failed when the
        # adapter handed us a malformed error event (no stop_reason at
        # all) — exactly the case we want to catch.
        final_message_source: str = "stream:fallback"
        thinking_active = False
        # Per-block thinking buffer.  Accumulates the text of the CURRENT
        # thinking block (between thinking_start/_delta and the event that
        # terminates it — clean end, tool interrupt, or stream-level error).
        # Fed by `on_thinking_delta` fires; flushed by `on_thinking_block_end`
        # with a reason string so the UI can decide its fold-marker footer.
        thinking_buf: list[str] = []
        stream = None

        def _end_thinking_block(reason: str) -> None:
            """Flush the current thinking block to on_thinking_block_end.

            ``reason`` is one of:
              * ``"end"``           — clean ``thinking_end`` event.
              * ``"tool_interrupt"`` — ended because a tool_call started.
              * ``"retry"``         — stream attempt failed; retry pending.
              * ``"error"``         — terminal error (no retry).
            Safe to call when no block is active (no-op).
            """
            if not thinking_buf:
                return
            block_text = "".join(thinking_buf)
            thinking_buf.clear()
            self._fire("on_thinking_block_end", block_text, reason)

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
            if self._cancel_event is not None:
                watcher_stop = threading.Event()
                _cancel_ev = self._cancel_event  # local capture for closure

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

                # The watcher only does ``Event.wait`` and
                # ``stream.close``; it never calls user-level code that
                # might consult a ContextVar.  Still snapshot the
                # parent's context for consistency with the other spawn
                # sites — the cost is one C-level dict copy and it
                # future-proofs against the watcher growing logic that
                # does observe CVs (e.g. tracing spans wrapping the
                # close call).
                _ctx_watcher = contextvars.copy_context()
                watcher_thread = threading.Thread(
                    target=_ctx_watcher.run,
                    args=(_stream_cancel_watcher,),
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
                        # Transitioning from reasoning → answer text means
                        # the thinking block is done cleanly (the model
                        # decided to start writing its reply).
                        _end_thinking_block("end")
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
                    chunk = event.get("text", "")
                    if chunk:
                        thinking_buf.append(chunk)
                        self._fire("on_thinking_delta", chunk)

                elif etype in ("tool_call_start", "tool_call_delta"):
                    if thinking_active:
                        thinking_active = False
                        self._fire("on_thinking_stop")
                        # A tool call cut the reasoning short — the UI's
                        # fold marker should signal the interrupt rather
                        # than treat it as a clean end.
                        _end_thinking_block("tool_interrupt")

                elif etype == "thinking_end":
                    if thinking_active:
                        thinking_active = False
                        self._fire("on_thinking_stop")
                    # Always flush on thinking_end (covers the case where
                    # thinking_active was already False because some other
                    # transition cleared it first).
                    _end_thinking_block("end")

                elif etype == "done":
                    # Defensive: a malformed adapter event missing 'message'
                    # used to crash with a raw KeyError here, surfacing as
                    # a generic loop error rather than a structured boundary
                    # validation failure.  Pull with .get() and let the
                    # downstream validation path (or the get_final_message
                    # fallback) handle the absence.
                    final_message = event.get("message")
                    if final_message is not None:
                        final_message_source = "stream:done"
                    # NOTE: do NOT flush buffered text here.  The terminal
                    # ``done`` event can still carry ``stop_reason="error"``
                    # (provider-side failure with a partially-emitted reply),
                    # in which case the partial text belongs on the retry's
                    # ``on_stream_retry`` callback — not on the user-visible
                    # ``on_text_delta`` stream.  The flush is deferred until
                    # AFTER the ``stop_reason == "error"`` gate below, so
                    # buffered mode only ever emits text from a confirmed
                    # clean completion.

                elif etype == "error":
                    final_message = event.get("message")
                    if final_message is not None:
                        final_message_source = "stream:error"

            if final_message is None:
                final_message = stream.get_final_message()
                # ``final_message_source`` stays at its initial "stream:fallback"
                # value so the validator labels this path correctly.

            # Provider boundary: validate the terminal stream message shape
            # before the loop trusts ``stop_reason`` / ``usage`` / ``content``
            # fields.  Gated by ``LoopConfig.validate_provider_output`` (or
            # ``JYAGENT_VALIDATE_PROVIDER_OUTPUT`` env var) — off by default.
            if self._should_validate_provider_output():
                self._validate_assistant_message_or_warn(
                    final_message, source=final_message_source,
                )

            stop_reason = final_message.get("stop_reason", "stop")
            if stop_reason == "error":
                error_msg = final_message.get("error_message", "Unknown streaming error")
                # Attach partial text so the retry layer can report it to
                # on_stream_retry.  Accumulated even in buffered mode — the
                # caller decides what (if anything) to do with it.
                err = RuntimeError(error_msg)
                err.partial_stream_text = "".join(text_parts)  # type: ignore[attr-defined]
                raise err

            # Buffered mode: NOW that we've confirmed a clean stream
            # (no error stop_reason, no exception in flight), flush the
            # accumulated text to the live ``on_text_delta`` stream.  The
            # flush is deliberately gated on success so the UI never sees
            # text from a failed attempt that the retry layer is about to
            # re-issue.
            if cfg.buffered_streaming:
                _flush_pending()

            # Successful completion: in live mode emitted_len already equals
            # the full text length; in buffered mode the deferred flush
            # above emitted it.  Nothing more to do.
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
            # Flush any in-progress thinking block so the UI's
            # on_thinking_block_end fires with reason="error".  The retry
            # layer will fire on_stream_retry on top of this; the UI is
            # expected to treat "error" + subsequent "retry" reset as a
            # rollback signal.
            _end_thinking_block("error")
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
            # Defensive: if we somehow exit with a non-empty thinking
            # buffer (e.g. a code path we missed), surface it as "end" so
            # the UI doesn't silently lose its preview state.  No-op if
            # already drained by the normal paths above.
            if thinking_buf:
                _end_thinking_block("end")
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

        Thin shim over the shared ``_retry_llm_call`` helper, which owns
        the actual retry-loop body.  The shim binds the dispatch
        callables to ``self.call_streaming`` / ``self.call_complete`` —
        ``LLMRunner`` callers never override those, so the binding is
        equivalent to direct method calls.  ``AgentLoop`` has its own
        shim (``_call_llm_with_retry``) that binds to its overridable
        ``_call_streaming`` / ``_call_complete`` thin delegates instead.

        Returns ``(step_text, tool_call_blocks, stop_reason, final_message)``.

        ``step`` is accepted for signature compatibility with the
        ``AgentLoop._call_llm_with_retry`` form but is currently unused;
        kept so callers that pass it positionally don't break.
        """
        return _retry_llm_call(
            config=self.config,
            context=context,
            options=options,
            call_streaming=self.call_streaming,
            call_complete=self.call_complete,
            fire=self._fire,
            is_cancelled=self._is_cancelled,
            cancellable_sleep=self._cancellable_sleep,
        )


__all__ = [
    "LLMRunner",
    "extract_text",
    "extract_tool_calls",
    "is_transient_error",
    "build_runtime_options",
]
