# Session statistics — token/cost tracking, timing.
#
# Thread-safe singleton that accumulates usage across the streaming loop.
# The CLI reads from this to display the status bar.

import threading
import time
from dataclasses import dataclass


# ─── Pricing (USD per million tokens) ──────────────────────────────────────
# Override via set_model_pricing() when needed.


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    cache_creation_per_million: float | None = None
    cache_read_per_million: float | None = None
    input_tokens_include_cache_reads: bool = False
    long_context_threshold_tokens: int | None = None
    long_context_input_multiplier: float = 1.0
    long_context_output_multiplier: float = 1.0


def _pricing(
    input_per_million: float,
    output_per_million: float,
    *,
    cache_creation_per_million: float | None = None,
    cache_read_per_million: float | None = None,
    input_tokens_include_cache_reads: bool = False,
    long_context_threshold_tokens: int | None = None,
    long_context_input_multiplier: float = 1.0,
    long_context_output_multiplier: float = 1.0,
) -> ModelPricing:
    return ModelPricing(
        input_per_million=input_per_million,
        output_per_million=output_per_million,
        cache_creation_per_million=cache_creation_per_million,
        cache_read_per_million=cache_read_per_million,
        input_tokens_include_cache_reads=input_tokens_include_cache_reads,
        long_context_threshold_tokens=long_context_threshold_tokens,
        long_context_input_multiplier=long_context_input_multiplier,
        long_context_output_multiplier=long_context_output_multiplier,
    )


_MODEL_PRICING = {
    "anthropic": {
        "claude-opus-4-6": _pricing(5.0, 25.0, cache_creation_per_million=6.25, cache_read_per_million=0.50),
        "claude-opus-4-7": _pricing(5.0, 25.0, cache_creation_per_million=6.25, cache_read_per_million=0.50),
    },
    "openai": {
        "gpt-5.4": _pricing(2.50, 15.0, cache_read_per_million=0.25),
    },
}

# Guards concurrent reads (``_lookup_pricing``) and writes
# (``set_model_pricing``).  On CPython 3.14 the GIL makes the canonical
# race (``RuntimeError: dictionary changed size during iteration``)
# extremely hard to reproduce — single dict ops are bytecode-atomic and
# ``sorted()`` over an items() view holds the GIL for the duration of
# its C-level iterator setup.  We still take the lock because:
#   * Free-threaded CPython (PEP 703) removes the GIL — the same code
#     becomes a real data race there.
#   * Future MCP-style code may register many providers concurrently
#     from background threads, widening the window.
#   * The lock is uncontended on the hot path (one read per LLM call)
#     so the cost is a couple of CAS instructions.
# Codex review 2026-04-25 Part 1 #6.
_PRICING_LOCK = threading.RLock()


def _coerce_pricing(pricing: ModelPricing | tuple[float, float] | tuple[float, float, float | None, float | None]) -> ModelPricing:
    if isinstance(pricing, ModelPricing):
        return pricing
    if len(pricing) == 2:
        return ModelPricing(pricing[0], pricing[1])
    if len(pricing) == 4:
        return ModelPricing(
            input_per_million=pricing[0],
            output_per_million=pricing[1],
            cache_creation_per_million=pricing[2],
            cache_read_per_million=pricing[3],
        )
    raise TypeError("pricing must be a ModelPricing or a 2-/4-item tuple")


def _lookup_pricing(provider: str, model: str) -> ModelPricing | None:
    """Find pricing by longest prefix match.

    Thread-safe: takes ``_PRICING_LOCK`` so a concurrent
    ``set_model_pricing`` cannot mutate the inner dict mid-iteration.
    """
    with _PRICING_LOCK:
        pricing_map = _MODEL_PRICING.get(provider, {})
        # Snapshot items inside the lock; the sort + linear scan happen on
        # the local list so we hold the lock for O(N) not O(N log N).
        items = list(pricing_map.items())
    for prefix, pricing in sorted(items, key=lambda x: -len(x[0])):
        if model.startswith(prefix):
            return _coerce_pricing(pricing)
    return None


def set_model_pricing(
    provider: str,
    model_prefix: str,
    pricing: ModelPricing | tuple[float, float] | tuple[float, float, float | None, float | None],
) -> None:
    with _PRICING_LOCK:
        _MODEL_PRICING.setdefault(provider, {})[model_prefix] = _coerce_pricing(pricing)


# ─── Pure pricing function ────────────────────────────────────────────────
#
# Single source of pricing math for the whole codebase.  Before 2026-04
# there were two implementations: ``SessionStats._record_cost`` (full
# semantics — long-context multipliers, cache-read credit, unknown-cost
# flags) and the engine's ``_CostTracker`` (simplified, missing both
# the long-context multiplier and the ``input_tokens_include_cache_reads``
# credit).  The two drifted on Anthropic 1M-context models: the session
# status bar reported one cost while ``LoopConfig.max_cost_usd`` enforced
# a different (lower) figure.  Codex review 2026-04-25 Part 1 #9/#10.
#
# Keep this function pure: no locks, no mutation, no I/O.  Both
# ``SessionStats._record_cost`` and the engine's ``_CostTracker`` call
# it; any future consumer (tracing, analytics, per-subagent budgeting)
# should route through here too.


@dataclass(frozen=True)
class CostBreakdown:
    """Result of ``compute_call_cost``.

    ``is_priced`` is True iff the (provider, model) pair had full
    pricing coverage for the token components present in ``usage``
    (e.g. if the call used cache_creation but the pricing entry has
    ``cache_creation_per_million=None``, the call is *unpriced* even
    if other components were priceable).
    """

    cost_usd: float
    is_priced: bool


def compute_call_cost(
    usage: dict | None,
    provider: str,
    model: str,
) -> CostBreakdown:
    """Price a single LLM call's token usage.

    Honours long-context multipliers (Anthropic 1M-context tier) and
    the ``input_tokens_include_cache_reads`` convention where the
    provider reports cache_read tokens as part of input_tokens and the
    caller must subtract them to avoid double billing.

    ``usage`` is the normalised dict form produced by the runtime (see
    ``llm.types.Usage``): keys ``input_tokens``, ``output_tokens``,
    ``cache_creation_input_tokens``, ``cache_read_input_tokens``.  A
    ``None`` or empty usage yields a zero-cost priced breakdown — i.e.
    the model made no billable call (e.g. client-side abort).

    When pricing is missing for any required component the breakdown
    reports ``is_priced=False`` and ``cost_usd=0.0``.  Callers decide
    whether to surface this (a warning, a budget lower bound, or a
    silent skip).
    """
    input_t = (usage or {}).get("input_tokens", 0) or 0
    output_t = (usage or {}).get("output_tokens", 0) or 0
    cache_create = (usage or {}).get("cache_creation_input_tokens", 0) or 0
    cache_read = (usage or {}).get("cache_read_input_tokens", 0) or 0

    # No token activity at all — trivially priced, zero cost.
    if not (input_t or output_t or cache_create or cache_read):
        return CostBreakdown(cost_usd=0.0, is_priced=True)

    pricing = _lookup_pricing(provider, model) if provider and model else None
    if pricing is None:
        return CostBreakdown(cost_usd=0.0, is_priced=False)

    # Component-pricing gaps make the whole call unpriced — better to
    # surface a lower bound than silently under-report.
    if cache_create and pricing.cache_creation_per_million is None:
        return CostBreakdown(cost_usd=0.0, is_priced=False)
    if cache_read and pricing.cache_read_per_million is None:
        return CostBreakdown(cost_usd=0.0, is_priced=False)

    prompt_tokens = input_t + cache_create + cache_read
    input_multiplier = 1.0
    output_multiplier = 1.0
    if (
        pricing.long_context_threshold_tokens is not None
        and prompt_tokens > pricing.long_context_threshold_tokens
    ):
        input_multiplier = pricing.long_context_input_multiplier
        output_multiplier = pricing.long_context_output_multiplier

    billable_input = input_t
    if pricing.input_tokens_include_cache_reads:
        billable_input = max(0, input_t - cache_read)

    cost = (
        billable_input * pricing.input_per_million * input_multiplier / 1_000_000
        + output_t * pricing.output_per_million * output_multiplier / 1_000_000
        + cache_create * (pricing.cache_creation_per_million or 0.0) * input_multiplier / 1_000_000
        + cache_read * (pricing.cache_read_per_million or 0.0) * input_multiplier / 1_000_000
    )
    return CostBreakdown(cost_usd=cost, is_priced=True)


class SessionStats:
    """Accumulates token usage and cost for the current session."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cache_creation_tokens = 0
            self.total_cache_read_tokens = 0
            self.turn_input_tokens = 0
            self.turn_output_tokens = 0
            self.api_calls = 0
            self.tool_calls = 0
            self.turns = 0
            self.session_start = time.time()
            self._provider = ""
            self._model = ""
            self._known_total_cost = 0.0
            self._known_turn_cost = 0.0
            self._has_unknown_total_cost = False
            self._has_unknown_turn_cost = False
            self.subagent_runs: list[dict] = []

    def new_turn(self):
        """Reset per-turn counters."""
        with self._lock:
            self.turn_input_tokens = 0
            self.turn_output_tokens = 0
            self._known_turn_cost = 0.0
            self._has_unknown_turn_cost = False
            self.turns += 1

    def set_active_model(self, provider: str, model: str) -> None:
        with self._lock:
            self._provider = provider or self._provider
            self._model = model or self._model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def _usage_value(self, usage, key: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return usage.get(key, 0) or 0
        return getattr(usage, key, 0) or 0

    def _record_cost(
        self,
        input_t: int,
        output_t: int,
        cache_create: int,
        cache_read: int,
        provider: str,
        model: str,
    ) -> None:
        """Add a single LLM call's cost to the running totals.

        Delegates the pricing math to ``compute_call_cost`` (single
        source of truth — see module docstring on the unification with
        the engine's ``_CostTracker``).  This method only handles the
        accounting side: bumping known/unknown totals and the per-turn
        slice.
        """
        usage = {
            "input_tokens": input_t,
            "output_tokens": output_t,
            "cache_creation_input_tokens": cache_create,
            "cache_read_input_tokens": cache_read,
        }
        breakdown = compute_call_cost(usage, provider, model)
        if not breakdown.is_priced:
            # Token activity present but unpriced — flag for UI ("≈" or
            # similar marker) without poisoning the numeric total.
            if input_t or output_t or cache_create or cache_read:
                self._has_unknown_total_cost = True
                self._has_unknown_turn_cost = True
            return
        self._known_total_cost += breakdown.cost_usd
        self._known_turn_cost += breakdown.cost_usd

    def record_usage(self, usage, provider: str = "", model: str = ""):
        """Record token usage from a normalized runtime response."""
        with self._lock:
            input_t = self._usage_value(usage, 'input_tokens')
            output_t = self._usage_value(usage, 'output_tokens')
            cache_create = self._usage_value(usage, 'cache_creation_input_tokens')
            cache_read = self._usage_value(usage, 'cache_read_input_tokens')

            if provider:
                self._provider = provider
            if model:
                self._model = model
            self.total_input_tokens += input_t
            self.total_output_tokens += output_t
            self.total_cache_creation_tokens += cache_create
            self.total_cache_read_tokens += cache_read
            self.turn_input_tokens += input_t
            self.turn_output_tokens += output_t
            self.api_calls += 1
            self._record_cost(
                input_t, output_t, cache_create, cache_read,
                self._provider, self._model,
            )

    def record_tool_call(self):
        with self._lock:
            self.tool_calls += 1

    def record_subagent_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str = "",
        model: str = "",
        *,
        task_preview: str = "",
        elapsed: float = 0.0,
        status: str = "",
        steps: int = 0,
        tool_calls: int = 0,
    ):
        """Record token usage from a sub-agent (counts toward session totals).

        The provider/model are used ONLY for cost calculation — the parent's
        active model label (self._provider / self._model) is never overwritten.
        """
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.turn_input_tokens += input_tokens
            self.turn_output_tokens += output_tokens
            self.api_calls += 1  # count sub-agent as at least 1 API call

            # Use the subagent's own provider/model for cost, not the parent's
            cost_provider = provider or self._provider
            cost_model = model or self._model
            self._record_cost(input_tokens, output_tokens, 0, 0, cost_provider, cost_model)

            self.subagent_runs.append({
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "task_preview": task_preview,
                "elapsed": elapsed,
                "status": status,
                "steps": steps,
                "tool_calls": tool_calls,
            })

    @property
    def total_cost(self) -> float | None:
        """Estimated total cost in USD."""
        with self._lock:
            if self._has_unknown_total_cost:
                return None
            return self._known_total_cost

    @property
    def turn_cost(self) -> float | None:
        """Estimated cost for the current turn."""
        with self._lock:
            if self._has_unknown_turn_cost:
                return None
            return self._known_turn_cost

    @property
    def subagent_summary(self) -> str:
        """One-line summary of sub-agent dispatches for the session."""
        with self._lock:
            runs = list(self.subagent_runs)
        if not runs:
            return ""
        total = len(runs)
        completed = sum(1 for r in runs if r.get("status") == "completed")
        total_in = sum(r.get("input_tokens", 0) for r in runs)
        total_out = sum(r.get("output_tokens", 0) for r in runs)
        return f"🤖 {completed}/{total} subagents, {self.format_tokens(total_in)}+{self.format_tokens(total_out)} tokens"

    @property
    def elapsed(self) -> float:
        return time.time() - self.session_start

    def format_tokens(self, n: int) -> str:
        """Human-friendly token count: 1234 → 1.2k, 1234567 → 1.2M."""
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n/1_000:.1f}k"
        return str(n)

    def format_cost(self, cost: float | None) -> str:
        """Format cost in dollars."""
        if cost is None:
            return "n/a"
        if cost < 0.01:
            return f"${cost:.4f}"
        if cost < 1.0:
            return f"${cost:.3f}"
        return f"${cost:.2f}"

    def summary_line(self) -> str:
        """One-line summary for status bar / bottom toolbar."""
        with self._lock:
            model_short = f"{self._provider}:{self._model}" if self._provider and self._model else "?"
            in_t = self.format_tokens(self.total_input_tokens)
            out_t = self.format_tokens(self.total_output_tokens)
            tool_calls = self.tool_calls
            elapsed_min = self.elapsed / 60
        # Compute cost outside the lock (total_cost also acquires _lock)
        cost = self.format_cost(self.total_cost)

        parts = [
            f"⚡ {model_short}",
            f"↑{in_t} ↓{out_t}",
            f"💰{cost}",
        ]
        if tool_calls > 0:
            parts.append(f"🔧{tool_calls}")
        if elapsed_min >= 1:
            parts.append(f"⏱{elapsed_min:.0f}m")

        return " │ ".join(parts)

    def turn_summary(self) -> str:
        """Summary for the just-completed turn (printed after response)."""
        with self._lock:
            in_t = self.format_tokens(self.turn_input_tokens)
            out_t = self.format_tokens(self.turn_output_tokens)
        cost = self.format_cost(self.turn_cost)
        return f"↑{in_t} ↓{out_t} ({cost})"


# ─── Singleton ────────────────────────────────────────────────────────────────

_stats = SessionStats()


def get_stats() -> SessionStats:
    return _stats


__all__ = [
    "ModelPricing",
    "CostBreakdown",
    "SessionStats",
    "compute_call_cost",
    "get_stats",
    "set_model_pricing",
]
