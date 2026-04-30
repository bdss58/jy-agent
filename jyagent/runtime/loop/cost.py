"""Per-run cost tracking for budget enforcement.

Extracted from ``engine.py`` to keep the loop controller focused on
orchestration.  ``_CostTracker`` is self-contained: it only depends on
``..stats.compute_call_cost`` for pricing math.
"""

from __future__ import annotations


class CostTracker:
    """Track estimated cost within a single run() for budget enforcement.

    Delegates pricing math to ``stats.compute_call_cost`` so the engine
    and ``SessionStats`` cannot drift on Anthropic 1M-context tier
    multipliers, the ``input_tokens_include_cache_reads`` credit, or
    cache-creation pricing.  The previous implementation reimplemented a
    simplified pricing formula and quietly under-counted cost on
    long-context calls.

    When a call's (provider, model) has no pricing entry the call's
    tokens are NOT included in the running total and ``unpriced_calls``
    is bumped.  The budget check still runs on the partial total — i.e.
    the accounted cost is a lower bound.  An earlier design returned
    ``None`` from ``known_cost`` in that case, which silently disabled
    the budget entirely; the current design reports a lower-bound cost
    and exposes ``has_unpriced_usage`` so the caller can warn once.
    """

    def __init__(self) -> None:
        self.total_cost: float = 0.0
        self.unpriced_calls: int = 0

    def record(self, usage: dict, provider: str, model: str) -> None:
        # Local import to keep the module import-cheap and avoid any
        # engine↔stats cycle hazards as more modules move to runtime/loop/.
        from ..stats import compute_call_cost

        breakdown = compute_call_cost(usage, provider, model)
        if not breakdown.is_priced:
            # Only count it as unpriced if there was actual token activity.
            # ``compute_call_cost`` already reports ``is_priced=True`` for
            # zero-token calls, so reaching this branch with no tokens is
            # impossible — but be explicit.
            if any(usage.get(k, 0) for k in ("input_tokens", "output_tokens")):
                self.unpriced_calls += 1
            return
        self.total_cost += breakdown.cost_usd

    @property
    def has_unpriced_usage(self) -> bool:
        return self.unpriced_calls > 0

    @property
    def cost(self) -> float:
        """Best-effort running total in USD.  When ``has_unpriced_usage``
        is True, this is a lower bound — unpriced calls are not included.
        """
        return self.total_cost


# Back-compat alias: existing callers import `_CostTracker` from engine.
# Engine re-exports the new name as `_CostTracker` to preserve that.
__all__ = ["CostTracker"]
