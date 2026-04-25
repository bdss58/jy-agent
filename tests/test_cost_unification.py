# tests/test_cost_unification.py — Pin the single-source-of-pricing
# invariant between SessionStats and the engine's _CostTracker.
#
# Codex review 2026-04-25 Part 1 #9/#10 flagged that the engine's
# _CostTracker reimplemented pricing with a simplified formula —
# missing the long-context multiplier and the
# `input_tokens_include_cache_reads` credit that SessionStats applied.
# The two trackers diverged on Anthropic 1M-context calls: status bar
# said one number, max_cost_usd budget enforced another.
#
# After unification on `compute_call_cost`, both code paths must
# produce bit-identical numbers for any usage/pricing combination.

from __future__ import annotations

import pytest

from jyagent.runtime import stats as st
from jyagent.runtime.loop import engine as le


@pytest.fixture
def restore_pricing():
    """Snapshot+restore the global pricing table around each test so
    tests don't bleed model registrations into each other."""
    snapshot = {
        provider: dict(models)
        for provider, models in st._MODEL_PRICING.items()
    }
    yield
    st._MODEL_PRICING.clear()
    st._MODEL_PRICING.update(snapshot)


class TestPricingUnification:
    """Both trackers must agree for every (usage, pricing) shape."""

    def _both_trackers(self, usage, provider, model):
        """Return (session_stats_cost, cost_tracker_cost) for the same call."""
        # Fresh SessionStats so we don't pick up state from the singleton.
        ss = st.SessionStats()
        ss.set_active_model(provider, model)
        ss.record_usage(usage, provider=provider, model=model)
        session_cost = ss.total_cost

        ct = le._CostTracker()
        ct.record(usage, provider, model)
        tracker_cost = ct.cost
        return session_cost, tracker_cost

    def test_basic_input_output_only(self, restore_pricing):
        st.set_model_pricing("test-vendor", "basic-model", (10.0, 20.0))
        usage = {"input_tokens": 1_000, "output_tokens": 500}
        a, b = self._both_trackers(usage, "test-vendor", "basic-model")
        assert a == pytest.approx(b)
        # 1000 * 10 / 1M + 500 * 20 / 1M = 0.01 + 0.01 = 0.02
        assert a == pytest.approx(0.02)

    def test_cache_create_and_read(self, restore_pricing):
        st.set_model_pricing(
            "test-vendor",
            "cache-model",
            st.ModelPricing(
                input_per_million=10.0,
                output_per_million=20.0,
                cache_creation_per_million=12.5,
                cache_read_per_million=1.0,
            ),
        )
        usage = {
            "input_tokens": 1_000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 2_000,
            "cache_read_input_tokens": 4_000,
        }
        a, b = self._both_trackers(usage, "test-vendor", "cache-model")
        assert a == pytest.approx(b), (
            "SessionStats and _CostTracker disagreed on cache-priced call"
        )

    def test_long_context_multiplier_is_honoured_by_both(self, restore_pricing):
        """The pre-unification _CostTracker IGNORED long_context_*_multiplier.
        For an Anthropic-1M-style pricing entry the two trackers differed
        by the multiplier value (e.g. 2x) on every above-threshold call.
        """
        st.set_model_pricing(
            "test-vendor",
            "1m-model",
            st.ModelPricing(
                input_per_million=3.0,
                output_per_million=15.0,
                cache_read_per_million=0.30,
                long_context_threshold_tokens=200_000,
                long_context_input_multiplier=2.0,
                long_context_output_multiplier=2.5,
            ),
        )
        # Total prompt = 250_000 > threshold → multipliers active.
        usage = {
            "input_tokens": 250_000,
            "output_tokens": 5_000,
        }
        a, b = self._both_trackers(usage, "test-vendor", "1m-model")
        assert a == pytest.approx(b), (
            f"Long-context multiplier divergence: SessionStats={a}, "
            f"_CostTracker={b}.  This is exactly the Codex 2026-04-25 "
            f"Part 1 #10 bug."
        )
        # Below-threshold call must NOT apply the multiplier.
        below = {"input_tokens": 100_000, "output_tokens": 1_000}
        a2, b2 = self._both_trackers(below, "test-vendor", "1m-model")
        assert a2 == pytest.approx(b2)
        assert a2 < a  # sanity: above-threshold cost is higher

    def test_input_tokens_include_cache_reads_credit(self, restore_pricing):
        """When the provider reports cache_read tokens as part of input_tokens,
        the caller must subtract them to avoid double billing.  The pre-
        unification _CostTracker did NOT apply this credit."""
        st.set_model_pricing(
            "test-vendor",
            "credit-model",
            st.ModelPricing(
                input_per_million=10.0,
                output_per_million=20.0,
                cache_read_per_million=1.0,
                input_tokens_include_cache_reads=True,
            ),
        )
        usage = {
            "input_tokens": 10_000,        # of which 6000 were cache reads
            "output_tokens": 500,
            "cache_read_input_tokens": 6_000,
        }
        a, b = self._both_trackers(usage, "test-vendor", "credit-model")
        assert a == pytest.approx(b), (
            f"input_tokens_include_cache_reads divergence: "
            f"SessionStats={a}, _CostTracker={b}"
        )

    def test_unpriced_model_marks_both_as_unpriced(self, restore_pricing):
        """A model with no pricing entry: SessionStats marks unknown,
        _CostTracker bumps unpriced_calls.  Both must report it."""
        usage = {"input_tokens": 1_000, "output_tokens": 500}

        ss = st.SessionStats()
        ss.set_active_model("nope", "nope-model")
        ss.record_usage(usage, provider="nope", model="nope-model")
        assert ss._has_unknown_total_cost is True
        # SessionStats.total_cost returns None when there's any unknown
        # component (so the UI can render "≈"); the priced subtotal is
        # in _known_total_cost.  Both are 0 when nothing was priced.
        assert ss.total_cost is None
        assert ss._known_total_cost == 0.0

        ct = le._CostTracker()
        ct.record(usage, "nope", "nope-model")
        assert ct.has_unpriced_usage is True
        assert ct.cost == 0.0

    def test_empty_usage_is_priced_zero(self, restore_pricing):
        """No tokens → priced, zero cost.  Edge case used by client-side
        aborts that call record_usage with an all-zero usage dict."""
        b = st.compute_call_cost({}, "anything", "anything")
        assert b.is_priced is True
        assert b.cost_usd == 0.0
