import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jyagent.session_stats as session_stats
from jyagent.session_stats import SessionStats


def test_openai_cached_input_uses_cached_rate_without_double_billing():
    stats = SessionStats()

    stats.record_usage(
        {
            "input_tokens": 11,
            "output_tokens": 5,
            "cache_read_input_tokens": 2,
        },
        provider="openai",
        model="gpt-5.4",
    )

    expected = ((11 - 2) * 2.50 + 2 * 0.25 + 5 * 15.0) / 1_000_000
    assert stats.total_input_tokens == 11
    assert stats.total_cache_read_tokens == 2
    assert stats.total_cost == pytest.approx(expected)


def test_openai_long_context_multiplier_applies_when_threshold_is_exceeded():
    stats = SessionStats()

    stats.record_usage(
        {
            "input_tokens": 300_000,
            "output_tokens": 10,
        },
        provider="openai",
        model="gpt-5.4",
    )

    expected = (300_000 * 2.50 * 2.0 + 10 * 15.0 * 1.5) / 1_000_000
    assert stats.total_cost == pytest.approx(expected)


def test_parse_openai_model_page_text_extracts_text_token_prices_and_long_context():
    page_text = """
    Pricing
    Text tokens
    Per 1M tokens
    Input
    $2.50
    Cached input
    $0.25
    Output
    $15.00
    Quick comparison
    For models with a 1.05M context window (GPT-5.4 and GPT-5.4 pro), prompts with >272K input tokens are priced at 2x input and 1.5x output for the full session for standard, batch, and flex.
    """

    pricing = session_stats._parse_openai_model_page_text(page_text)

    assert pricing is not None
    assert pricing.input_per_million == 2.50
    assert pricing.cache_read_per_million == 0.25
    assert pricing.output_per_million == 15.0
    assert pricing.input_tokens_include_cache_reads is True
    assert pricing.long_context_threshold_tokens == 272_000
    assert pricing.long_context_input_multiplier == 2.0
    assert pricing.long_context_output_multiplier == 1.5


def test_lookup_pricing_fetches_openai_model_docs_for_unknown_snapshot(monkeypatch):
    session_stats._OPENAI_DOCS_PRICING_CACHE.pop("gpt-test", None)
    session_stats._OPENAI_DOCS_PRICING_CACHE.pop("gpt-test-2026-04-01", None)

    def fake_fetch(model: str) -> str | None:
        if model == "gpt-test":
            return """
            Text tokens
            Per 1M tokens
            Input
            $0.40
            Cached input
            $0.10
            Output
            $1.60
            Modalities
            """
        return None

    monkeypatch.setattr(session_stats, "_fetch_openai_model_page_text", fake_fetch)

    pricing = session_stats._lookup_pricing("openai", "gpt-test-2026-04-01")

    assert pricing is not None
    assert pricing.input_per_million == 0.40
    assert pricing.cache_read_per_million == 0.10
    assert pricing.output_per_million == 1.60
