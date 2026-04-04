# Session statistics — token/cost tracking, timing.
#
# Thread-safe singleton that accumulates usage across the streaming loop.
# The CLI reads from this to display the status bar.

import threading
import time
import re
from dataclasses import dataclass


# ─── Pricing (USD per million tokens) ──────────────────────────────────────
# Override via set_model_pricing(). Common OpenAI models are pinned locally,
# and unknown OpenAI models fall back to the official model docs page.


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
        "claude-sonnet-4": _pricing(3.0, 15.0, cache_creation_per_million=3.75, cache_read_per_million=0.30),
        "claude-3-5-sonnet": _pricing(3.0, 15.0, cache_creation_per_million=3.75, cache_read_per_million=0.30),
        "claude-3-7-sonnet": _pricing(3.0, 15.0, cache_creation_per_million=3.75, cache_read_per_million=0.30),
        "claude-3-5-haiku": _pricing(0.80, 4.0, cache_creation_per_million=1.0, cache_read_per_million=0.08),
        "claude-3-haiku": _pricing(0.25, 1.25, cache_creation_per_million=0.3125, cache_read_per_million=0.025),
        "claude-3-opus": _pricing(15.0, 75.0, cache_creation_per_million=18.75, cache_read_per_million=1.5),
        "claude-opus-4": _pricing(15.0, 75.0, cache_creation_per_million=18.75, cache_read_per_million=1.5),
    },
    "openai": {
        "gpt-5.4-pro": _pricing(
            30.0,
            180.0,
            input_tokens_include_cache_reads=True,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
        ),
        "gpt-5.4-mini": _pricing(0.75, 4.50, cache_read_per_million=0.075, input_tokens_include_cache_reads=True),
        "gpt-5.4-nano": _pricing(0.20, 1.25, cache_read_per_million=0.02, input_tokens_include_cache_reads=True),
        "gpt-5.4": _pricing(
            2.50,
            15.0,
            cache_read_per_million=0.25,
            input_tokens_include_cache_reads=True,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
        ),
        "gpt-5-pro": _pricing(15.0, 120.0, input_tokens_include_cache_reads=True),
        "gpt-5-mini": _pricing(0.25, 2.0, cache_read_per_million=0.025, input_tokens_include_cache_reads=True),
        "gpt-5-nano": _pricing(0.05, 0.40, cache_read_per_million=0.005, input_tokens_include_cache_reads=True),
        "gpt-5.3-chat-latest": _pricing(1.75, 14.0, cache_read_per_million=0.175, input_tokens_include_cache_reads=True),
        "gpt-5.3-codex": _pricing(1.75, 14.0, cache_read_per_million=0.175, input_tokens_include_cache_reads=True),
        "gpt-5": _pricing(1.25, 10.0, cache_read_per_million=0.125, input_tokens_include_cache_reads=True),
    },
}

_OPENAI_MODEL_PAGE_BASE_URL = "https://developers.openai.com/api/docs/models"
_OPENAI_DOCS_FETCH_LOCK = threading.Lock()
_OPENAI_DOCS_PRICING_CACHE: dict[str, ModelPricing | None] = {}


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


def _openai_model_candidates(model: str) -> list[str]:
    candidates: list[str] = []

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(model)
    add(re.sub(r"-20\d{2}-\d{2}-\d{2}$", "", model))
    return candidates


def _parse_price_token(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "-":
        return None
    if value.startswith("$"):
        value = value[1:]
    return float(value.replace(",", ""))


def _parse_token_count(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMB]?)", value.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported token count: {value}")

    number = float(match.group(1))
    suffix = match.group(2).upper()
    multiplier = {
        "": 1,
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
    }[suffix]
    return int(number * multiplier)


def _parse_openai_model_page_text(text: str) -> ModelPricing | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    price_labels = {"Input", "Cached input", "Output"}
    stop_markers = {"Quick comparison", "Modalities", "Rate limits", "Pricing"}
    pricing_values: dict[str, str] = {}

    for index, line in enumerate(lines):
        if line != "Text tokens":
            continue

        current_label: str | None = None
        for next_line in lines[index + 1:index + 24]:
            if next_line in stop_markers:
                break
            if next_line in price_labels:
                current_label = next_line
                continue
            if current_label and (next_line.startswith("$") or next_line == "-"):
                pricing_values[current_label] = next_line
                current_label = None

        if "Input" in pricing_values and "Output" in pricing_values:
            break
        pricing_values = {}

    if "Input" not in pricing_values or "Output" not in pricing_values:
        return None

    long_context_match = re.search(
        r"prompts with >([0-9]+(?:\.[0-9]+)?[KMB]?) input tokens are priced at "
        r"([0-9]+(?:\.[0-9]+)?)x input and ([0-9]+(?:\.[0-9]+)?)x output",
        text,
        re.IGNORECASE,
    )

    long_context_threshold_tokens = None
    long_context_input_multiplier = 1.0
    long_context_output_multiplier = 1.0
    if long_context_match:
        long_context_threshold_tokens = _parse_token_count(long_context_match.group(1))
        long_context_input_multiplier = float(long_context_match.group(2))
        long_context_output_multiplier = float(long_context_match.group(3))

    return ModelPricing(
        input_per_million=_parse_price_token(pricing_values["Input"]) or 0.0,
        output_per_million=_parse_price_token(pricing_values["Output"]) or 0.0,
        cache_creation_per_million=None,
        cache_read_per_million=_parse_price_token(pricing_values.get("Cached input")),
        input_tokens_include_cache_reads=True,
        long_context_threshold_tokens=long_context_threshold_tokens,
        long_context_input_multiplier=long_context_input_multiplier,
        long_context_output_multiplier=long_context_output_multiplier,
    )


def _fetch_openai_model_page_text(model: str) -> str | None:
    try:
        import httpx
        from bs4 import BeautifulSoup
    except Exception:
        return None

    try:
        response = httpx.get(f"{_OPENAI_MODEL_PAGE_BASE_URL}/{model}", follow_redirects=True, timeout=5.0)
        response.raise_for_status()
    except Exception:
        return None

    return BeautifulSoup(response.text, "html.parser").get_text("\n")


def _lookup_openai_pricing_from_docs(model: str) -> ModelPricing | None:
    for candidate in _openai_model_candidates(model):
        if candidate in _OPENAI_DOCS_PRICING_CACHE:
            cached = _OPENAI_DOCS_PRICING_CACHE[candidate]
            if cached is not None:
                set_model_pricing("openai", candidate, cached)
                return cached

    with _OPENAI_DOCS_FETCH_LOCK:
        for candidate in _openai_model_candidates(model):
            if candidate in _OPENAI_DOCS_PRICING_CACHE:
                cached = _OPENAI_DOCS_PRICING_CACHE[candidate]
                if cached is not None:
                    set_model_pricing("openai", candidate, cached)
                    return cached
                continue

            page_text = _fetch_openai_model_page_text(candidate)
            pricing = _parse_openai_model_page_text(page_text) if page_text else None
            _OPENAI_DOCS_PRICING_CACHE[candidate] = pricing
            if pricing is not None:
                set_model_pricing("openai", candidate, pricing)
                return pricing

    return None


def _lookup_pricing(provider: str, model: str) -> ModelPricing | None:
    """Find pricing by longest prefix match."""
    pricing_map = _MODEL_PRICING.get(provider, {})
    for prefix, pricing in sorted(pricing_map.items(), key=lambda x: -len(x[0])):
        if model.startswith(prefix):
            return _coerce_pricing(pricing)
    if provider == "openai":
        return _lookup_openai_pricing_from_docs(model)
    return None


def set_model_pricing(
    provider: str,
    model_prefix: str,
    pricing: ModelPricing | tuple[float, float] | tuple[float, float, float | None, float | None],
) -> None:
    _MODEL_PRICING.setdefault(provider, {})[model_prefix] = _coerce_pricing(pricing)


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
        pricing = _lookup_pricing(provider, model) if provider and model else None
        if pricing is None:
            if input_t or output_t or cache_create or cache_read:
                self._has_unknown_total_cost = True
                self._has_unknown_turn_cost = True
            return

        if cache_create and pricing.cache_creation_per_million is None:
            self._has_unknown_total_cost = True
            self._has_unknown_turn_cost = True
            return

        if cache_read and pricing.cache_read_per_million is None:
            self._has_unknown_total_cost = True
            self._has_unknown_turn_cost = True
            return

        prompt_tokens = input_t + cache_create + cache_read
        input_multiplier = 1.0
        output_multiplier = 1.0
        if pricing.long_context_threshold_tokens is not None and prompt_tokens > pricing.long_context_threshold_tokens:
            input_multiplier = pricing.long_context_input_multiplier
            output_multiplier = pricing.long_context_output_multiplier

        billable_input_tokens = input_t
        if pricing.input_tokens_include_cache_reads:
            billable_input_tokens = max(0, input_t - cache_read)

        input_cost = billable_input_tokens * pricing.input_per_million * input_multiplier / 1_000_000
        output_cost = output_t * pricing.output_per_million * output_multiplier / 1_000_000
        cache_create_cost = (
            cache_create * (pricing.cache_creation_per_million or 0.0) * input_multiplier / 1_000_000
        )
        cache_read_cost = cache_read * (pricing.cache_read_per_million or 0.0) * input_multiplier / 1_000_000
        total = input_cost + output_cost + cache_create_cost + cache_read_cost
        self._known_total_cost += total
        self._known_turn_cost += total

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

    def record_subagent_usage(self, input_tokens: int, output_tokens: int, provider: str = "", model: str = ""):
        """Record token usage from a sub-agent (counts toward session totals)."""
        with self._lock:
            if provider:
                self._provider = provider
            if model:
                self._model = model
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.turn_input_tokens += input_tokens
            self.turn_output_tokens += output_tokens
            self.api_calls += 1  # count sub-agent as at least 1 API call
            self._record_cost(input_tokens, output_tokens, 0, 0, self._provider, self._model)

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
