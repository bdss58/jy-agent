# Session statistics — token/cost tracking, timing.
#
# Thread-safe singleton that accumulates usage across the streaming loop.
# The CLI reads from this to display the status bar.

import time
import threading


# ─── Pricing (USD per million tokens) ──────────────────────────────────────
# Updated for Claude Sonnet 4 / Haiku / Opus.  Override via set_model_pricing().

_MODEL_PRICING = {
    # model-prefix → (input_$/M, output_$/M)
    "claude-sonnet-4":       (3.0, 15.0),
    "claude-3-5-sonnet":     (3.0, 15.0),
    "claude-3-7-sonnet":     (3.0, 15.0),
    "claude-3-5-haiku":      (0.80, 4.0),
    "claude-3-haiku":        (0.25, 1.25),
    "claude-3-opus":         (15.0, 75.0),
    "claude-opus-4":         (15.0, 75.0),
}

_DEFAULT_PRICING = (3.0, 15.0)  # Sonnet pricing as fallback


def _lookup_pricing(model: str) -> tuple:
    """Find pricing by longest prefix match."""
    for prefix, pricing in sorted(_MODEL_PRICING.items(), key=lambda x: -len(x[0])):
        if model.startswith(prefix):
            return pricing
    return _DEFAULT_PRICING


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
            self._model = ""

    def new_turn(self):
        """Reset per-turn counters."""
        with self._lock:
            self.turn_input_tokens = 0
            self.turn_output_tokens = 0
            self.turns += 1

    def record_usage(self, usage, model: str = ""):
        """Record token usage from an Anthropic API response.
        
        `usage` is the message.usage object with input_tokens, output_tokens,
        and optionally cache_creation_input_tokens, cache_read_input_tokens.
        """
        with self._lock:
            input_t = getattr(usage, 'input_tokens', 0) or 0
            output_t = getattr(usage, 'output_tokens', 0) or 0
            cache_create = getattr(usage, 'cache_creation_input_tokens', 0) or 0
            cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0

            self.total_input_tokens += input_t
            self.total_output_tokens += output_t
            self.total_cache_creation_tokens += cache_create
            self.total_cache_read_tokens += cache_read
            self.turn_input_tokens += input_t
            self.turn_output_tokens += output_t
            self.api_calls += 1
            if model:
                self._model = model

    def record_tool_call(self):
        with self._lock:
            self.tool_calls += 1

    @property
    def total_cost(self) -> float:
        """Estimated total cost in USD."""
        pricing = _lookup_pricing(self._model)
        with self._lock:
            input_cost = self.total_input_tokens * pricing[0] / 1_000_000
            output_cost = self.total_output_tokens * pricing[1] / 1_000_000
            # Cache creation costs 25% more than input, cache read costs 90% less
            cache_create_cost = self.total_cache_creation_tokens * (pricing[0] * 1.25) / 1_000_000
            cache_read_cost = self.total_cache_read_tokens * (pricing[0] * 0.1) / 1_000_000
        return input_cost + output_cost + cache_create_cost + cache_read_cost

    @property
    def turn_cost(self) -> float:
        """Estimated cost for the current turn."""
        pricing = _lookup_pricing(self._model)
        with self._lock:
            return (self.turn_input_tokens * pricing[0] + self.turn_output_tokens * pricing[1]) / 1_000_000

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

    def format_cost(self, cost: float) -> str:
        """Format cost in dollars."""
        if cost < 0.01:
            return f"${cost:.4f}"
        if cost < 1.0:
            return f"${cost:.3f}"
        return f"${cost:.2f}"

    def summary_line(self) -> str:
        """One-line summary for status bar / bottom toolbar."""
        with self._lock:
            model_short = self._model.split("-202")[0] if "-202" in self._model else (self._model or "?")
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
