from __future__ import annotations

from typing import Any, cast

from .types import (
    AnthropicThinkingAdaptiveConfig,
    AnthropicThinkingDisabledConfig,
    AnthropicThinkingEnabledConfig,
    OpenAIReasoningConfig,
)


_OPENAI_REASONING_KEYS = {"effort", "summary"}
_OPENAI_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_OPENAI_REASONING_SUMMARIES = {"auto", "concise", "detailed"}

_ANTHROPIC_THINKING_KEYS = {"type", "budget_tokens", "display"}
_ANTHROPIC_THINKING_TYPES = {"disabled", "adaptive", "enabled"}
_ANTHROPIC_THINKING_DISPLAYS = {"summarized", "omitted"}
_ANTHROPIC_MIN_BUDGET_TOKENS = 1024


def _ensure_mapping(reasoning: Any, *, provider: str) -> dict[str, Any]:
    if not isinstance(reasoning, dict):
        raise ValueError(f"{provider} reasoning config must be a dict, got {type(reasoning).__name__}.")
    return reasoning


def _ensure_allowed_keys(reasoning: dict[str, Any], *, provider: str, allowed_keys: set[str]) -> None:
    unknown_keys = sorted(key for key in reasoning if key not in allowed_keys)
    if unknown_keys:
        allowed = ", ".join(sorted(allowed_keys))
        raise ValueError(f"{provider} reasoning config has unsupported keys: {unknown_keys}. Allowed keys: {allowed}.")


def _require_literal(value: Any, *, field_name: str, allowed_values: set[str], provider: str) -> str:
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"{provider} reasoning config field '{field_name}' must be one of: {allowed}.")
    return cast(str, value)


def validate_openai_reasoning(reasoning: Any) -> OpenAIReasoningConfig:
    config = _ensure_mapping(reasoning, provider="OpenAI")
    if "generate_summary" in config:
        raise ValueError("OpenAI reasoning config field 'generate_summary' is deprecated; use 'summary' instead.")
    _ensure_allowed_keys(config, provider="OpenAI", allowed_keys=_OPENAI_REASONING_KEYS)

    validated: OpenAIReasoningConfig = {}

    if "effort" in config:
        validated["effort"] = cast(
            Any,
            _require_literal(
                config["effort"],
                field_name="effort",
                allowed_values=_OPENAI_REASONING_EFFORTS,
                provider="OpenAI",
            ),
        )
    if "summary" in config:
        validated["summary"] = cast(
            Any,
            _require_literal(
                config["summary"],
                field_name="summary",
                allowed_values=_OPENAI_REASONING_SUMMARIES,
                provider="OpenAI",
            ),
        )
    return validated


def validate_anthropic_thinking(
    reasoning: Any,
    *,
    max_output_tokens: int | None,
) -> AnthropicThinkingDisabledConfig | AnthropicThinkingAdaptiveConfig | AnthropicThinkingEnabledConfig:
    config = _ensure_mapping(reasoning, provider="Anthropic")
    _ensure_allowed_keys(config, provider="Anthropic", allowed_keys=_ANTHROPIC_THINKING_KEYS)

    if "type" not in config:
        raise ValueError("Anthropic reasoning config requires a 'type' field.")
    thinking_type = _require_literal(
        config["type"],
        field_name="type",
        allowed_values=_ANTHROPIC_THINKING_TYPES,
        provider="Anthropic",
    )

    if thinking_type == "disabled":
        if set(config) != {"type"}:
            raise ValueError("Anthropic disabled thinking config only supports the 'type' field.")
        return {"type": "disabled"}

    validated_display = None
    if "display" in config:
        validated_display = _require_literal(
            config["display"],
            field_name="display",
            allowed_values=_ANTHROPIC_THINKING_DISPLAYS,
            provider="Anthropic",
        )

    if thinking_type == "adaptive":
        if "budget_tokens" in config:
            raise ValueError("Anthropic adaptive thinking config does not support 'budget_tokens'.")
        validated: AnthropicThinkingAdaptiveConfig = {"type": "adaptive"}
        if validated_display is not None:
            validated["display"] = cast(Any, validated_display)
        return validated

    if "budget_tokens" not in config:
        raise ValueError("Anthropic enabled thinking config requires 'budget_tokens'.")
    budget_tokens = config["budget_tokens"]
    if not isinstance(budget_tokens, int) or isinstance(budget_tokens, bool):
        raise ValueError("Anthropic enabled thinking config field 'budget_tokens' must be an integer.")
    if budget_tokens < _ANTHROPIC_MIN_BUDGET_TOKENS:
        raise ValueError(
            f"Anthropic enabled thinking config field 'budget_tokens' must be >= {_ANTHROPIC_MIN_BUDGET_TOKENS}."
        )
    if max_output_tokens is None:
        raise ValueError(
            "Anthropic enabled thinking config requires RuntimeOptions.max_output_tokens to validate 'budget_tokens'."
        )
    if budget_tokens >= max_output_tokens:
        raise ValueError("Anthropic enabled thinking config requires 'budget_tokens' to be less than max_output_tokens.")

    validated = {
        "type": "enabled",
        "budget_tokens": budget_tokens,
    }
    if validated_display is not None:
        validated["display"] = cast(Any, validated_display)
    return cast(AnthropicThinkingEnabledConfig, validated)
