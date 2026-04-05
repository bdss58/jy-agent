from __future__ import annotations

from typing import Any, cast

from .types import (
    AnthropicReasoningConfig,
    AnthropicThinkingAdaptiveConfig,
    AnthropicThinkingDisabledConfig,
    OpenAIReasoningConfig,
)


_OPENAI_REASONING_KEYS = {"effort", "summary"}
_OPENAI_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_OPENAI_REASONING_SUMMARIES = {"auto", "concise", "detailed"}

_ANTHROPIC_REASONING_KEYS = {"type", "budget_tokens", "display", "effort"}
_ANTHROPIC_THINKING_TYPES = {"disabled", "adaptive"}
_ANTHROPIC_THINKING_DISPLAYS = {"summarized", "omitted"}
_ANTHROPIC_REASONING_EFFORTS = {"low", "medium", "high", "max"}
_ANTHROPIC_ADAPTIVE_MODEL_PREFIXES = ("claude-sonnet-4-6", "claude-opus-4-6")
_ANTHROPIC_EFFORT_MODEL_PREFIXES = ("claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-5")
_ANTHROPIC_MAX_EFFORT_MODEL_PREFIXES = ("claude-opus-4-6",)


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


def _normalize_model_name(model: str | None) -> str:
    return (model or "").strip().lower()


def _matches_model_prefix(model: str | None, prefixes: tuple[str, ...]) -> bool:
    normalized = _normalize_model_name(model)
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _supports_anthropic_adaptive_thinking(model: str | None) -> bool:
    return _matches_model_prefix(model, _ANTHROPIC_ADAPTIVE_MODEL_PREFIXES)


def _supports_anthropic_effort(model: str | None) -> bool:
    return _matches_model_prefix(model, _ANTHROPIC_EFFORT_MODEL_PREFIXES)


def _supports_anthropic_max_effort(model: str | None) -> bool:
    return _matches_model_prefix(model, _ANTHROPIC_MAX_EFFORT_MODEL_PREFIXES)


def _anthropic_budget_migration_error() -> ValueError:
    return ValueError(
        "Anthropic manual thinking budgets are no longer supported. "
        "Remove 'budget_tokens' / ANTHROPIC_THINKING_BUDGET_TOKENS and use adaptive thinking "
        "plus ANTHROPIC_REASONING_EFFORT instead."
    )


def _anthropic_disabled_fields_error(fields: set[str]) -> ValueError:
    field_list = ", ".join(sorted(fields))
    return ValueError(
        f"Anthropic disabled thinking config does not support: {field_list}. "
        "Use only {'type': 'disabled'}."
    )


def _anthropic_adaptive_requires_claude_46_error(model: str | None) -> ValueError:
    return ValueError(
        f"Anthropic adaptive thinking is not supported by model '{model or '<unset>'}'. "
        "Switch to 'claude-sonnet-4-6' or 'claude-opus-4-6'."
    )


def _anthropic_effort_unsupported_error(model: str | None) -> ValueError:
    return ValueError(
        f"Anthropic reasoning effort is not supported by model '{model or '<unset>'}'. "
        "Use 'claude-sonnet-4-6', 'claude-opus-4-6', or 'claude-opus-4-5'."
    )


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


def validate_anthropic_reasoning(reasoning: Any, *, model: str | None = None) -> AnthropicReasoningConfig:
    config = _ensure_mapping(reasoning, provider="Anthropic")
    _ensure_allowed_keys(config, provider="Anthropic", allowed_keys=_ANTHROPIC_REASONING_KEYS)

    if "budget_tokens" in config:
        raise _anthropic_budget_migration_error()
    if config.get("type") == "enabled":
        raise _anthropic_budget_migration_error()

    thinking_type = None
    if "type" in config:
        thinking_type = _require_literal(
            config["type"],
            field_name="type",
            allowed_values=_ANTHROPIC_THINKING_TYPES,
            provider="Anthropic",
        )

    validated_display = None
    if "display" in config:
        validated_display = _require_literal(
            config["display"],
            field_name="display",
            allowed_values=_ANTHROPIC_THINKING_DISPLAYS,
            provider="Anthropic",
        )

    validated_effort = None
    if "effort" in config:
        validated_effort = _require_literal(
            config["effort"],
            field_name="effort",
            allowed_values=_ANTHROPIC_REASONING_EFFORTS,
            provider="Anthropic",
        )

    if thinking_type == "disabled":
        invalid_fields = set(config) - {"type"}
        if invalid_fields:
            raise _anthropic_disabled_fields_error(invalid_fields)
        return cast(AnthropicThinkingDisabledConfig, {"type": "disabled"})

    resolved_type = thinking_type
    if resolved_type is None and validated_effort is not None and _supports_anthropic_adaptive_thinking(model):
        resolved_type = "adaptive"

    if validated_display is not None and resolved_type != "adaptive":
        raise ValueError("Anthropic reasoning config field 'display' requires thinking type 'adaptive'.")

    if resolved_type == "adaptive" and not _supports_anthropic_adaptive_thinking(model):
        raise _anthropic_adaptive_requires_claude_46_error(model)

    if validated_effort is not None:
        if not _supports_anthropic_effort(model):
            raise _anthropic_effort_unsupported_error(model)
        if validated_effort == "max" and not _supports_anthropic_max_effort(model):
            raise ValueError(
                f"Anthropic reasoning effort 'max' is only supported by model 'claude-opus-4-6', not '{model or '<unset>'}'."
            )

    if resolved_type == "adaptive" and validated_effort is None:
        validated_effort = "medium"

    if resolved_type is None and validated_effort is None:
        raise ValueError("Anthropic reasoning config must include at least one of: type, effort.")

    validated: AnthropicThinkingAdaptiveConfig = {}
    if resolved_type == "adaptive":
        validated["type"] = "adaptive"
        if validated_display is not None:
            validated["display"] = cast(Any, validated_display)
    if validated_effort is not None:
        validated["effort"] = cast(Any, validated_effort)
    return cast(AnthropicReasoningConfig, validated)


def build_anthropic_request_reasoning(
    reasoning: Any,
    *,
    model: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    validated = validate_anthropic_reasoning(reasoning, model=model)

    thinking = None
    if validated.get("type") == "disabled":
        thinking = {"type": "disabled"}
    elif validated.get("type") == "adaptive":
        thinking = {"type": "adaptive"}
        if "display" in validated:
            thinking["display"] = validated["display"]

    output_config = None
    if "effort" in validated:
        output_config = {"effort": validated["effort"]}

    return thinking, output_config
