from __future__ import annotations

from typing import Any, Protocol

from ..config import build_model_spec, get_reasoning_config_for_provider
from .types import AssistantMessage, Context, ModelSpec, LLMOptions, LLMStream


class ProviderAdapter(Protocol):
    provider: str
    api_name: str

    def stream(self, model_spec: ModelSpec, context: Context, options: LLMOptions | None = None) -> LLMStream:
        ...

    def complete(self, model_spec: ModelSpec, context: Context, options: LLMOptions | None = None) -> AssistantMessage:
        ...


_ADAPTERS: dict[str, ProviderAdapter] = {}

# Sentinel for "caller did not pass a value" — lets callers explicitly pass
# `reasoning=None` to disable reasoning while still keeping the default behavior
# (auto-derive from env via get_reasoning_config_for_provider) for others.
_UNSET: Any = object()


def register_adapter(adapter: ProviderAdapter) -> None:
    _ADAPTERS[adapter.provider] = adapter


def get_adapter(provider: str) -> ProviderAdapter:
    adapter = _ADAPTERS.get(provider)
    if adapter is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Available: {sorted(_ADAPTERS)}")
    return adapter


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS)


class LLMOwner:
    def __init__(self, model_spec: ModelSpec):
        resolved = build_model_spec(model_spec.provider, model_spec.model, source="model spec provider")
        get_adapter(resolved.provider)
        self._model_spec = resolved
        self._session_id = ""

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    def label(self) -> str:
        return self._model_spec.label()

    def switch_model(self, provider: str, model: str) -> ModelSpec:
        resolved = build_model_spec(provider, model, source="/model provider")
        get_adapter(resolved.provider)
        self._model_spec = resolved
        return self._model_spec

    def set_session_id(self, session_id: str | None) -> None:
        self._session_id = session_id or ""

    def _options_with_session(self, options: LLMOptions | None) -> LLMOptions:
        options = options or LLMOptions()
        if not self._session_id:
            return options
        metadata = dict(options.metadata or {})
        metadata.setdefault("session_id", self._session_id)
        if metadata == (options.metadata or {}):
            return options
        return LLMOptions(
            max_output_tokens=options.max_output_tokens,
            timeout=options.timeout,
            reasoning=options.reasoning,
            metadata=metadata,
            tool_choice=options.tool_choice,
        )

    def stream(self, context: Context, options: LLMOptions | None = None, model_spec: ModelSpec | None = None) -> LLMStream:
        resolved = model_spec or self._model_spec
        options = self._options_with_session(options)
        return get_adapter(resolved.provider).stream(resolved, context, options)

    def complete(self, context: Context, options: LLMOptions | None = None, model_spec: ModelSpec | None = None) -> AssistantMessage:
        resolved = model_spec or self._model_spec
        options = self._options_with_session(options)
        return get_adapter(resolved.provider).complete(resolved, context, options)

    def complete_text(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        max_output_tokens: int | None = None,
        model_spec: ModelSpec | None = None,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
        reasoning: Any = _UNSET,
    ) -> str:
        # Default: auto-derive reasoning config from env (may fail for models
        # that don't support adaptive thinking). Callers can pass reasoning=None
        # to explicitly disable reasoning — useful for cheap utility calls
        # (e.g. skill router) where extended thinking is wasteful and may be
        # rejected by validation for non-4.6+ Anthropic models.
        if reasoning is _UNSET:
            reasoning = get_reasoning_config_for_provider(
                (model_spec or self._model_spec).provider,
                max_output_tokens=max_output_tokens,
                model=(model_spec or self._model_spec).model,
            )
        message = self.complete(
            {
                "system_prompt": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
            },
            options=LLMOptions(
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                reasoning=reasoning,
                metadata={
                    "component": "llm_owner",
                    "mode": "complete_text",
                    **(metadata or {}),
                },
            ),
            model_spec=model_spec,
        )
        parts = []
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
