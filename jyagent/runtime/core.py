from __future__ import annotations

from typing import Protocol

from ..config import get_reasoning_config_for_provider
from .types import AssistantMessage, Context, ModelSpec, RuntimeOptions, RuntimeStream


class RuntimeAdapter(Protocol):
    provider: str
    api_name: str

    def stream(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> RuntimeStream:
        ...

    def complete(self, model_spec: ModelSpec, context: Context, options: RuntimeOptions | None = None) -> AssistantMessage:
        ...


_ADAPTERS: dict[str, RuntimeAdapter] = {}


def register_adapter(adapter: RuntimeAdapter) -> None:
    _ADAPTERS[adapter.provider] = adapter


def get_adapter(provider: str) -> RuntimeAdapter:
    adapter = _ADAPTERS.get(provider)
    if adapter is None:
        raise ValueError(f"Unknown runtime provider '{provider}'. Available: {sorted(_ADAPTERS)}")
    return adapter


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS)


class RuntimeOwner:
    def __init__(self, model_spec: ModelSpec):
        self._model_spec = model_spec

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    def label(self) -> str:
        return self._model_spec.label()

    def switch_model(self, provider: str, model: str) -> ModelSpec:
        self._model_spec = ModelSpec(provider=provider, model=model)
        return self._model_spec

    def stream(self, context: Context, options: RuntimeOptions | None = None, model_spec: ModelSpec | None = None) -> RuntimeStream:
        resolved = model_spec or self._model_spec
        return get_adapter(resolved.provider).stream(resolved, context, options)

    def complete(self, context: Context, options: RuntimeOptions | None = None, model_spec: ModelSpec | None = None) -> AssistantMessage:
        resolved = model_spec or self._model_spec
        return get_adapter(resolved.provider).complete(resolved, context, options)

    def complete_text(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        max_output_tokens: int | None = None,
        model_spec: ModelSpec | None = None,
        timeout: float | None = None,
    ) -> str:
        message = self.complete(
            {
                "system_prompt": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
            },
            options=RuntimeOptions(
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                reasoning=get_reasoning_config_for_provider(
                    (model_spec or self._model_spec).provider,
                    max_output_tokens=max_output_tokens,
                ),
            ),
            model_spec=model_spec,
        )
        parts = []
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
