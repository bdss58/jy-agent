"""OpenAI provider adapter stub.

This module registers an OpenAI adapter when the ``openai`` SDK is available.
The implementation is a placeholder — all methods raise ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Any

try:
    import openai as _openai_sdk  # noqa: F401 — presence check only
except ImportError:
    # openai SDK not installed — skip registration silently
    raise  # re-raise so _auto_register_providers catches it

from ..core import register_adapter
from ..types import AssistantMessage, Context, ModelSpec, RuntimeOptions, RuntimeStream

# Register "openai" as a known provider in the config layer too.
from ...config import register_provider as _register_config_provider


class OpenAIAdapter:
    provider = "openai"
    api_name = "openai-responses"

    def stream(
        self,
        model_spec: ModelSpec,
        context: Context,
        options: RuntimeOptions | None = None,
    ) -> RuntimeStream:
        raise NotImplementedError("OpenAI provider not yet implemented")

    def complete(
        self,
        model_spec: ModelSpec,
        context: Context,
        options: RuntimeOptions | None = None,
    ) -> AssistantMessage:
        raise NotImplementedError("OpenAI provider not yet implemented")


register_adapter(OpenAIAdapter())
_register_config_provider("openai")
