"""AI provider factory."""

from __future__ import annotations

from studio.providers.base import FreeOnlyPolicyError, LLMProvider, LLMResponse, ProviderError, TokenUsage
from studio.providers.gemini import GeminiProvider
from studio.providers.openai_compat import LlamaCppProvider, OllamaProvider, OpenRouterProvider


def get_provider(name: str = "gemini", *, free_only: bool = True, model: str | None = None) -> LLMProvider:
    key = name.lower().strip()
    if key == "gemini":
        return GeminiProvider(models=(model,) if model else None, free_only=free_only)
    if key == "openrouter":
        return OpenRouterProvider(default_model=model or "google/gemma-3-27b-it:free", free_only=free_only)
    if key == "ollama":
        return OllamaProvider(default_model=model or "qwen2.5:7b")
    if key in ("llamacpp", "llama.cpp"):
        return LlamaCppProvider(default_model=model or "local-model")
    raise ValueError(f"Unknown AI provider {name!r}. Options: gemini, openrouter, ollama, llamacpp")


__all__ = [
    "FreeOnlyPolicyError",
    "GeminiProvider",
    "LLMProvider",
    "LLMResponse",
    "LlamaCppProvider",
    "OllamaProvider",
    "OpenRouterProvider",
    "ProviderError",
    "TokenUsage",
    "get_provider",
]
