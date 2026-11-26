"""OpenAI-compatible providers: OpenRouter (free-only), Ollama, and llama.cpp."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Type, TypeVar

from pydantic import BaseModel

from studio.providers.base import FreeOnlyPolicyError, LLMProvider, LLMResponse, ProviderError

T = TypeVar("T", bound=BaseModel)


class OpenAICompatProvider(LLMProvider):
    """Zero-dependency HTTP client for OpenAI-compatible /v1/chat/completions endpoints."""

    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        default_model: str,
        api_key: str = "",
        free_only: bool = True,
        timeout_s: float = 60.0,
    ) -> None:
        super().__init__(free_only=free_only, timeout_s=timeout_s)
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.api_key = api_key

    def _validate_model(self, model: str) -> None:
        pass

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            raise ProviderError(f"{self.name} HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise ProviderError(f"{self.name} request failed: {exc}") from exc

    def generate_text(self, prompt: str, *, system: str = "", model: str | None = None, temperature: float = 0.6) -> LLMResponse:
        m = model or self.default_model
        self._validate_model(m)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        t0 = time.perf_counter()
        data = self._post({"model": m, "messages": messages, "temperature": temperature})
        dt = time.perf_counter() - t0
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        self.usage.add(pt, ct)
        return LLMResponse(text=text, model=m, provider=self.name, latency_s=dt, prompt_tokens=pt, completion_tokens=ct)

    def generate_structured(
        self, prompt: str, schema: Type[T], *, system: str = "", model: str | None = None, temperature: float = 0.6
    ) -> tuple[T, LLMResponse]:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        sys_prompt = f"{system}\n\nReturn ONLY valid JSON matching this schema:\n{schema_json}".strip()
        resp = self.generate_text(prompt, system=sys_prompt, model=model, temperature=temperature)
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = schema.model_validate_json(raw)
        return parsed, resp


class OpenRouterProvider(OpenAICompatProvider):
    """OpenRouter provider with strict :free model enforcement when free_only=True."""

    name = "openrouter"

    def __init__(self, api_key: str | None = None, default_model: str = "google/gemma-3-27b-it:free", free_only: bool = True, timeout_s: float = 60.0) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ProviderError("OPENROUTER_API_KEY is not set.")
        super().__init__("https://openrouter.ai/api/v1", default_model, api_key=key, free_only=free_only, timeout_s=timeout_s)

    def _validate_model(self, model: str) -> None:
        if self.free_only and not model.endswith(":free"):
            raise FreeOnlyPolicyError(
                f"OpenRouter model {model!r} is not marked ':free'. Refusing billable model under free_only=True."
            )


class OllamaProvider(OpenAICompatProvider):
    """Local Ollama server (http://localhost:11434/v1)."""

    name = "ollama"

    def __init__(self, default_model: str = "qwen2.5:7b", base_url: str | None = None, timeout_s: float = 120.0) -> None:
        url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        super().__init__(url, default_model, free_only=True, timeout_s=timeout_s)


class LlamaCppProvider(OpenAICompatProvider):
    """Local llama.cpp server (http://localhost:8080/v1)."""

    name = "llamacpp"

    def __init__(self, default_model: str = "local-model", base_url: str | None = None, timeout_s: float = 120.0) -> None:
        url = base_url or os.environ.get("LLAMACPP_BASE_URL", "http://localhost:8080/v1")
        super().__init__(url, default_model, free_only=True, timeout_s=timeout_s)
