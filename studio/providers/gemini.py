"""Google Gemini provider (default free-tier LLM backend)."""

from __future__ import annotations

import os
import time
from typing import Type, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from studio.ai.env import gemini_api_key
from studio.providers.base import FreeOnlyPolicyError, LLMProvider, LLMResponse, ProviderError

T = TypeVar("T", bound=BaseModel)

DEFAULT_GEMINI_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
)
TRANSIENT_CODES = {429, 500, 502, 503, 504}


class GeminiProvider(LLMProvider):
    """Gemini API provider using google-genai with lite-first fallback and backoff."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        models: tuple[str, ...] | None = None,
        free_only: bool = True,
        timeout_s: float = 60.0,
        retries_per_model: int = 2,
    ) -> None:
        super().__init__(free_only=free_only, timeout_s=timeout_s)
        key = api_key or gemini_api_key()
        self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        env_model = os.environ.get("GEMINI_MODEL")
        self.models = (env_model,) if env_model else (models or DEFAULT_GEMINI_MODELS)
        self.retries_per_model = retries_per_model

    def _call_with_fallback(self, contents: str, config: types.GenerateContentConfig, model: str | None) -> tuple[object, str, float]:
        chain = (model,) if model else self.models
        last_err: Exception | None = None
        for m in chain:
            for attempt in range(1, self.retries_per_model + 1):
                t0 = time.perf_counter()
                try:
                    resp = self._client.models.generate_content(model=m, contents=contents, config=config)
                    return resp, m, time.perf_counter() - t0
                except genai_errors.APIError as exc:
                    last_err = exc
                    if exc.code in TRANSIENT_CODES and attempt < self.retries_per_model:
                        time.sleep(4.0 * attempt)
                        continue
                    break
                except Exception as exc:
                    last_err = exc
                    if attempt < self.retries_per_model:
                        time.sleep(4.0 * attempt)
                        continue
                    break
        raise ProviderError(f"Gemini failed across {chain}: {last_err}")

    def generate_text(self, prompt: str, *, system: str = "", model: str | None = None) -> LLMResponse:
        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=0.6,
        )
        resp, used_model, dt = self._call_with_fallback(prompt, cfg, model)
        text = getattr(resp, "text", "") or ""
        self.usage.add()
        return LLMResponse(text=text, model=used_model, provider=self.name, latency_s=dt)

    def generate_structured(
        self, prompt: str, schema: Type[T], *, system: str = "", model: str | None = None
    ) -> tuple[T, LLMResponse]:
        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.6,
        )
        resp, used_model, dt = self._call_with_fallback(prompt, cfg, model)
        raw = getattr(resp, "text", "") or ""
        parsed = schema.model_validate_json(raw)
        self.usage.add()
        return parsed, LLMResponse(text=raw, model=used_model, provider=self.name, latency_s=dt)
