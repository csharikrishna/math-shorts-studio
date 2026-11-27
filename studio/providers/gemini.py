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
                    if getattr(exc, "code", None) == 400 and ("too many states" in str(exc).lower() or "schema" in str(exc).lower()):
                        raise ProviderError(f"Gemini schema complexity error on {m}: {exc}") from exc
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

    def generate_text(self, prompt: str, *, system: str = "", model: str | None = None, temperature: float = 0.6) -> LLMResponse:
        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
        )
        resp, used_model, dt = self._call_with_fallback(prompt, cfg, model)
        text = getattr(resp, "text", "") or ""
        um = getattr(resp, "usage_metadata", None)
        pt = int(getattr(um, "prompt_token_count", 0) or 0)
        ct = int(getattr(um, "candidates_token_count", 0) or 0)
        self.usage.add(pt, ct)
        return LLMResponse(text=text, model=used_model, provider=self.name, latency_s=dt, prompt_tokens=pt, completion_tokens=ct)

    def generate_structured(
        self, prompt: str, schema: Type[T], *, system: str = "", model: str | None = None, temperature: float = 0.6
    ) -> tuple[T, LLMResponse]:
        import json as _json

        raw_schema = schema.model_json_schema()

        def _strip_unsupported(node: object) -> object:
            if isinstance(node, dict):
                node.pop("additionalProperties", None)
                for val in node.values():
                    _strip_unsupported(val)
            elif isinstance(node, list):
                for item in node:
                    _strip_unsupported(item)
            return node

        has_ap = "additionalProperties" in _json.dumps(raw_schema)
        if has_ap:
            cleaned = _strip_unsupported(raw_schema)
            cfg = types.GenerateContentConfig(
                system_instruction=system or None,
                response_mime_type="application/json",
                response_json_schema=cleaned,
                temperature=temperature,
            )
        else:
            cfg = types.GenerateContentConfig(
                system_instruction=system or None,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            )
        try:
            resp, used_model, dt = self._call_with_fallback(prompt, cfg, model)
            raw = getattr(resp, "text", "") or ""
            parsed = schema.model_validate_json(raw)
        except ProviderError as exc:
            err_l = str(exc).lower()
            if "too many states" not in err_l and "invalid_argument" not in err_l and "schema" not in err_l:
                raise
            # Schema exceeded Gemini grammar state cap on constrained decoding; fall back cleanly to JSON-mode + Pydantic validation
            def _compact_schema(node: object) -> object:
                if isinstance(node, dict):
                    node.pop("additionalProperties", None)
                    node.pop("description", None)
                    node.pop("title", None)
                    for val in node.values():
                        _compact_schema(val)
                elif isinstance(node, list):
                    for item in node:
                        _compact_schema(item)
                return node

            compact_fields = _json.dumps(_compact_schema(schema.model_json_schema()), separators=(",", ":"))
            json_prompt = (
                f"{prompt}\n\n"
                f"Return STRICT valid JSON conforming to `{schema.__name__}` matching schema:\n{compact_fields}"
            )
            json_cfg = types.GenerateContentConfig(
                system_instruction=system or None,
                response_mime_type="application/json",
                temperature=temperature,
            )
            resp, used_model, dt = self._call_with_fallback(json_prompt, json_cfg, model)
            raw = getattr(resp, "text", "") or ""
            for _rep in range(3):
                try:
                    parsed = schema.model_validate_json(raw)
                    break
                except Exception as val_exc:
                    if _rep == 2:
                        raise
                    repair_prompt = (
                        f"Fix ONLY the JSON validation error below and output complete valid JSON matching `{schema.__name__}`:\n"
                        f"ERROR: {val_exc}\n\nBAD_JSON:\n{raw}"
                    )
                    resp, used_model, dt = self._call_with_fallback(repair_prompt, json_cfg, used_model)
                    raw = getattr(resp, "text", "") or ""
        um = getattr(resp, "usage_metadata", None)
        pt = int(getattr(um, "prompt_token_count", 0) or 0)
        ct = int(getattr(um, "candidates_token_count", 0) or 0)
        self.usage.add(pt, ct)
        return parsed, LLMResponse(text=raw, model=used_model, provider=self.name, latency_s=dt, prompt_tokens=pt, completion_tokens=ct)
