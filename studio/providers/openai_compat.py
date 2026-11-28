"""OpenAI-compatible providers: NVIDIA NIM, OpenRouter (free-only), Ollama, and llama.cpp."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Type, TypeVar

from pydantic import BaseModel

from studio.ai.env import nvidia_api_key
from studio.providers.base import FreeOnlyPolicyError, LLMProvider, LLMResponse, ProviderError

T = TypeVar("T", bound=BaseModel)
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


def _extract_json_block(raw: str) -> str:
    """Strip markdown fences or surrounding prose to recover the outermost JSON payload."""
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not txt or (txt.startswith("{") and txt.endswith("}")) or (txt.startswith("[") and txt.endswith("]")):
        return txt
    start_obj = txt.find("{")
    start_arr = txt.find("[")
    starts = [i for i in (start_obj, start_arr) if i >= 0]
    if not starts:
        return txt
    start = min(starts)
    open_ch = txt[start]
    close_ch = "}" if open_ch == "{" else "]"
    end = txt.rfind(close_ch)
    if end > start:
        return txt[start : end + 1].strip()
    return txt


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
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(free_only=free_only, timeout_s=timeout_s)
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.api_key = api_key
        self.max_tokens = max_tokens

    def _validate_model(self, model: str) -> None:
        pass

    def _build_payload(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        json_mode: bool = False,
    ) -> dict:
        payload: dict = {"model": model, "messages": messages, "temperature": temperature}
        if self.max_tokens is not None:
            payload["max_tokens"] = int(self.max_tokens)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, payload: dict, *, max_attempts: int = 3) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
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
                last_exc = ProviderError(f"{self.name} HTTP {exc.code}: {body}")
                if exc.code not in _TRANSIENT_HTTP_CODES or attempt == max_attempts:
                    raise last_exc from exc
                time.sleep(min(4.0, 1.0 * attempt))
            except Exception as exc:
                last_exc = ProviderError(f"{self.name} request failed: {exc}")
                if attempt == max_attempts:
                    raise last_exc from exc
                time.sleep(min(4.0, 1.0 * attempt))
        raise ProviderError(f"{self.name} request failed: {last_exc}")

    def _chat(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.6,
        json_mode: bool = False,
    ) -> LLMResponse:
        m = model or self.default_model
        self._validate_model(m)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = self._build_payload(m, messages, temperature, json_mode=json_mode)
        t0 = time.perf_counter()
        try:
            data = self._post(payload)
        except ProviderError as exc:
            # Retry once without response_format if the backend rejects json_object
            if json_mode and "response_format" in payload and "400" in str(exc):
                payload = self._build_payload(m, messages, temperature, json_mode=False)
                data = self._post(payload)
            else:
                raise
        dt = time.perf_counter() - t0
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content")
        if text is None:
            text = msg.get("reasoning_content") or ""
        usage = data.get("usage", {}) or {}
        pt, ct = int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
        self.usage.add(pt, ct)
        return LLMResponse(text=str(text), model=m, provider=self.name, latency_s=dt, prompt_tokens=pt, completion_tokens=ct)

    def generate_text(self, prompt: str, *, system: str = "", model: str | None = None, temperature: float = 0.6) -> LLMResponse:
        return self._chat(prompt, system=system, model=model, temperature=temperature, json_mode=False)

    def generate_structured(
        self, prompt: str, schema: Type[T], *, system: str = "", model: str | None = None, temperature: float = 0.6
    ) -> tuple[T, LLMResponse]:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        sys_prompt = f"{system}\n\nReturn ONLY valid JSON matching this schema:\n{schema_json}".strip()
        resp = self._chat(prompt, system=sys_prompt, model=model, temperature=temperature, json_mode=True)
        raw = _extract_json_block(resp.text)
        for _rep in range(3):
            try:
                parsed = schema.model_validate_json(raw)
                return parsed, resp
            except Exception as exc:
                if _rep == 2:
                    raise ProviderError(f"{self.name} JSON validation failed for {schema.__name__}: {exc}") from exc
                repair_prompt = (
                    f"Fix ONLY the JSON validation error below and return complete valid JSON matching `{schema.__name__}`:\n"
                    f"ERROR: {exc}\n\nBAD_JSON:\n{raw}"
                )
                resp = self._chat(repair_prompt, system=sys_prompt, model=model, temperature=min(temperature, 0.25), json_mode=True)
                raw = _extract_json_block(resp.text)
        raise ProviderError(f"{self.name} failed structured generation for {schema.__name__}")


class NvidiaProvider(OpenAICompatProvider):
    """NVIDIA NIM OpenAI-compatible provider (defaults to nvidia/nemotron-3.5-lightning-30b-a3b)."""

    name = "nvidia"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        base_url: str | None = None,
        free_only: bool = True,
        timeout_s: float = 120.0,
        max_tokens: int = 16384,
        enable_thinking: bool = False,
    ) -> None:
        if api_key is None:
            try:
                key = nvidia_api_key()
            except RuntimeError as exc:
                raise ProviderError(str(exc)) from exc
        else:
            key = api_key
            if not key:
                raise ProviderError("NVIDIA_API_KEY is empty.")
        model_name = default_model or os.environ.get("NVIDIA_MODEL") or DEFAULT_NVIDIA_MODEL
        url = base_url or os.environ.get("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
        self.enable_thinking = bool(enable_thinking)
        super().__init__(url, model_name, api_key=key, free_only=free_only, timeout_s=timeout_s, max_tokens=max_tokens)

    def _build_payload(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        json_mode: bool = False,
    ) -> dict:
        payload = super()._build_payload(model, messages, temperature, json_mode=json_mode)
        # Structured JSON calls disable thinking tokens to maximize token efficiency and valid JSON yield.
        use_thinking = self.enable_thinking and not json_mode
        payload["chat_template_kwargs"] = {"enable_thinking": use_thinking}
        if use_thinking and self.max_tokens:
            payload["reasoning_budget"] = int(self.max_tokens)
        return payload


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
