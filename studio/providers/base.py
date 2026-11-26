"""Provider-independent AI abstraction with strict free-only policy and usage tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    """Raised when an AI provider call fails or exhausts retries."""


class FreeOnlyPolicyError(ProviderError):
    """Raised when a billable/non-free model is requested under free_only=True."""


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def add(self, prompt: int = 0, completion: int = 0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.requests += 1


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMProvider(ABC):
    """Unified interface for text and structured JSON generation."""

    name: str = "base"

    def __init__(self, free_only: bool = True, timeout_s: float = 60.0) -> None:
        self.free_only = free_only
        self.timeout_s = timeout_s
        self.usage = TokenUsage()

    @abstractmethod
    def generate_text(self, prompt: str, *, system: str = "", model: str | None = None, temperature: float = 0.6) -> LLMResponse:
        """Generate plain text."""

    @abstractmethod
    def generate_structured(
        self, prompt: str, schema: Type[T], *, system: str = "", model: str | None = None, temperature: float = 0.6
    ) -> tuple[T, LLMResponse]:
        """Generate and validate a Pydantic model."""
