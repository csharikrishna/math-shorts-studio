"""Default unified ComponentRegistry combining math ops, visuals, motion, compositions, captions, and cues."""

from __future__ import annotations

from functools import lru_cache

from studio.registry.animations import register_animations
from studio.registry.base import ComponentCategory, ComponentRegistry, ComponentSpec
from studio.registry.compositions import adapt_composition_for_aspect, register_compositions
from studio.registry.math_ops import register_math_ops
from studio.registry.visuals import register_visuals


@lru_cache(maxsize=1)
def get_default_registry() -> ComponentRegistry:
    reg = ComponentRegistry()
    register_math_ops(reg)
    register_visuals(reg)
    register_animations(reg)
    register_compositions(reg)
    return reg


def capabilities_prompt_summary(*, backend: str | None = None, aspect: str | None = None, compact: bool = True) -> str:
    return get_default_registry().capabilities_prompt_summary(backend=backend, aspect=aspect, compact=compact)


__all__ = [
    "ComponentCategory",
    "ComponentRegistry",
    "ComponentSpec",
    "adapt_composition_for_aspect",
    "capabilities_prompt_summary",
    "get_default_registry",
]
