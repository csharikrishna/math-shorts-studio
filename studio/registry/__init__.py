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
    return get_default_registry().capabilities_prompt_summary(backend=backend, aspect=aspect, compact=compact, certified_only=True)


def backend_capability_matrix() -> dict[str, dict[str, bool]]:
    """Return authoritative per-backend support matrix generated strictly from certified implemented specs."""
    reg = get_default_registry()
    matrix: dict[str, dict[str, bool]] = {}
    for spec in reg.all_specs():
        certified = bool(spec.implemented and not spec.certification_errors())
        matrix[f"{spec.category}:{spec.name}"] = {
            "manimgx": bool(certified and spec.supported_backends in ("both", "manimgx") and spec.backend_support.get("manimgx", True)),
            "hyperframes": bool(certified and spec.supported_backends in ("both", "hyperframes") and spec.backend_support.get("hyperframes", True)),
        }
    return matrix


def verify_registry_contracts(registry: ComponentRegistry | None = None) -> list[str]:
    """Run enforceable contract certification checks across all registered components."""
    reg = registry or get_default_registry()
    failures: list[str] = []
    for spec in reg.all_specs():
        if not spec.implemented:
            continue
        errs = spec.certification_errors()
        for msg in errs:
            failures.append(f"{spec.category}:{spec.name} {msg}")
    return failures


def select_scene_backend(scene: object, requested: str = "auto") -> str:
    """Choose best backend ('manimgx' or 'hyperframes') per scene when requested in ('auto', 'hybrid')."""
    req = (requested or "manimgx").strip().lower()
    if req in ("manimgx", "hyperframes"):
        return req
    objs = getattr(scene, "objects", [])
    kinds = [str(getattr(o, "kind", "")) for o in objs]
    constructions = [str(getattr(c, "op", "")) for c in getattr(scene, "constructions", [])]
    # ManimGX owns heavy geometric dissections, vector fields, parametric morphisms, and Typst LaTeX
    if any(k in ("angle_arc", "projection", "parametric_curve", "vector_field") for k in kinds):
        return "manimgx"
    if any("pythagorean" in c or "rotate" in c or "triangle" in c for c in constructions):
        return "manimgx"
    # HyperFrames excels cleanly on editorial tables, bar charts, network graphs, time-series comparisons, and crisp numeric ticks
    if any(k in ("table", "bar_chart", "graph_network") for k in kinds) or (
        getattr(scene, "composition", "") in ("dual_time_series", "before_after_simulation", "time_frequency_split", "model_with_readout")
        and "parametric_curve" not in kinds
    ):
        return "hyperframes"
    return "manimgx"


__all__ = [
    "ComponentCategory",
    "ComponentRegistry",
    "ComponentSpec",
    "adapt_composition_for_aspect",
    "backend_capability_matrix",
    "capabilities_prompt_summary",
    "get_default_registry",
    "select_scene_backend",
    "verify_registry_contracts",
]
