"""Core adaptable component registry interfaces for creative and director prompts."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

ComponentCategory = Literal[
    "math_operation",
    "visual_object",
    "animation_mechanism",
    "composition_template",
    "audio_cue",
    "caption_style",
]

BackendSupport = Literal["both", "manimgx", "hyperframes"]


class ComponentSpec(BaseModel):
    """Self-describing reusable capability exposed dynamically to planning agents."""

    name: str
    category: ComponentCategory
    description: str
    inputs_schema: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    supported_backends: BackendSupport = "both"
    aspect_support: list[str] = Field(default_factory=lambda: ["16:9", "9:16"])
    math_grounding: str = ""
    example_use_case: str = ""


class ComponentRegistry:
    """Thread-safe in-memory catalog of adaptable math, visual, motion, and layout primitives."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], ComponentSpec] = {}

    def register(self, spec: ComponentSpec) -> ComponentSpec:
        self._items[(spec.category, spec.name)] = spec
        return spec

    def register_many(self, specs: list[ComponentSpec]) -> None:
        for s in specs:
            self.register(s)

    def get(self, name: str, category: ComponentCategory | None = None) -> ComponentSpec | None:
        if category is not None:
            return self._items.get((category, name))
        for (_cat, item_name), spec in self._items.items():
            if item_name == name:
                return spec
        return None

    def list_by_category(self, category: ComponentCategory) -> list[ComponentSpec]:
        return [spec for (cat, _), spec in self._items.items() if cat == category]

    def search(self, query: str) -> list[ComponentSpec]:
        q = query.strip().lower()
        if not q:
            return self.all_specs()
        hits: list[ComponentSpec] = []
        for spec in self._items.values():
            hay = f"{spec.name} {spec.category} {spec.description} {spec.math_grounding} {spec.example_use_case}".lower()
            if q in hay:
                hits.append(spec)
        return hits

    def all_specs(self) -> list[ComponentSpec]:
        return list(self._items.values())

    def capabilities_prompt_summary(
        self,
        *,
        backend: str | None = None,
        aspect: str | None = None,
        compact: bool = True,
    ) -> str:
        """Render an up-to-date capability summary injected into Gemini creative/director prompts."""
        lines: list[str] = ["ADAPTABLE ENGINE CAPABILITY REGISTRY (use any combination; zero hardcoded topic routes):"]
        order: list[ComponentCategory] = [
            "math_operation",
            "visual_object",
            "animation_mechanism",
            "composition_template",
            "caption_style",
            "audio_cue",
        ]
        for cat in order:
            specs = self.list_by_category(cat)
            filtered = [
                s
                for s in specs
                if (backend is None or s.supported_backends in ("both", backend))
                and (aspect is None or aspect in s.aspect_support)
            ]
            if not filtered:
                continue
            lines.append(f"\n[{cat.upper()}] ({len(filtered)} available)")
            for s in filtered:
                inputs = ", ".join(s.inputs_schema[:5]) if s.inputs_schema else "none"
                if compact:
                    ground = f" | grounding={s.math_grounding}" if s.math_grounding else ""
                    lines.append(f"- {s.name}: {s.description} (inputs: {inputs}; outputs: {', '.join(s.outputs)}{ground})")
                else:
                    lines.append(
                        f"- {s.name} [{s.supported_backends}, {'/'.join(s.aspect_support)}]: {s.description} "
                        f"| inputs=[{inputs}] | outputs=[{', '.join(s.outputs)}] | e.g. {s.example_use_case}"
                    )
        return "\n".join(lines)
