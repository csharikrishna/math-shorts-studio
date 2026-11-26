"""Versioned multi-scene VideoPlan schema (V2) for composable educational filmmaking."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

from studio.ai.schema import ColorRole, SfxCue

PrimitiveKind = Literal[
    "contrast_waves",
    "single_wave",
    "stacked_addition",
    "progressive_sum",
    "zoom_overshoot",
    "spectrum_bars",
]


class ScenePrimitive(BaseModel):
    kind: PrimitiveKind
    expr: Optional[str] = None
    secondary_expr: Optional[str] = None
    harmonics: list[int] = Field(default_factory=list)
    label: str = ""


class SceneSpec(BaseModel):
    scene_id: str
    purpose: str
    narration: str
    formula: Optional[str] = None
    color_role: ColorRole = "primary"
    sfx_cue: SfxCue = "pop_marimba"
    primitive: ScenePrimitive
    y_range: Optional[list[float]] = None


class VideoPlan(BaseModel):
    schema_version: str = "2.0"
    topic: str
    title: str
    x_range: list[float] = Field(default_factory=lambda: [-3.6, 3.6])
    y_range: list[float] = Field(default_factory=lambda: [-1.6, 1.6])
    scenes: list[SceneSpec]
