"""Composable Object-and-Event Scene Language (VideoPlan V3)."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

from studio.ai.schema import ColorRole, SfxCue

ObjectKind = Literal[
    "axes",
    "function_plot",
    "parametric_curve",
    "bar_chart",
    "equation",
    "point",
    "line",
    "vector",
    "polygon",
    "circle",
    "number_line",
    "annotation",
]

EventOp = Literal[
    "create",
    "draw",
    "transform",
    "highlight",
    "move",
    "scale",
    "rotate",
    "remove",
    "focus",
    "animate_param",
]


class SceneObject(BaseModel):
    id: str = Field(description="Unique snake_case object identifier.")
    kind: ObjectKind
    persist: bool = True
    color_role: ColorRole = "primary"
    # Math / geometry fields (kind-dependent)
    expr: Optional[str] = None
    t_range: Optional[list[float]] = None
    x_range: Optional[list[float]] = None
    y_range: Optional[list[float]] = None
    tex: Optional[str] = None
    text: Optional[str] = None
    values: list[float] = Field(default_factory=list)
    bar_labels: list[str] = Field(default_factory=list)
    log_scale: bool = False
    coords: Optional[list[float]] = None
    end_coords: Optional[list[float]] = None
    vertices: list[list[float]] = Field(default_factory=list)
    radius: Optional[float] = None
    axes_id: Optional[str] = None
    equal_aspect: bool = False
    param_name: Optional[str] = None
    param_init: float = 0.0
    label: str = ""


class AnimationEvent(BaseModel):
    op: EventOp
    target_id: str
    source_id: Optional[str] = None
    anchor_mark: Optional[str] = None
    shift: Optional[list[float]] = None
    scale_factor: Optional[float] = None
    angle_deg: Optional[float] = None
    param_range: Optional[list[float]] = None


class SceneV3(BaseModel):
    scene_id: str
    purpose: str
    narration: str
    color_role: ColorRole = "primary"
    sfx_cue: SfxCue = "pop_marimba"
    objects: list[SceneObject] = Field(default_factory=list)
    events: list[AnimationEvent] = Field(default_factory=list)


class VideoPlanV3(BaseModel):
    schema_version: str = "3.0"
    topic: str
    title: str
    audience: str = "general"
    scenes: list[SceneV3]
