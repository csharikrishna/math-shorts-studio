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
    "angle_arc",
    "projection",
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
    card: bool = False


ConstructionKind = Literal[
    "right_triangle",
    "square_on_segment",
    "pythagorean_dissection",
    "translate_polygon",
    "rotate_polygon",
]


class ConstructionParams(BaseModel):
    """Typed parameters for deterministic mathematical construction operations."""

    a: Optional[float] = None
    b: Optional[float] = None
    origin: Optional[list[float]] = None
    edge_index: Optional[int] = None
    outward: Optional[bool] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    angle_deg: Optional[float] = None
    center: Optional[list[float]] = None
    p1: Optional[list[float]] = None
    p2: Optional[list[float]] = None


class ConstructionOp(BaseModel):
    """Typed deterministic mathematical construction operation."""

    op: ConstructionKind
    output_ids: list[str] = Field(default_factory=list)
    source_id: Optional[str] = None
    params: ConstructionParams = Field(default_factory=ConstructionParams)
    color_role: ColorRole = "primary"
    persist: bool = True
    label: str = ""


ClaimRuleKind = Literal[
    "right_triangle",
    "square",
    "congruent_triangles",
    "area_conservation",
    "partition_of",
    "nonzero_area",
]


class StructuredClaim(BaseModel):
    """Explicit mathematical constraint verified against live geometry."""

    kind: ClaimRuleKind
    object_ids: list[str] = Field(default_factory=list)
    container_id: Optional[str] = None
    statement: str = ""


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
    constructions: list[ConstructionOp] = Field(default_factory=list)
    claims: list[StructuredClaim] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    events: list[AnimationEvent] = Field(default_factory=list)


class VideoPlanV3(BaseModel):
    schema_version: str = "3.0"
    topic: str
    title: str
    audience: str = "general"
    strategy: str = ""
    target_duration_s: float = 55.0
    scenes: list[SceneV3]
