"""Renderer-independent evaluated scene runtime state across ManimGX & HyperFrames."""

from __future__ import annotations

import math
from typing import Any, Optional
from pydantic import BaseModel, Field

from studio.ai.v3_schema import SceneObject, SceneV3, VideoPlanV3
from studio.ai.world_model import materialize_constructions, rotate_polygon, translate_polygon


class ObjectRuntimeState(BaseModel):
    """Evaluated live mathematical & visual state of one SceneObject across beats/backends."""

    object_id: str
    kind: str
    visible: bool = False
    persist: bool = True
    color_role: str = "primary"
    axes_id: Optional[str] = None
    group_id: Optional[str] = None
    coords: Optional[list[float]] = None
    end_coords: Optional[list[float]] = None
    vertices: list[list[float]] = Field(default_factory=list)
    points_list: list[list[float]] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    bar_labels: list[str] = Field(default_factory=list)
    matrix_rows: list[list[str]] = Field(default_factory=list)
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[list[str]] = Field(default_factory=list)
    edge_pairs: list[list[str]] = Field(default_factory=list)
    node_labels: list[str] = Field(default_factory=list)
    expr: Optional[str] = None
    tex: Optional[str] = None
    text: Optional[str] = None
    x_range: Optional[list[float]] = None
    y_range: Optional[list[float]] = None
    t_range: Optional[list[float]] = None
    radius: Optional[float] = None
    equal_aspect: bool = False
    param_name: Optional[str] = None
    param_value: float = 0.0
    label: str = ""
    card: bool = False
    panel_slot: str = "auto"
    translation: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    rotation_deg: float = 0.0
    scale_factor: float = 1.0
    step_history: list[list[float]] = Field(default_factory=list)

    def to_scene_object(self) -> SceneObject:
        """Materialize current evaluated state back into a SceneObject declaration."""
        return SceneObject(
            id=self.object_id,
            kind=self.kind,  # type: ignore[arg-type]
            persist=self.persist,
            color_role=self.color_role,  # type: ignore[arg-type]
            expr=self.expr,
            t_range=list(self.t_range) if self.t_range is not None else None,
            x_range=list(self.x_range) if self.x_range is not None else None,
            y_range=list(self.y_range) if self.y_range is not None else None,
            tex=self.tex,
            text=self.text,
            values=list(self.values),
            bar_labels=list(self.bar_labels),
            coords=list(self.coords) if self.coords is not None else None,
            end_coords=list(self.end_coords) if self.end_coords is not None else None,
            vertices=[list(v) for v in self.vertices],
            radius=self.radius,
            axes_id=self.axes_id,
            equal_aspect=self.equal_aspect,
            param_name=self.param_name,
            param_init=float(self.param_value),
            label=self.label,
            card=self.card,
            matrix_rows=[list(r) for r in self.matrix_rows],
            table_headers=list(self.table_headers),
            table_rows=[list(r) for r in self.table_rows],
            points_list=[list(p) for p in self.points_list],
            edge_pairs=[list(e) for e in self.edge_pairs],
            node_labels=list(self.node_labels),
            panel_slot=self.panel_slot,  # type: ignore[arg-type]
        )


class SceneRuntimeState(BaseModel):
    """Canonical backend-independent runtime snapshot across ManimGX & HyperFrames boundaries."""

    objects: dict[str, ObjectRuntimeState] = Field(default_factory=dict)
    removed_ids: set[str] = Field(default_factory=set)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    composition: str = "auto"
    camera_zoom: float = 1.0
    camera_target_id: Optional[str] = None
    active_equation_id: Optional[str] = None
    active_annotation_id: Optional[str] = None
    model_params: dict[str, float] = Field(default_factory=dict)
    scene_history: list[str] = Field(default_factory=list)


def _scale_pts(pts: list[list[float]], factor: float) -> list[list[float]]:
    if not pts:
        return []
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return [[cx + (p[0] - cx) * factor, cy + (p[1] - cy) * factor] for p in pts]


def _rotate_vec(v: list[float], origin: list[float], deg: float) -> list[float]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = float(v[0]) - float(origin[0]), float(v[1]) - float(origin[1])
    return [float(origin[0]) + dx * c - dy * s, float(origin[1]) + dx * s + dy * c]


def advance_scene_state(state: SceneRuntimeState, scene: SceneV3) -> SceneRuntimeState:
    """Purely advance SceneRuntimeState through one SceneV3 (objects + post-event transforms + retirement)."""
    nxt = state.model_copy(deep=True)
    nxt.composition = str(scene.composition or nxt.composition)
    nxt.scene_history.append(scene.scene_id)

    referenced = (
        {e.target_id for e in scene.events}
        | {e.source_id for e in scene.events if e.source_id}
        | {o.axes_id for o in scene.objects if o.axes_id}
    )
    has_new_apparatus = any(o.kind in ("axes", "table", "matrix", "bar_chart", "graph_network") for o in scene.objects)
    if has_new_apparatus:
        retired_axes: set[str] = set()
        for oid, ost in list(nxt.objects.items()):
            if not ost.visible or oid in referenced or ost.axes_id in referenced:
                continue
            if ost.kind in ("axes", "table", "matrix", "bar_chart", "graph_network", "annotation", "function_plot", "parametric_curve", "data_series", "waveform", "particle_system", "mechanical_body", "vector_field"):
                ost.visible = False
                nxt.removed_ids.add(oid)
                if ost.kind == "axes":
                    retired_axes.add(oid)
        if retired_axes:
            for oid, ost in list(nxt.objects.items()):
                if oid not in referenced and ost.axes_id in retired_axes:
                    ost.visible = False
                    nxt.removed_ids.add(oid)

    new_eq = [o.id for o in scene.objects if o.kind == "equation"]
    if new_eq and nxt.active_equation_id and nxt.active_equation_id in nxt.objects and nxt.active_equation_id not in referenced:
        nxt.objects[nxt.active_equation_id].visible = False
        nxt.removed_ids.add(nxt.active_equation_id)

    for obj in scene.objects:
        nxt.removed_ids.discard(obj.id)
        nxt.objects[obj.id] = ObjectRuntimeState(
            object_id=obj.id,
            kind=obj.kind,
            visible=True,
            persist=bool(obj.persist),
            color_role=obj.color_role,
            axes_id=obj.axes_id,
            coords=list(obj.coords) if obj.coords is not None else None,
            end_coords=list(obj.end_coords) if obj.end_coords is not None else None,
            vertices=[list(v) for v in obj.vertices],
            points_list=[list(p) for p in obj.points_list],
            values=list(obj.values),
            bar_labels=list(obj.bar_labels),
            matrix_rows=[list(r) for r in obj.matrix_rows],
            table_headers=list(obj.table_headers),
            table_rows=[list(r) for r in obj.table_rows],
            edge_pairs=[list(e) for e in obj.edge_pairs],
            node_labels=list(obj.node_labels),
            expr=obj.expr,
            tex=obj.tex,
            text=obj.text,
            x_range=list(obj.x_range) if obj.x_range is not None else None,
            y_range=list(obj.y_range) if obj.y_range is not None else None,
            t_range=list(obj.t_range) if obj.t_range is not None else None,
            radius=obj.radius,
            equal_aspect=bool(obj.equal_aspect),
            param_name=obj.param_name,
            param_value=float(obj.param_init),
            label=obj.label,
            card=bool(obj.card),
            panel_slot=obj.panel_slot,
        )

    if getattr(scene, "active_model_params", None):
        for mk, mv in scene.active_model_params.items():
            nxt.model_params[str(mk)] = float(mv)
    for ev in scene.events:
        if ev.op in ("animate_param", "set_param"):
            obj_m = next((o for o in scene.objects if o.id == ev.target_id), None)
            m_key = str((obj_m.model_id if obj_m else None) or ev.target_id)
            if ev.op == "set_param" and ev.target_param is not None:
                nxt.model_params[m_key] = float(ev.target_param)
            elif ev.param_range and len(ev.param_range) == 2:
                nxt.model_params[m_key] = float(ev.param_range[1])
            elif ev.param_values:
                nxt.model_params[m_key] = float(ev.param_values[-1])
        if ev.op == "duplicate" and ev.source_id and ev.source_id in nxt.objects:
            src_st = nxt.objects[ev.source_id].model_copy(deep=True)
            src_st.object_id = ev.target_id
            src_st.visible = True
            nxt.removed_ids.discard(ev.target_id)
            if ev.shift:
                dx, dy = float(ev.shift[0]), float(ev.shift[1])
                if src_st.vertices:
                    src_st.vertices = translate_polygon(src_st.vertices, dx, dy)
                if src_st.coords and len(src_st.coords) >= 2:
                    src_st.coords = [src_st.coords[0] + dx, src_st.coords[1] + dy]
                src_st.translation = [src_st.translation[0] + dx, src_st.translation[1] + dy]
            nxt.objects[ev.target_id] = src_st
            continue

        st = nxt.objects.get(ev.target_id)
        if st is None:
            continue

        if ev.op in ("create", "draw", "reveal", "trace", "sequence", "parallel"):
            st.visible = True
            nxt.removed_ids.discard(st.object_id)
        elif ev.op in ("transform", "morph") and ev.source_id and ev.source_id in nxt.objects:
            nxt.objects[ev.source_id].visible = False
            nxt.removed_ids.add(ev.source_id)
            st.visible = True
        elif ev.op == "group":
            members = [ev.target_id] + ([ev.source_id] if ev.source_id else [])
            nxt.groups[ev.target_id] = members
            for mid in members:
                if mid in nxt.objects:
                    nxt.objects[mid].group_id = ev.target_id
        elif ev.op == "move" and ev.shift:
            dx, dy = float(ev.shift[0]), float(ev.shift[1])
            st.translation = [st.translation[0] + dx, st.translation[1] + dy]
            if st.vertices:
                st.vertices = translate_polygon(st.vertices, dx, dy)
            if st.coords and len(st.coords) >= 2:
                st.coords = [st.coords[0] + dx, st.coords[1] + dy]
            if st.end_coords and len(st.end_coords) >= 2:
                st.end_coords = [st.end_coords[0] + dx, st.end_coords[1] + dy]
            if st.points_list:
                st.points_list = [[p[0] + dx, p[1] + dy] for p in st.points_list]
        elif ev.op == "rotate" and ev.angle_deg is not None:
            ang = float(ev.angle_deg)
            st.rotation_deg = (st.rotation_deg + ang) % 360.0
            if st.vertices:
                cx = sum(p[0] for p in st.vertices) / len(st.vertices)
                cy = sum(p[1] for p in st.vertices) / len(st.vertices)
                st.vertices = rotate_polygon(st.vertices, math.radians(ang), (cx, cy))
            elif st.coords and st.end_coords and len(st.coords) >= 2 and len(st.end_coords) >= 2:
                st.end_coords = _rotate_vec(st.end_coords, st.coords, ang)
        elif ev.op == "scale" and ev.scale_factor is not None:
            sf = float(ev.scale_factor)
            st.scale_factor *= sf
            if st.vertices:
                st.vertices = _scale_pts(st.vertices, sf)
            if st.coords and st.end_coords and len(st.coords) >= 2 and len(st.end_coords) >= 2:
                st.end_coords = [
                    st.coords[0] + (st.end_coords[0] - st.coords[0]) * sf,
                    st.coords[1] + (st.end_coords[1] - st.coords[1]) * sf,
                ]
            if st.radius is not None:
                st.radius = float(st.radius) * sf
        elif ev.op == "zoom":
            nxt.camera_zoom = float(ev.scale_factor or 1.25)
            nxt.camera_target_id = ev.target_id
        elif ev.op == "focus":
            nxt.camera_target_id = ev.target_id
        elif ev.op in ("animate_param", "interpolate_values"):
            if ev.param_range and len(ev.param_range) >= 2:
                st.param_value = float(ev.param_range[-1])
            elif ev.param_values:
                st.param_value = float(ev.param_values[-1])
        elif ev.op == "iterate_states":
            if ev.param_values:
                st.param_value = float(ev.param_values[-1])
                st.step_history.append([float(v) for v in ev.param_values])
        elif ev.op == "remove":
            st.visible = False
            st.persist = False
            nxt.removed_ids.add(ev.target_id)

    if new_eq:
        nxt.active_equation_id = new_eq[-1]
    new_ann = [o.id for o in scene.objects if o.kind == "annotation"]
    if new_ann:
        nxt.active_annotation_id = new_ann[-1]

    # End-of-scene non-persistent retirement: removed objects must NOT survive to downstream backends
    for obj in scene.objects:
        if not obj.persist and obj.id in nxt.objects:
            nxt.objects[obj.id].visible = False
            nxt.removed_ids.add(obj.id)
    return nxt


def evaluate_plan_states(plan: VideoPlanV3) -> list[SceneRuntimeState]:
    """Return ordered list of evaluated SceneRuntimeState snapshots (one per scene)."""
    mat = materialize_constructions(plan)
    cur = SceneRuntimeState()
    out: list[SceneRuntimeState] = []
    for sc in mat.scenes:
        cur = advance_scene_state(cur, sc)
        out.append(cur)
    return out


def materialize_carry_objects(
    state: SceneRuntimeState,
    needed_ids: set[str],
    *,
    include_visible_persistent: bool = True,
) -> list[SceneObject]:
    """Emit evaluated carry SceneObjects for cross-backend segment boundaries without resurrecting removed items."""
    carry: list[SceneObject] = []
    for oid, ost in state.objects.items():
        if oid in state.removed_ids or not ost.visible:
            continue
        if (oid in needed_ids) or (include_visible_persistent and ost.persist):
            carry.append(ost.to_scene_object())
    return carry
