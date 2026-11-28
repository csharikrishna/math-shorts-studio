"""Composable Object-and-Event ManimGX renderer for VideoPlan V3/V3.1."""

from __future__ import annotations

import json
import os
from pathlib import Path
import manimgx as m
import numpy as np

from scenes.multi_scene import plot_clipped_segments
from studio.ai.expr import compile_expr
from studio.ai.v3_schema import AnimationEvent, SceneObject, VideoPlanV3
from studio.ai.world_model import materialize_constructions
from studio.audio.tts import BeatSpec
from studio.config import CAPTION_FONT, HEADER_FONT, MATH_SERIF_FONT
from studio.design.fonts import effective_px_scale, manimgx_font_paths, px_to_manim_font_size
from studio.layout import CaptionCard, fit_mobject
from studio.layout.solver import (
    AdaptiveFrameZones,
    PixelBox,
    compute_adaptive_zones,
    fit_axes_pixel_box,
    px_box_to_manim,
    resolve_composition_layout,
    resolve_non_overlapping_boxes,
)
from studio.scene import StoryScene

_FONT_KWARGS: dict = {"font_paths": manimgx_font_paths()} if manimgx_font_paths() else {}

DEFAULT_V3_PLAN = Path(__file__).resolve().parent.parent / "plans" / "v3_demo.json"
V3_PLAN_PATH = Path(os.environ.get("STUDIO_V3_PLAN", str(DEFAULT_V3_PLAN)))
V3_PLAN = (
    materialize_constructions(VideoPlanV3.model_validate_json(V3_PLAN_PATH.read_text(encoding="utf-8")))
    if V3_PLAN_PATH.exists()
    else VideoPlanV3(topic="Demo", title="Demo", scenes=[])
)

ROLE_TO_THEME = {
    "primary": "accent_primary",
    "secondary": "accent_secondary",
    "tertiary": "accent_tertiary",
    "success": "success",
    "highlight": "accent_highlight",
    "danger": "danger",
}


def _clip_segment_rect(
    p0: list[float],
    p1: list[float],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[list[float], list[float]] | None:
    """Liang-Barsky 2D segment clip against axis bounds."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0 - xmin),
        (dx, xmax - x0),
        (-dy, y0 - ymin),
        (dy, ymax - y0),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    return ([x0 + t0 * dx, y0 + t0 * dy], [x0 + t1 * dx, y0 + t1 * dy])


def _clip_polyline_runs(
    pts: list[list[float]],
    xr: list[float],
    yr: list[float],
) -> list[list[list[float]]]:
    """Clip a 2D polyline to [xr, yr] and split into contiguous in-bounds runs."""
    xmin, xmax = min(float(xr[0]), float(xr[1])), max(float(xr[0]), float(xr[1]))
    ymin, ymax = min(float(yr[0]), float(yr[1])), max(float(yr[0]), float(yr[1]))
    runs: list[list[list[float]]] = []
    cur: list[list[float]] = []
    for a, b in zip(pts[:-1], pts[1:]):
        clipped = _clip_segment_rect(a, b, xmin, xmax, ymin, ymax)
        if clipped is None:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
            continue
        ca, cb = clipped
        if not cur:
            cur = [ca, cb]
        else:
            last = cur[-1]
            if abs(last[0] - ca[0]) + abs(last[1] - ca[1]) <= 1e-5:
                cur.append(cb)
            else:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = [ca, cb]
    if len(cur) >= 2:
        runs.append(cur)
    return runs


class V3Scene(StoryScene):
    """Persistent object-and-event educational animation engine (VideoPlan V3/V3.1)."""

    HAS_HEADER_BANNER = False
    HAS_EQUATION_BAND = any(any(o.kind == "equation" for o in sc.objects) for sc in V3_PLAN.scenes)

    BEATS = [
        BeatSpec(id=s.scene_id, narration=s.narration, caption=s.purpose, pause_after=0.30, speaker=s.speaker)
        for s in V3_PLAN.scenes
    ]

    def construct(self) -> None:
        self.setup_studio()
        L, T = self.layout, self.theme
        if not V3_PLAN.scenes:
            return
        self.caption_card = CaptionCard(L, T, V3_PLAN.scenes[0].narration, caption_style=getattr(self, "caption_style", None))
        self.add(self.caption_card.group)

        self.registry: dict[str, m.Mobject] = {}
        self.obj_kinds: dict[str, str] = {}
        self.obj_axes: dict[str, str | None] = {}
        self._obj_specs: dict[str, SceneObject] = {}
        self._obj_boxes: dict[str, PixelBox] = {}
        self._hidden_persisted: dict[str, m.Mobject] = {}
        self.axes_ranges: dict[str, tuple[list[float], list[float]]] = {}
        self.params: dict[str, m.ValueTracker] = {}
        self._swept_models: set[str] = set()
        self._model_selected_u: dict[str, float] = {}
        self._model_keyframes: dict[str, list] = {}
        self._active_eq_id: str | None = None
        self._active_ann_id: str | None = None
        self._geom_frame: tuple[float, np.ndarray] | None = None
        self._live_unaxed_polys: set[str] = set()
        self._px_scale = effective_px_scale(self.profile.pixel_width, self.profile.pixel_height)

        for idx, sc in enumerate(V3_PLAN.scenes):
            self._cur_scene_idx = idx
            referenced_ids = (
                {e.target_id for e in sc.events}
                | {e.source_id for e in sc.events if e.source_id}
                | {o.axes_id for o in sc.objects if o.axes_id}
            )
            referenced_ids |= {
                self.obj_axes[rid]
                for rid in list(referenced_ids)
                if self.obj_axes.get(rid)
            }
            new_eq_ids = [o.id for o in sc.objects if o.kind == "equation"]
            new_ann_ids = [o.id for o in sc.objects if o.kind == "annotation"]
            transform_srcs = {e.source_id for e in sc.events if e.op in ("transform", "morph") and e.source_id}
            has_new_apparatus = any(o.kind in ("axes", "table", "matrix", "bar_chart", "graph_network") for o in sc.objects)

            main_anims: list[m.Animation] = []
            # Restore any hidden persisted objects (and their parent axes if needed) before computing zones/obstacles
            restore_queue = list(referenced_ids)
            for rid in list(restore_queue):
                sp = self._obj_specs.get(rid)
                if sp and sp.axes_id and sp.axes_id not in restore_queue:
                    restore_queue.insert(0, sp.axes_id)
            for ref_id in restore_queue:
                if ref_id in self._hidden_persisted and ref_id not in self.registry:
                    restored_mob = self._hidden_persisted.pop(ref_id)
                    self.registry[ref_id] = restored_mob
                    if ref_id in self._obj_specs:
                        self.obj_kinds[ref_id] = self._obj_specs[ref_id].kind
                        self.obj_axes[ref_id] = self._obj_specs[ref_id].axes_id
                    main_anims.append(m.FadeIn(restored_mob))

            # Retire superseded center-stage apparatus & stale top-band equations BEFORE zone/obstacle computation
            if has_new_apparatus:
                retire_axes: set[str] = set()
                for lid in list(self.registry.keys()):
                    lkind = self.obj_kinds.get(lid, "")
                    laxes = self.obj_axes.get(lid)
                    if lid in referenced_ids or laxes in referenced_ids:
                        continue
                    if lkind in ("axes", "table", "matrix", "bar_chart", "graph_network", "annotation", "function_plot", "parametric_curve", "data_series", "waveform", "particle_system", "mechanical_body", "vector_field"):
                        old_mob = self.registry.pop(lid)
                        self.obj_kinds.pop(lid, None)
                        self.obj_axes.pop(lid, None)
                        self._obj_boxes.pop(lid, None)
                        self._live_unaxed_polys.discard(lid)
                        if lkind == "axes":
                            retire_axes.add(lid)
                        if lid in self._obj_specs and self._obj_specs[lid].persist:
                            self._hidden_persisted[lid] = old_mob
                        main_anims.append(m.FadeOut(old_mob))
                if retire_axes:
                    for lid in list(self.registry.keys()):
                        if lid in referenced_ids:
                            continue
                        if self.obj_axes.get(lid) in retire_axes:
                            old_mob = self.registry.pop(lid)
                            l_spec = self._obj_specs.get(lid)
                            self.obj_kinds.pop(lid, None)
                            self.obj_axes.pop(lid, None)
                            self._obj_boxes.pop(lid, None)
                            if l_spec and l_spec.persist:
                                self._hidden_persisted[lid] = old_mob
                            main_anims.append(m.FadeOut(old_mob))

            if new_ann_ids and self._active_ann_id and self._active_ann_id in self.registry:
                if self._active_ann_id not in referenced_ids:
                    main_anims.append(m.FadeOut(self.registry.pop(self._active_ann_id)))
                    self._obj_boxes.pop(self._active_ann_id, None)
                self._active_ann_id = None

            if (new_eq_ids or has_new_apparatus) and self._active_eq_id and self._active_eq_id in self.registry:
                if self._active_eq_id not in referenced_ids and self._active_eq_id not in transform_srcs:
                    old_eq = self.registry.pop(self._active_eq_id)
                    self._obj_boxes.pop(self._active_eq_id, None)
                    self.obj_kinds.pop(self._active_eq_id, None)
                    main_anims.append(m.FadeOut(old_eq))
                    self._active_eq_id = None
                elif new_eq_ids:
                    self._active_eq_id = None

            axes_future_eq = False
            axes_future_port_split = False
            is_port_frame = L.frame_h > L.frame_w
            ax_persist_ids = {o.id for o in sc.objects if o.kind == "axes" and o.persist}
            if ax_persist_ids:
                for later_sc in V3_PLAN.scenes[idx + 1:]:
                    if any(lo.kind == "axes" for lo in later_sc.objects):
                        break
                    later_refs = {e.target_id for e in later_sc.events} | {lo.axes_id for lo in later_sc.objects if lo.axes_id}
                    keeps_axes = bool(later_refs & ax_persist_ids)
                    if any(lo.kind == "equation" for lo in later_sc.objects):
                        axes_future_eq = True
                    if is_port_frame and keeps_axes and any(lo.kind in ("matrix", "table", "bar_chart") for lo in later_sc.objects):
                        axes_future_port_split = True
                    if not keeps_axes and any(lo.kind in ("table", "matrix", "bar_chart", "graph_network") for lo in later_sc.objects):
                        break
            has_eq = bool(new_eq_ids) or bool(self._active_eq_id and self._active_eq_id in self.registry) or axes_future_eq
            self._cur_zones: AdaptiveFrameZones = compute_adaptive_zones(
                self.profile.name,
                has_header_banner=False,
                has_equation_strip=has_eq,
                caption_style=getattr(self, "caption_style", "editorial"),
            )
            self._cur_scene_comp = resolve_composition_layout(
                getattr(sc, "composition", "auto"),
                [o.kind for o in sc.objects],
                aspect_label=self._cur_zones.aspect_label,
            )
            self._cur_has_side_ann = any(o.kind == "annotation" for o in sc.objects)
            self._scene_obstacles: list[PixelBox] = [
                self._obj_boxes[oid] for oid in self.registry if oid in self._obj_boxes
            ]
            # Auto-assign side-by-side / stacked slots if multiple conflicting apparatus/un-axed polygons co-occur
            if axes_future_port_split:
                for cand in sc.objects:
                    if cand.kind == "axes" and cand.persist and cand.panel_slot in ("auto", "center"):
                        cand.panel_slot = "top"
            has_live_axes = any(self.obj_kinds.get(lid) == "axes" for lid in self.registry)
            if has_live_axes:
                for cand in sc.objects:
                    if cand.kind in ("matrix", "table", "bar_chart") and cand.panel_slot in ("auto", "center"):
                        cand.panel_slot = "bottom" if is_port_frame else "right"
            new_app = [
                o for o in sc.objects
                if o.kind in ("axes", "table", "matrix", "bar_chart", "graph_network")
                or (o.kind == "polygon" and not o.axes_id)
            ]
            if len(new_app) >= 2:
                used_slots = {o.panel_slot for o in new_app if o.panel_slot in ("left", "right", "top", "bottom")}
                s_first = "top" if is_port_frame else "left"
                s_second = "bottom" if is_port_frame else "right"
                for cand in new_app:
                    if cand.panel_slot in ("auto", "center"):
                        if s_first not in used_slots and ("left" not in used_slots if is_port_frame else "top" not in used_slots):
                            cand.panel_slot = s_first
                            used_slots.add(s_first)
                        elif s_second not in used_slots and ("right" not in used_slots if is_port_frame else "bottom" not in used_slots):
                            cand.panel_slot = s_second
                            used_slots.add(s_second)
            self._cur_ann_index = 0
            self._cur_ann_total = sum(1 for o in sc.objects if o.kind == "annotation")
            self._cur_eq_index = 0
            self._cur_eq_total = sum(1 for o in sc.objects if o.kind == "equation")
            self._cur_scene_obj_ids = {o.id for o in sc.objects}

            for obj in sc.objects:
                if obj.id in self.registry:
                    raise RuntimeError(f"Scene {sc.scene_id}: duplicate live object {obj.id!r}")
                self._obj_specs[obj.id] = obj
                self.registry[obj.id] = self._build_object(obj)
                self.obj_kinds[obj.id] = obj.kind
                self.obj_axes[obj.id] = obj.axes_id
                if obj.kind == "annotation":
                    self._cur_ann_index += 1
                if obj.kind == "equation":
                    self._cur_eq_index += 1
                if obj.kind == "polygon" and not obj.axes_id:
                    self._live_unaxed_polys.add(obj.id)

            cues: dict[str, list[AnimationEvent]] = {}
            anchored_create_ids: set[str] = set()
            created_in_scene: set[str] = set()
            for ev in sc.events:
                if ev.op in ("create", "draw", "reveal", "transform", "morph", "trace", "sequence", "parallel", "duplicate"):
                    created_in_scene.add(ev.target_id)
                if ev.anchor_mark:
                    cues.setdefault(ev.anchor_mark, []).append(ev)
                    if ev.op in ("create", "draw", "reveal", "trace"):
                        anchored_create_ids.add(ev.target_id)

            # Prevent bare empty crosshair axes waits when every plotted curve/dot is deferred to a late bookmark
            plotted_kinds = ("data_series", "function_plot", "parametric_curve", "vector", "point", "polygon", "waveform", "particle_system")
            if any(o.kind == "axes" for o in sc.objects):
                plotted_objs = [o.id for o in sc.objects if o.kind in plotted_kinds]
                unanchored_plots = [oid for oid in plotted_objs if oid not in anchored_create_ids]
                if plotted_objs and not unanchored_plots:
                    first_plot_id = plotted_objs[0]
                    for mk in list(cues.keys()):
                        kept_evs: list[AnimationEvent] = []
                        for ce in cues[mk]:
                            if ce.target_id == first_plot_id and ce.op in ("create", "draw", "reveal", "trace"):
                                main_anims.append(self._compile_event(ce))
                                anchored_create_ids.discard(first_plot_id)
                            else:
                                kept_evs.append(ce)
                        if kept_evs:
                            cues[mk] = kept_evs
                        else:
                            cues.pop(mk)

            # Skip unanchored mutate/rotate on objects whose create waits for a later bookmark
            for ev in sc.events:
                if not ev.anchor_mark:
                    src_anchored = bool(ev.source_id and ev.source_id in anchored_create_ids)
                    tgt_anchored = ev.target_id in anchored_create_ids
                    if (tgt_anchored and ev.op in ("rotate", "move", "scale", "zoom", "transform", "morph")) or (src_anchored and ev.op in ("transform", "morph")):
                        anchor_match = ev.target_id if tgt_anchored else ev.source_id
                        for mk, evl in cues.items():
                            if any(ce.target_id == anchor_match for ce in evl):
                                evl.append(ev)
                                break
                    else:
                        main_anims.append(self._compile_event(ev))

            # Ensure EVERY declared scene object without an explicit creation event fades in cleanly
            for o in sc.objects:
                if o.id not in created_in_scene and o.id in self.registry:
                    main_anims.append(m.FadeIn(self.registry[o.id]))

            def _compile_cue_group(evs: list[AnimationEvent]) -> list[m.Animation]:
                by_target: dict[str, list[m.Animation]] = {}
                order: list[str] = []
                for e in evs:
                    anim = self._compile_event(e)
                    if e.target_id not in by_target:
                        by_target[e.target_id] = []
                        order.append(e.target_id)
                    by_target[e.target_id].append(anim)
                out_anims: list[m.Animation] = []
                for tid in order:
                    grp = by_target[tid]
                    out_anims.append(grp[0] if len(grp) == 1 else m.Succession(*grp))
                return out_anims

            cue_callables = {
                mark: (lambda evs=ev_list: _compile_cue_group(evs))
                for mark, ev_list in cues.items()
            }

            prior_stage_live = any(
                oid not in {o.id for o in sc.objects}
                and self.obj_kinds.get(oid) not in ("equation", "annotation")
                for oid in self.registry
            )
            if (idx == 0 or not prior_stage_live) and sc.objects and anchored_create_ids and all(o.id in anchored_create_ids for o in sc.objects):
                first_obj = sc.objects[0]
                main_anims.append(m.FadeIn(self.registry[first_obj.id]))
                for mk, evl in list(cues.items()):
                    cues[mk] = [e for e in evl if not (e.target_id == first_obj.id and e.op in ("create", "draw", "reveal"))]
                    if not cues[mk]:
                        cues.pop(mk)
                cue_callables = {mark: (lambda evs=ev_list: _compile_cue_group(evs)) for mark, ev_list in cues.items()}

            # Synchronize persisted swept models cleanly back to active_model_params or canonical selected_u when not re-swept
            scene_sweep_keys: set[str] = set()
            for ev_sw in sc.events:
                if ev_sw.op in ("animate_param", "set_param", "iterate_states", "interpolate_values"):
                    sp_tgt = self._obj_specs.get(ev_sw.target_id) or next((o for o in sc.objects if o.id == ev_sw.target_id), None)
                    if sp_tgt is not None:
                        scene_sweep_keys.add(f"__sweep_{getattr(sp_tgt, 'model_id', None) or ev_sw.target_id}")
            active_m_params = dict(getattr(sc, "active_model_params", {}) or {})
            for sw_k in list(self._swept_models):
                if sw_k in scene_sweep_keys or sw_k not in self.params:
                    continue
                mid_name = sw_k.replace("__sweep_", "", 1)
                tgt_u = float(self._model_selected_u.get(sw_k, 0.0))
                if mid_name in active_m_params and sw_k in self._model_keyframes:
                    kfs_m = self._model_keyframes[sw_k]
                    f_t = float(active_m_params[mid_name])
                    if kfs_m:
                        best_i = min(range(len(kfs_m)), key=lambda i_k: abs(float(kfs_m[i_k].param) - f_t))
                        tgt_u = float(kfs_m[best_i].u if getattr(kfs_m[best_i], "u", None) is not None else (best_i / max(1, len(kfs_m) - 1)))
                tr_m = self.params[sw_k]
                if abs(float(tr_m.get_value()) - tgt_u) > 1e-4:
                    main_anims.append(tr_m.animate.set_value(tgt_u))

            self.play_beat(sc.scene_id, *main_anims, cues=cue_callables or None, sfx=sc.sfx_cue)
            if new_eq_ids:
                self._active_eq_id = new_eq_ids[-1]
            if new_ann_ids:
                self._active_ann_id = new_ann_ids[-1]

            # Record renderer object bounding metadata per scene for scene-aware QA
            b_dir_env = os.environ.get("STUDIO_BUILD_DIR")
            if b_dir_env:
                meta_map = getattr(self, "_scene_meta_log", {})
                meta_map[sc.scene_id] = [
                    {
                        "id": oid,
                        "kind": self.obj_kinds.get(oid, "generic"),
                        "box": [round(b.x, 1), round(b.y, 1), round(b.w, 1), round(b.h, 1)],
                    }
                    for oid, b in self._obj_boxes.items()
                ]
                self._scene_meta_log = meta_map
                if idx == len(V3_PLAN.scenes) - 1:
                    (Path(b_dir_env) / "scene_frame_metadata.json").write_text(
                        json.dumps(meta_map, indent=2), encoding="utf-8"
                    )

            to_drop = [self.registry.pop(o.id) for o in sc.objects if not o.persist and o.id in self.registry]
            for o in sc.objects:
                if not o.persist:
                    self._live_unaxed_polys.discard(o.id)
                    self._obj_boxes.pop(o.id, None)
                    self.obj_kinds.pop(o.id, None)
                    self.obj_axes.pop(o.id, None)
            if to_drop:
                self.remove(*to_drop)
            # Reset composition frame when no un-axed polygons remain live
            if not any(oid in self.registry for oid in self._live_unaxed_polys):
                self._live_unaxed_polys.clear()
                self._geom_frame = None

    def _manim_to_px_box(self, mob: m.Mobject, role: str = "generic") -> PixelBox:
        zones = self._cur_zones
        L = self.layout
        sx = zones.width_px / L.frame_w
        sy = zones.height_px / L.frame_h
        c = mob.get_center()
        w_px = max(24.0, float(mob.width) * sx)
        h_px = max(18.0, float(mob.height) * sy)
        cx_px = (float(c[0]) * sx) + 0.5 * zones.width_px
        cy_px = 0.5 * zones.height_px - (float(c[1]) * sy)
        return PixelBox(x=cx_px - 0.5 * w_px, y=cy_px - 0.5 * h_px, w=w_px, h=h_px, role=role)

    def _slot_center(self, obj: SceneObject, scene_comp: str = "auto") -> np.ndarray:
        L = self.layout
        zones = getattr(self, "_cur_zones", None)
        is_port = L.frame_h > L.frame_w
        aspect_lbl = "9:16" if is_port else "16:9"
        eff_comp = resolve_composition_layout(scene_comp, [obj.kind], aspect_lbl)
        if zones is not None:
            st = zones.stage_box
            slot = obj.panel_slot
            if slot == "auto" and eff_comp in ("side_by_side_comparison", "stacked_comparison"):
                slot = "left" if not is_port else "top"
            if slot == "left":
                b = PixelBox(st.x, st.y, st.w, st.h * 0.53) if is_port else PixelBox(st.x, st.y, st.w * 0.48, st.h)
            elif slot == "right":
                b = PixelBox(st.x, st.y + st.h * 0.57, st.w, st.h * 0.40) if is_port else PixelBox(st.x + st.w * 0.52, st.y, st.w * 0.48, st.h)
            elif slot == "top":
                b = PixelBox(st.x, st.y, st.w, st.h * (0.53 if is_port else 0.48))
            elif slot == "bottom":
                b = PixelBox(st.x, st.y + st.h * (0.57 if is_port else 0.52), st.w, st.h * (0.40 if is_port else 0.46))
            else:
                b = st
            cx, cy, _, _ = px_box_to_manim(b, zones, L.frame_w, L.frame_h)
            return np.array([cx, cy, 0.0])
        dx = L.stage_w * 0.24
        dy = L.stage_h * 0.22
        slot = obj.panel_slot
        if slot == "left":
            return L.stage_center + (np.array([0.0, dy, 0.0]) if is_port else np.array([-dx, 0.0, 0.0]))
        if slot == "right":
            return L.stage_center + (np.array([0.0, -dy, 0.0]) if is_port else np.array([dx, 0.0, 0.0]))
        if slot == "top":
            return L.stage_center + np.array([0.0, dy, 0.0])
        if slot == "bottom":
            return L.stage_center + np.array([0.0, -dy, 0.0])
        return L.stage_center

    def _compile_event(self, ev: AnimationEvent) -> m.Animation:
        if ev.op == "duplicate" and ev.source_id and ev.source_id in self.registry and ev.target_id not in self.registry:
            copy_mob = self.registry[ev.source_id].copy()
            if ev.shift:
                copy_mob.shift(np.array([ev.shift[0], ev.shift[1], 0.0]))
            self.registry[ev.target_id] = copy_mob
            return m.TransformFromCopy(self.registry[ev.source_id], copy_mob)
        if ev.op == "remove" and ev.target_id not in self.registry:
            return m.Wait(0.01)
        if ev.target_id not in self.registry:
            raise RuntimeError(f"AnimationEvent target_id {ev.target_id!r} not found in registry.")
        mob = self.registry[ev.target_id]

        if ev.op in ("create", "draw", "reveal"):
            obj_kind = self.obj_kinds.get(ev.target_id, "")
            # Stroke-draw curves, lines, axes, polygons cleanly; fade UI text/matrices/tables crisply so letters aren't half-drawn mid-beat
            if isinstance(mob, m.VMobject) and obj_kind not in ("matrix", "table", "annotation", "equation"):
                return m.Create(mob)
            return m.FadeIn(mob)
        if ev.op in ("transform", "morph"):
            if not ev.source_id or ev.source_id not in self.registry:
                raise RuntimeError(f"{ev.op} source_id {ev.source_id!r} not found in registry.")
            src = self.registry[ev.source_id]
            src_kind = self.obj_kinds.get(ev.source_id)
            dst_kind = self.obj_kinds.get(ev.target_id)
            if src_kind != dst_kind:
                # Cross-kind visual morph (e.g. matrix -> polygon) highlights source and reveals target cleanly
                return m.AnimationGroup(m.Indicate(src, scale_factor=1.06), m.FadeIn(mob))
            if src_kind == "equation" and isinstance(src, m.MathTex) and isinstance(mob, m.MathTex):
                self.registry[ev.target_id] = mob
                self.registry.pop(ev.source_id, None)
                try:
                    return m.TransformMatchingTex(src, mob)
                except Exception:
                    return m.ReplacementTransform(src, mob)
            if src_kind == "equation" and isinstance(src, m.VGroup) and isinstance(mob, m.VGroup) and len(src) >= 2 and len(mob) >= 2:
                self.registry[ev.target_id] = mob
                self.registry.pop(ev.source_id, None)
                return m.ReplacementTransform(src, mob)
            self.registry[ev.target_id] = src
            if ev.source_id in self._live_unaxed_polys:
                self._live_unaxed_polys.discard(ev.source_id)
                self._live_unaxed_polys.add(ev.target_id)
            return m.Transform(src, mob)
        if ev.op == "highlight":
            if self.obj_kinds.get(ev.target_id) == "equation" and isinstance(mob, m.VGroup) and len(mob) >= 2:
                return m.Indicate(mob[1], scale_factor=1.08, color=self.theme.accent_highlight)
            return m.Indicate(mob, scale_factor=1.08, color=self.theme.accent_highlight)
        if ev.op == "group":
            members = [mob]
            if ev.source_id and ev.source_id in self.registry:
                members.append(self.registry[ev.source_id])
            grp = m.VGroup(*members)
            self.registry[ev.target_id] = grp
            return m.AnimationGroup(*(m.Indicate(sub, scale_factor=1.05) for sub in grp), lag_ratio=0.06)
        if ev.op == "move" and ev.shift:
            s = self._get_geom_scale() if not getattr(mob, "_uses_axes", False) else 1.0
            vec = np.array([ev.shift[0] * s, ev.shift[1] * s, (ev.shift[2] * s) if len(ev.shift) > 2 else 0.0])
            return mob.animate.shift(vec)
        if ev.op == "scale" and ev.scale_factor:
            return mob.animate.scale(ev.scale_factor)
        if ev.op == "zoom":
            sf = min(float(ev.scale_factor or 1.24), max(1.04, (self.layout.safe_w * 0.88) / max(float(mob.width), 1e-3)))
            # Camera-like reframing on target assembly + dim/brighten emphasis bounded to stage
            shift_to_center = (self.layout.stage_center - mob.get_center()) * 0.35
            return m.AnimationGroup(
                mob.animate.shift(shift_to_center).scale(sf),
                m.Indicate(mob, scale_factor=1.04, color=self.theme.accent_primary),
            )
        if ev.op == "rotate" and ev.angle_deg:
            # Never turn a matrix, table, equation, annotation, axes, or bar_chart diagonally sideways
            if self.obj_kinds.get(ev.target_id, "") in ("matrix", "table", "equation", "axes", "annotation", "bar_chart"):
                raise ValueError(f"Cannot rotate non-geometric object {ev.target_id!r} of kind {self.obj_kinds.get(ev.target_id)!r}.")
            return m.Rotate(mob, angle=float(np.deg2rad(ev.angle_deg)))
        if ev.op == "remove":
            self.registry.pop(ev.target_id, None)
            self.obj_kinds.pop(ev.target_id, None)
            self.obj_axes.pop(ev.target_id, None)
            self._obj_boxes.pop(ev.target_id, None)
            self._live_unaxed_polys.discard(ev.target_id)
            child_fades = []
            for lid in [k for k in list(self.registry.keys()) if self.obj_axes.get(k) == ev.target_id]:
                ch_m = self.registry.pop(lid)
                self.obj_kinds.pop(lid, None)
                self.obj_axes.pop(lid, None)
                self._obj_boxes.pop(lid, None)
                child_fades.append(m.FadeOut(ch_m))
            if child_fades:
                return m.AnimationGroup(m.FadeOut(mob), *child_fades)
            return m.FadeOut(mob)
        if ev.op == "focus":
            return m.Circumscribe(mob, color=self.theme.accent_highlight)
        if ev.op == "trace":
            if isinstance(mob, m.VMobject):
                return m.Create(mob)
            raise ValueError(f"Unsupported 'trace' op on non-VMobject {ev.target_id!r}.")
        if ev.op == "sequence":
            if isinstance(mob, m.VGroup) and len(mob) > 1:
                return m.LaggedStart(*(m.FadeIn(sub) for sub in mob), lag_ratio=0.16)
            return m.Create(mob) if isinstance(mob, m.VMobject) else m.FadeIn(mob)
        if ev.op == "parallel":
            targets = [mob]
            if ev.source_id and ev.source_id in self.registry:
                targets.append(self.registry[ev.source_id])
            subs = [sub for t_m in targets for sub in (list(t_m) if isinstance(t_m, m.VGroup) and len(t_m) > 1 else [t_m])]
            return m.AnimationGroup(*(m.Create(sub) if isinstance(sub, m.VMobject) else m.FadeIn(sub) for sub in subs))
        if ev.op in ("animate_param", "set_param", "iterate_states", "interpolate_values"):
            tracker = getattr(mob, "_param_tracker", None)
            sw_key = getattr(mob, "_sweep_key", None)
            if sw_key:
                self._swept_models.add(str(sw_key))
            if tracker is not None and ev.op == "set_param" and ev.target_param is not None:
                kfs = getattr(mob, "_sweep_keyframes", None) or (self._model_keyframes.get(str(sw_key)) if sw_key else None)
                if kfs:
                    best_i = min(range(len(kfs)), key=lambda i_k: abs(float(kfs[i_k].param) - float(ev.target_param)))
                    tgt_u = float(kfs[best_i].u if getattr(kfs[best_i], "u", None) is not None else (best_i / max(1, len(kfs) - 1)))
                else:
                    tgt_u = float(ev.target_param)
                return tracker.animate.set_value(tgt_u)
            if tracker is not None and ev.param_range and len(ev.param_range) == 2:
                tracker.set_value(float(ev.param_range[0]))
                return tracker.animate.set_value(float(ev.param_range[1]))
            if tracker is not None and ev.param_values:
                tracker.set_value(float(ev.param_values[0]))
                return m.Succession(*(tracker.animate.set_value(float(v)) for v in ev.param_values[1:]))
            if tracker is not None:
                tracker.set_value(0.0)
                return tracker.animate.set_value(1.0)
            step_states = getattr(mob, "_step_bar_states", None)
            if step_states and isinstance(mob, m.VGroup):
                bars_only = [sub for sub in mob if isinstance(sub, m.Rectangle)]
                if bars_only and len(step_states) >= 2:
                    anims_seq: list[m.Animation] = []
                    base_bottom = min(float(b.get_bottom()[1]) for b in bars_only)
                    max_bh = getattr(mob, "_bar_max_h", 2.0)
                    for st_vals in step_states[1:]:
                        mx_v = max([abs(v) for v in st_vals] + [1e-6])
                        st_anims = []
                        for b_rect, val in zip(bars_only, st_vals):
                            new_h = max(0.03, abs(float(val)) / mx_v * max_bh)
                            new_rect = b_rect.copy().stretch_to_fit_height(new_h)
                            new_rect.move_to(np.array([b_rect.get_center()[0], base_bottom + new_h * 0.5, 0.0]))
                            st_anims.append(m.Transform(b_rect, new_rect))
                        anims_seq.append(m.AnimationGroup(*st_anims))
                    return m.Succession(*anims_seq)
            if self.obj_kinds.get(ev.target_id) == "data_series" and isinstance(mob, m.VMobject):
                return m.Create(mob)
            if isinstance(mob, m.VGroup) and len(mob) >= 1:
                if len(mob) == 1:
                    return m.ShowPassingFlash(mob[0].copy().set_stroke(color=self.theme.accent_highlight, width=5.0), time_width=0.35) if isinstance(mob[0], m.VMobject) else m.Indicate(mob[0], scale_factor=1.06)
                return m.LaggedStart(*(m.Indicate(sub, scale_factor=1.06) for sub in mob), lag_ratio=0.14)
            if isinstance(mob, m.VMobject):
                return m.ShowPassingFlash(mob.copy().set_stroke(color=self.theme.accent_highlight, width=5.0), time_width=0.35)
            raise ValueError(
                f"Unsupported {ev.op} on {ev.target_id!r} ({self.obj_kinds.get(ev.target_id)}): "
                "requires a parameterized object (_param_tracker) or multi-part group."
            )
        return m.FadeIn(mob)

    def _get_geom_frame(self) -> tuple[float, np.ndarray]:
        if self._geom_frame is None:
            L = self.layout
            start = getattr(self, "_cur_scene_idx", 0)
            xs: list[float] = []
            ys: list[float] = []
            for sc in V3_PLAN.scenes[start:]:
                has_unaxed = False
                for o in sc.objects:
                    if o.kind == "polygon" and not o.axes_id and o.vertices:
                        has_unaxed = True
                        for v in o.vertices:
                            xs.append(float(v[0]))
                            ys.append(float(v[1]))
                if has_unaxed and all(not o.persist for o in sc.objects if o.kind == "polygon" and not o.axes_id):
                    break
            if xs and ys:
                cx = 0.5 * (min(xs) + max(xs))
                cy = 0.5 * (min(ys) + max(ys))
                span_x = max(max(xs) - min(xs), 1e-6)
                span_y = max(max(ys) - min(ys), 1e-6)
                scale = min((L.stage_w * 0.78) / span_x, (L.stage_h * 0.72) / span_y)
                self._geom_frame = (scale, np.array([cx, cy, 0.0]))
            else:
                self._geom_frame = (1.0, np.zeros(3))
        return self._geom_frame

    def _get_geom_scale(self) -> float:
        return self._get_geom_frame()[0]

    def _resolve_axes(self, aid: str | None) -> m.Axes | None:
        cand = self.registry.get(aid or "")
        if isinstance(cand, m.Axes):
            return cand
        return getattr(cand, "_wrapped_axes", None)

    def _to_point(self, coords: list[float], axes_id: str | None) -> np.ndarray:
        ax = self._resolve_axes(axes_id)
        if isinstance(ax, m.Axes):
            return ax.c2p(coords[0], coords[1])
        s, center = self._get_geom_frame()
        return self.layout.stage_center + np.array([(coords[0] - center[0]) * s, (coords[1] - center[1]) * s, 0.0])

    def _build_object(self, obj: SceneObject) -> m.Mobject:
        L, T = self.layout, self.theme
        color = getattr(T, ROLE_TO_THEME[obj.color_role])

        if obj.kind == "axes":
            xr = obj.x_range or [-3.5, 3.5]
            yr = obj.y_range or [-1.5, 1.5]
            zones = self._cur_zones
            pbox = fit_axes_pixel_box(
                zones,
                xr=xr,
                yr=yr,
                equal_aspect=obj.equal_aspect,
                composition=self._cur_scene_comp,
                panel_slot=obj.panel_slot,
                has_side_annotations=getattr(self, "_cur_has_side_ann", False),
            )
            cx, cy, x_len, y_len = px_box_to_manim(pbox, zones, L.frame_w, L.frame_h)
            axis_col = T.text_muted if not getattr(T, "is_light", False) else T.text_primary
            has_chrome = bool(getattr(obj, "x_label", "") or getattr(obj, "y_label", "") or getattr(obj, "legend_items", None))
            eq_clearance = 0.72 if has_chrome else (0.82 if zones.equation_box.h > 0 else 0.88)
            x_fit_scale = 0.86 if has_chrome else 0.90
            y_shift = -0.06 if zones.equation_box.h > 0 else 0.0
            dx_ax = max(1e-6, float(xr[1]) - float(xr[0]))
            dy_ax = max(1e-6, float(yr[1]) - float(yr[0]))
            x_step = dx_ax / 4.0
            y_step = dy_ax / 4.0
            ax = m.Axes(
                x_range=[xr[0], xr[1], x_step],
                y_range=[yr[0], yr[1], y_step],
                x_length=x_len * x_fit_scale,
                y_length=y_len * eq_clearance,
                axis_config={
                    "color": axis_col,
                    "stroke_width": 2.2,
                    "include_tip": True,
                    "include_numbers": True,
                    "font_size": px_to_manim_font_size(24 if L.frame_h > L.frame_w else 20, self.profile.pixel_height),
                    "line_to_number_buff": 0.18,
                    "decimal_number_config": {"num_decimal_places": 0 if dx_ax >= 8.0 and dy_ax >= 8.0 else 1},
                },
                x_axis_config={"decimal_number_config": {"num_decimal_places": 0 if dx_ax >= 8.0 else 1}},
                y_axis_config={"decimal_number_config": {"num_decimal_places": 0 if dy_ax >= 8.0 else 1}},
                tips=True,
            ).move_to(np.array([cx, cy + y_shift, 0.0]))
            self.axes_ranges[obj.id] = (xr, yr)
            self._obj_boxes[obj.id] = pbox
            self._scene_obstacles.append(pbox)
            # Attach crisp x/y axis labels & legend cleanly outside tick labels (no intersecting card outline)
            extras: list[m.Mobject] = [ax]
            lbl_fs = px_to_manim_font_size(max(22, self._px_scale.body_px - 2), self.profile.pixel_height)
            min_leg_h = max(L.min_text_h, 0.24)
            if getattr(obj, "x_label", ""):
                xl_mob = m.Text(str(obj.x_label), font_size=lbl_fs, font=CAPTION_FONT, color=axis_col, **_FONT_KWARGS)
                fit_mobject(xl_mob, max_width=x_len * 0.44, max_height=0.34, min_height=min_leg_h)
                xl_mob.next_to(ax.x_axis, m.DOWN, buff=0.22).align_to(ax, m.RIGHT)
                extras.append(xl_mob)
            if getattr(obj, "y_label", ""):
                yl_mob = m.Text(str(obj.y_label), font_size=lbl_fs, font=CAPTION_FONT, color=axis_col, **_FONT_KWARGS)
                fit_mobject(yl_mob, max_width=x_len * 0.38, max_height=0.34, min_height=min_leg_h)
                yl_mob.next_to(ax, m.UP, buff=0.14).align_to(ax, m.LEFT)
                extras.append(yl_mob)
            if getattr(obj, "legend_items", None):
                leg_rows = m.VGroup()
                for li in obj.legend_items[:3]:
                    if not li:
                        continue
                    l_role = li[1] if len(li) > 1 else "primary"
                    l_col = getattr(T, ROLE_TO_THEME.get(l_role, "accent_primary"))
                    swatch = m.Dot(radius=0.06, color=l_col)
                    l_txt = m.Text(str(li[0]), font_size=max(20, lbl_fs - 2), font=CAPTION_FONT, color=T.text_primary, **_FONT_KWARGS)
                    fit_mobject(l_txt, max_width=x_len * 0.46, max_height=0.30, min_height=min_leg_h)
                    row_g = m.VGroup(swatch, l_txt).arrange(m.RIGHT, buff=0.08)
                    leg_rows.add(row_g)
                if len(leg_rows) > 0:
                    leg_rows.arrange(m.DOWN, aligned_edge=m.RIGHT, buff=0.06)
                    leg_rows.next_to(ax, m.UP, buff=0.14).align_to(ax, m.RIGHT)
                    extras.append(leg_rows)
            if len(extras) > 1:
                ax_grp = m.VGroup(*extras)
                ax_grp.move_to(np.array([cx, cy + y_shift, 0.0]))
                ax_grp._wrapped_axes = ax  # type: ignore[attr-defined]
                return ax_grp
            return ax

        if obj.kind == "function_plot":
            if not obj.expr and (obj.points_list or getattr(obj, "sweep_keyframes", None)):
                return self._build_data_series_mob(obj, color)
            ax = self._resolve_axes(obj.axes_id)
            if isinstance(ax, m.Axes) and obj.axes_id in self.axes_ranges:
                xr, yr = self.axes_ranges[obj.axes_id]
            else:
                xr, yr = obj.x_range or [-3.5, 3.5], obj.y_range or [-1.5, 1.5]
                ax = m.Axes(x_range=[xr[0], xr[1], 1.0], y_range=[yr[0], yr[1], 1.0], x_length=L.stage_w * 0.88, y_length=L.stage_h * 0.65).move_to(L.stage_center)
            fn = compile_expr(obj.expr or "0", ("x",))
            return plot_clipped_segments(ax, fn, xr, yr, color=color, stroke_width=3.5)

        if obj.kind == "parametric_curve":
            if not obj.expr and (obj.points_list or getattr(obj, "sweep_keyframes", None)):
                return self._build_data_series_mob(obj, color)
            ax = self._resolve_axes(obj.axes_id)
            if not isinstance(ax, m.Axes):
                xr, yr = obj.x_range or [-2.0, 2.0], obj.y_range or [-2.0, 2.0]
                ax = m.Axes(x_range=[xr[0], xr[1], 1.0], y_range=[yr[0], yr[1], 1.0], x_length=L.stage_w * 0.8, y_length=L.stage_h * 0.6).move_to(L.stage_center)
            p0, p1 = (obj.expr or "cos(t); sin(t)").split(";", 1)
            fx, fy = compile_expr(p0, ("t",)), compile_expr(p1, ("t",))
            tr = obj.t_range or [0.0, float(2 * np.pi)]
            ts = np.linspace(tr[0], tr[1], 240)
            raw_pts = [[float(fx(t=t)), float(fy(t=t))] for t in ts]
            ax_bounds = self.axes_ranges.get(obj.axes_id or "")
            if ax_bounds is not None:
                runs = _clip_polyline_runs(raw_pts, ax_bounds[0], ax_bounds[1])
                if not runs:
                    return m.VGroup()
                if len(runs) == 1:
                    curve = m.VMobject()
                    curve.set_points_smoothly([ax.c2p(p[0], p[1]) for p in runs[0]])
                    curve.set_stroke(color=color, width=3.5)
                    return curve
                grp = m.VGroup()
                for run in runs:
                    sub = m.VMobject()
                    sub.set_points_smoothly([ax.c2p(p[0], p[1]) for p in run])
                    sub.set_stroke(color=color, width=3.5)
                    grp.add(sub)
                return grp
            pts = [ax.c2p(p[0], p[1]) for p in raw_pts]
            curve = m.VMobject()
            curve.set_points_smoothly(pts)
            curve.set_stroke(color=color, width=3.5)
            return curve

        if obj.kind == "equation":
            zones = self._cur_zones
            eq_fs = px_to_manim_font_size(self._px_scale.equation_px, self.profile.pixel_height)
            eq = m.MathTex(obj.tex or "x", font_size=eq_fs, color=color)
            eq_i = getattr(self, "_cur_eq_index", 0)
            eq_tot = max(1, getattr(self, "_cur_eq_total", 1))
            is_port = L.frame_h > L.frame_w
            eb = zones.equation_box
            _, _, eq_band_w, eq_band_h = px_box_to_manim(eb, zones, L.frame_w, L.frame_h)
            cx_eq, cy_eq, _, _ = px_box_to_manim(eb, zones, L.frame_w, L.frame_h)
            eq_center = np.array([cx_eq, cy_eq, 0.0])
            max_w_ratio = 0.78 if obj.card else 0.80
            max_h_ratio = 0.86
            if eq_tot >= 2:
                if not is_port:
                    slot_w = eq_band_w / eq_tot
                    eq_center[0] = (eq_i - 0.5 * (eq_tot - 1)) * slot_w
                    max_w_ratio = min(0.88, (slot_w * 0.86) / max(eq_band_w, 1e-6))
                else:
                    slot_h = eq_band_h / eq_tot
                    eq_center[1] = cy_eq + (0.5 * (eq_tot - 1) - eq_i) * slot_h
                    max_h_ratio = min(0.90, (slot_h * 0.82) / max(eq_band_h, 1e-6))
            pad_x = 0.52 if obj.card else 0.0
            pad_y = 0.36 if obj.card else 0.0
            fit_h = max(0.30, (eq_band_h * max_h_ratio if obj.card else max(0.45, eq_band_h * max_h_ratio)) - pad_y)
            fit_mobject(eq, max_width=max(0.6, eq_band_w * max_w_ratio - pad_x), max_height=fit_h)
            eq.move_to(eq_center)
            if obj.card:
                # Subtle hairline card only when explicitly opted in via obj.card=True; always wrap cleanly around fitted eq
                pill = m.RoundedRectangle(
                    corner_radius=0.14,
                    width=eq.width + pad_x,
                    height=eq.height + pad_y,
                    fill_color=T.surface,
                    fill_opacity=0.88,
                    stroke_color=color if getattr(T, "is_light", False) else T.surface_border,
                    stroke_opacity=0.65,
                    stroke_width=1.4,
                ).move_to(eq_center)
                out_eq: m.Mobject = m.VGroup(pill, eq)
            else:
                out_eq = eq
            pbox = self._manim_to_px_box(out_eq, role="equation")
            eq_obs = [b for b in self._scene_obstacles if b.role == "equation"]
            placed_box = resolve_non_overlapping_boxes([pbox], eq_obs, zones, gap_px=10.0)[0]
            px_cx, px_cy, _, _ = px_box_to_manim(placed_box, zones, L.frame_w, L.frame_h)
            out_eq.move_to(np.array([px_cx, px_cy, 0.0]))
            self._obj_boxes[obj.id] = placed_box
            self._scene_obstacles.append(placed_box)
            return out_eq

        if obj.kind == "annotation":
            zones = self._cur_zones
            kfs_ann = list(getattr(obj, "sweep_keyframes", []) or [])
            longest_kf_lbl = max((kf.label for kf in kfs_ann if kf.label), key=len, default="") if kfs_ann else ""
            raw_lbl = (obj.text or obj.label or longest_kf_lbl or "").strip()
            lbl_fs = px_to_manim_font_size(self._px_scale.math_label_px, self.profile.pixel_height)
            has_tex = any(tok in raw_lbl for tok in ("\\", "_", "^"))
            if has_tex:
                try:
                    txt: m.Mobject = m.MathTex(raw_lbl, font_size=lbl_fs, color=color)
                except Exception:
                    txt = m.Text(raw_lbl, font_size=lbl_fs - 4, font=CAPTION_FONT, weight="SEMIBOLD", color=color, **_FONT_KWARGS)
            else:
                txt = m.Text(raw_lbl, font_size=lbl_fs - 4, font=CAPTION_FONT, weight="SEMIBOLD", color=color, **_FONT_KWARGS)
            is_port = L.frame_h > L.frame_w
            fit_mobject(txt, max_width=L.safe_w * (0.84 if is_port else 0.34), max_height=0.64)
            ax_mob = self._resolve_axes(obj.axes_id) if obj.axes_id else None
            if ax_mob is None and not obj.coords:
                ax_mob = next((self._resolve_axes(k) for k in self.registry if self._resolve_axes(k) is not None), None)
            ann_i = getattr(self, "_cur_ann_index", 0)
            ann_tot = max(1, getattr(self, "_cur_ann_total", 1))
            if obj.coords and len(obj.coords) >= 2 and not is_port and not obj.axes_id:
                pos = np.array([
                    max(-L.safe_w * 0.45 + txt.width * 0.5, min(L.safe_w * 0.45 - txt.width * 0.5, float(obj.coords[0]))),
                    max(-L.stage_h * 0.45 + txt.height * 0.5, min(L.stage_h * 0.45 - txt.height * 0.5, float(obj.coords[1]))),
                    0.0,
                ])
            elif isinstance(ax_mob, m.Axes) and not is_port:
                fit_mobject(txt, max_width=L.safe_w * 0.32, max_height=0.58)
                right_x = 0.5 * (ax_mob.get_right()[0] + L.safe_w * 0.5)
                left_x = 0.5 * (-L.safe_w * 0.5 + ax_mob.get_left()[0])
                if ann_tot >= 2 and (ann_i % 2 == 1) and (left_x - txt.width * 0.5 >= -L.safe_w * 0.49):
                    pref_x = left_x
                elif right_x + txt.width * 0.5 <= L.safe_w * 0.49:
                    pref_x = right_x
                else:
                    pref_x = min(L.safe_w * 0.48 - txt.width * 0.5, L.safe_w * 0.30)
                pref_y = ax_mob.get_top()[1] - txt.height * 0.6 - (ann_i // (2 if ann_tot >= 2 else 1)) * 0.68
                pos = np.array([pref_x, pref_y, 0.0])
            elif isinstance(ax_mob, m.Axes) and is_port:
                fit_mobject(txt, max_width=L.safe_w * 0.84, max_height=0.56)
                if ann_i % 2 == 0:
                    pos = np.array([0.0, ax_mob.get_top()[1] + txt.height * 0.55 + 0.24 + (ann_i // 2) * 0.58, 0.0])
                else:
                    pos = np.array([0.0, ax_mob.get_bottom()[1] - txt.height * 0.55 - 0.24 - (ann_i // 2) * 0.58, 0.0])
            elif obj.coords and len(obj.coords) >= 2:
                pos = self._to_point(obj.coords, obj.axes_id)
            else:
                y_shift = (0.5 * (ann_tot - 1) - ann_i) * 0.72
                pos = L.stage_center + np.array([0.0, L.stage_h * 0.18 + y_shift, 0.0])
            txt.move_to(pos)
            raw_box = self._manim_to_px_box(txt, role="annotation")
            placed_box = resolve_non_overlapping_boxes([raw_box], self._scene_obstacles, zones, gap_px=12.0)[0]
            px_cx, px_cy, _, _ = px_box_to_manim(placed_box, zones, L.frame_w, L.frame_h)
            txt.move_to(np.array([px_cx, px_cy, 0.0]))
            self._obj_boxes[obj.id] = placed_box
            self._scene_obstacles.append(placed_box)
            if getattr(obj, "sweep_keyframes", None):
                sw_key = f"__sweep_{obj.model_id or obj.id}"
                init_u = float(getattr(obj, "param_init", 0.0) or 0.0)
                tracker = self.params.setdefault(sw_key, m.ValueTracker(init_u))
                self._model_selected_u.setdefault(sw_key, init_u)
                kfs = list(obj.sweep_keyframes)
                self._model_keyframes.setdefault(sw_key, kfs)
                ctr_pos = np.array([px_cx, px_cy, 0.0])
                col_ann = color
                fs_ann = max(18, lbl_fs - 4)
                def _mk_sw_ann(t=tracker, frames=kfs, pos_c=ctr_pos, c=col_ann, fs=fs_ann):
                    u = max(0.0, min(1.0, float(t.get_value())))
                    idx = min(len(frames) - 1, int(round(u * max(1, len(frames) - 1))))
                    lbl = frames[idx].label or raw_lbl or ""
                    mob_a = m.Text(lbl, font_size=fs, font=CAPTION_FONT, weight="SEMIBOLD", color=c, **_FONT_KWARGS)
                    fit_mobject(mob_a, max_width=L.safe_w * (0.84 if is_port else 0.34), max_height=0.58)
                    return mob_a.move_to(pos_c)
                dyn_ann = m.always_redraw(_mk_sw_ann)
                dyn_ann._param_tracker = tracker  # type: ignore[attr-defined]
                dyn_ann._sweep_key = sw_key  # type: ignore[attr-defined]
                dyn_ann._sweep_keyframes = kfs  # type: ignore[attr-defined]
                return dyn_ann
            return txt

        if obj.kind == "point":
            if getattr(obj, "sweep_keyframes", None):
                sw_key = f"__sweep_{obj.model_id or obj.id}"
                init_u = float(getattr(obj, "param_init", 0.0) or 0.0)
                tracker = self.params.setdefault(sw_key, m.ValueTracker(init_u))
                self._model_selected_u.setdefault(sw_key, init_u)
                kfs = list(obj.sweep_keyframes)
                self._model_keyframes.setdefault(sw_key, kfs)
                ax_ref = self.registry.get(obj.axes_id or "")
                ax = getattr(ax_ref, "_wrapped_axes", ax_ref)
                is_spec_pk = obj.model_output in ("spectrum_peak", "time_point")
                def _mk_sw_dot(t=tracker, frames=kfs, a=ax, c=color, spec_mode=is_spec_pk):
                    u = max(0.0, min(1.0, float(t.get_value())))
                    idx = min(len(frames) - 1, int(round(u * max(1, len(frames) - 1))))
                    kf = frames[idx]
                    xy = kf.spec_point if (spec_mode and kf.spec_point) else (kf.point or [0.0, 0.0])
                    pos = a.c2p(xy[0], xy[1]) if isinstance(a, m.Axes) else (L.stage_center + np.array([xy[0], xy[1], 0.0]))
                    return m.Dot(pos, radius=0.11, color=c)
                mob = m.always_redraw(_mk_sw_dot)
                mob._param_tracker = tracker  # type: ignore[attr-defined]
                mob._sweep_key = sw_key  # type: ignore[attr-defined]
                mob._sweep_keyframes = kfs  # type: ignore[attr-defined]
                return mob
            if obj.param_name and obj.expr and ";" in obj.expr:
                tracker = self.params.setdefault(obj.param_name, m.ValueTracker(obj.param_init))
                p0, p1 = obj.expr.split(";", 1)
                fx, fy = compile_expr(p0, (obj.param_name,)), compile_expr(p1, (obj.param_name,))
                ax = self._resolve_axes(obj.axes_id)
                def _mk_dot(t=tracker, f1=fx, f2=fy, a=ax, c=color, pn=obj.param_name):
                    v = t.get_value()
                    x, y = float(f1(**{pn: v})), float(f2(**{pn: v}))
                    pos = a.c2p(x, y) if isinstance(a, m.Axes) else (L.stage_center + np.array([x, y, 0.0]))
                    return m.Dot(pos, radius=0.10, color=c)
                mob = m.always_redraw(_mk_dot)
                mob._param_tracker = tracker  # type: ignore[attr-defined]
                return mob
            return m.Dot(self._to_point(obj.coords or [0.0, 0.0], obj.axes_id), radius=0.09, color=color)

        if obj.kind in ("line", "vector"):
            if obj.kind == "vector" and getattr(obj, "sweep_keyframes", None):
                sw_key = f"__sweep_{obj.model_id or obj.id}"
                init_u = float(getattr(obj, "param_init", 0.0) or 0.0)
                tracker = self.params.setdefault(sw_key, m.ValueTracker(init_u))
                self._model_selected_u.setdefault(sw_key, init_u)
                kfs = list(obj.sweep_keyframes)
                self._model_keyframes.setdefault(sw_key, kfs)
                ax_ref = self.registry.get(obj.axes_id or "")
                ax = getattr(ax_ref, "_wrapped_axes", ax_ref)
                c0 = obj.coords or [0.0, 0.0]
                def _mk_sw_vec(t=tracker, frames=kfs, a=ax, c=color, orig=c0):
                    u = max(0.0, min(1.0, float(t.get_value())))
                    idx = min(len(frames) - 1, int(round(u * max(1, len(frames) - 1))))
                    kf = frames[idx]
                    st_xy = getattr(kf, "vector_start", None) or orig
                    end_xy = kf.vector_end or kf.point or [0.01, 0.0]
                    s = a.c2p(st_xy[0], st_xy[1]) if isinstance(a, m.Axes) else (L.stage_center + np.array([st_xy[0], st_xy[1], 0.0]))
                    e = a.c2p(end_xy[0], end_xy[1]) if isinstance(a, m.Axes) else (L.stage_center + np.array([end_xy[0], end_xy[1], 0.0]))
                    if np.linalg.norm(e - s) < 0.04:
                        return m.Dot(s, radius=0.05, color=c)
                    return m.Arrow(s, e, color=c, buff=0)
                mob = m.always_redraw(_mk_sw_vec)
                mob._param_tracker = tracker  # type: ignore[attr-defined]
                mob._sweep_key = sw_key  # type: ignore[attr-defined]
                mob._sweep_keyframes = kfs  # type: ignore[attr-defined]
                return mob
            if obj.kind == "vector" and obj.param_name and obj.expr and ";" in obj.expr:
                tracker = self.params.setdefault(obj.param_name, m.ValueTracker(obj.param_init))
                p0, p1 = obj.expr.split(";", 1)
                fx, fy = compile_expr(p0, (obj.param_name,)), compile_expr(p1, (obj.param_name,))
                ax = self._resolve_axes(obj.axes_id)
                c0 = obj.coords or [0.0, 0.0]
                def _mk_vec(t=tracker, f1=fx, f2=fy, a=ax, c=color, pn=obj.param_name, orig=c0):
                    v = t.get_value()
                    x, y = float(f1(**{pn: v})), float(f2(**{pn: v}))
                    s = a.c2p(orig[0], orig[1]) if isinstance(a, m.Axes) else (L.stage_center + np.array([orig[0], orig[1], 0.0]))
                    e = a.c2p(x, y) if isinstance(a, m.Axes) else (L.stage_center + np.array([x, y, 0.0]))
                    return m.Arrow(s, e, color=c, buff=0)
                mob = m.always_redraw(_mk_vec)
                mob._param_tracker = tracker  # type: ignore[attr-defined]
                return mob
            p0 = self._to_point(obj.coords or [0.0, 0.0], obj.axes_id)
            p1 = self._to_point(obj.end_coords or [1.0, 1.0], obj.axes_id)
            return m.Arrow(p0, p1, color=color, buff=0) if obj.kind == "vector" else m.Line(p0, p1, color=color, stroke_width=3.0)

        if obj.kind == "polygon":
            pts = [self._to_point(v, obj.axes_id) for v in obj.vertices]
            poly = m.Polygon(*pts, color=color, fill_color=color, fill_opacity=0.28, stroke_width=3.2)
            poly._uses_axes = bool(obj.axes_id)  # type: ignore[attr-defined]
            if not obj.axes_id and obj.panel_slot in ("left", "right", "top", "bottom"):
                fit_mobject(poly, max_width=L.stage_w * 0.40, max_height=L.stage_h * 0.54)
                poly.move_to(self._slot_center(obj, getattr(self, "_cur_scene_comp", "auto")))
                pbox = self._manim_to_px_box(poly, role="polygon")
                placed_box = resolve_non_overlapping_boxes([pbox], self._scene_obstacles, self._cur_zones, gap_px=18.0)[0]
                px_cx, px_cy, _, _ = px_box_to_manim(placed_box, self._cur_zones, L.frame_w, L.frame_h)
                poly.move_to(np.array([px_cx, px_cy, 0.0]))
                self._obj_boxes[obj.id] = placed_box
                self._scene_obstacles.append(placed_box)
            return poly

        if obj.kind == "circle":
            pos = self._to_point(obj.coords or [0.0, 0.0], obj.axes_id)
            r = obj.radius or 1.0
            ax = self._resolve_axes(obj.axes_id)
            if isinstance(ax, m.Axes):
                r = float(np.linalg.norm(ax.c2p(r, 0.0) - ax.c2p(0.0, 0.0)))
            else:
                r = r * self._get_geom_scale()
            c_mob = m.Circle(radius=r, color=color, stroke_width=3.2).move_to(pos)
            c_mob._uses_axes = bool(obj.axes_id)  # type: ignore[attr-defined]
            return c_mob

        if obj.kind == "number_line":
            xr = obj.x_range or [-3.0, 3.0]
            step = max(1.0, round((xr[1] - xr[0]) / 6.0, 2))
            nl = m.NumberLine(
                x_range=[xr[0], xr[1], step],
                length=L.stage_w * 0.85,
                color=color,
                include_numbers=True,
                label_direction=m.DOWN,
                line_to_number_buff=0.35,
                font_size=22,
            ).move_to(L.stage_center + np.array([0.0, -L.stage_h * 0.25, 0.0]))
            return nl

        if obj.kind == "bar_chart" and obj.values:
            cur_ids = getattr(self, "_cur_scene_obj_ids", set())
            if not obj.axes_id and obj.panel_slot in ("auto", "center"):
                for k_rem, v in list(self.registry.items()):
                    if k_rem in cur_ids:
                        continue
                    if isinstance(v, m.Axes) or hasattr(v, "_wrapped_axes") or getattr(v, "_is_bar_chart", False) or self.obj_kinds.get(k_rem) in ("axes", "data_series", "point", "vector", "function_plot", "parametric_curve"):
                        self.remove(v)
                        self.registry.pop(k_rem, None)
            vals = obj.values
            if obj.log_scale:
                logs = [float(np.log10(max(v, 1e-12))) for v in vals]
                lo, hi = min(logs) - 0.5, max(logs)
                norm = [(v - lo) / max(hi - lo, 1e-6) for v in logs]
            else:
                hi = max(abs(v) for v in vals) or 1.0
                norm = [v / hi for v in vals]
            grp = m.VGroup()
            n = len(norm)
            is_split = obj.panel_slot in ("left", "right")
            chart_w = L.stage_w * (0.42 if is_split else 0.78)
            w = chart_w / max(1, n)
            base_y = -L.stage_h * 0.14
            max_h = L.stage_h * (0.34 if obj.panel_slot in ("top", "bottom") else 0.40)
            lbl_step = max(1, (n + 7) // 8)
            for i, frac in enumerate(norm):
                h = max(0.0, abs(frac) * max_h)
                bar_w = min(w * 0.58, 1.35)
                if h > 0.0:
                    bar = m.Rectangle(width=bar_w, height=h, fill_color=color, fill_opacity=0.88, stroke_width=0)
                    x_pos = -0.5 * chart_w + (i + 0.5) * w
                    y_off = h * 0.5 if frac >= 0 else -h * 0.5
                    bar.move_to(np.array([x_pos, base_y + y_off, 0.0]))
                    grp.add(bar)
                if i < len(obj.bar_labels) and obj.bar_labels[i] and (i % lbl_step == 0 or (i == n - 1 and (n - 1) % lbl_step >= max(2, lbl_step // 2))):
                    from studio.renderers.hyperframes.typography import _pretty_math_label

                    lbl_str = _pretty_math_label(obj.bar_labels[i]) if "\\" in obj.bar_labels[i] or "_" in obj.bar_labels[i] else obj.bar_labels[i]
                    lbl = m.Text(lbl_str, font_size=26, font=CAPTION_FONT, color=T.text_muted, **_FONT_KWARGS)
                    fit_mobject(lbl, max_width=w * lbl_step * 0.85, min_height=L.min_text_h)
                    x_pos = -0.5 * chart_w + (i + 0.5) * w
                    lbl.move_to(np.array([x_pos, base_y - 0.34, 0.0]))
                    grp.add(lbl)
            grp.move_to(self._slot_center(obj, getattr(self, "_cur_scene_comp", "auto")))
            pbox = self._manim_to_px_box(grp, role="bar_chart")
            placed_box = resolve_non_overlapping_boxes([pbox], self._scene_obstacles, self._cur_zones, gap_px=18.0)[0]
            px_cx, px_cy, _, _ = px_box_to_manim(placed_box, self._cur_zones, L.frame_w, L.frame_h)
            grp.move_to(np.array([px_cx, px_cy, 0.0]))
            self._obj_boxes[obj.id] = placed_box
            self._scene_obstacles.append(placed_box)
            grp._is_bar_chart = True  # type: ignore[attr-defined]
            grp._bar_max_h = max_h  # type: ignore[attr-defined]
            if obj.points_list and len(obj.points_list) >= 2:
                grp._step_bar_states = obj.points_list  # type: ignore[attr-defined]
            if getattr(obj, "sweep_keyframes", None) and any(getattr(kf, "bar_values", None) for kf in obj.sweep_keyframes):
                sw_kfs = [kf for kf in obj.sweep_keyframes if getattr(kf, "bar_values", None)]
                sw_key = f"__sweep_{obj.model_id or obj.id}"
                init_u = float(getattr(obj, "param_init", 0.0) or 0.0)
                tracker = self.params.setdefault(sw_key, m.ValueTracker(init_u))
                self._model_selected_u.setdefault(sw_key, init_u)
                self._model_keyframes.setdefault(sw_key, sw_kfs)
                hi_sw = max([abs(float(v)) for kf in sw_kfs for v in (kf.bar_values or [])] + [hi, 1e-6])
                ctr_b = grp.get_center().copy()
                base_y_pos = float(ctr_b[1] - max_h * 0.35)
                def _mk_sw_bars(t=tracker, frames=sw_kfs, b_ctr=ctr_b, cw=chart_w, bw=w, mh=max_h, by=base_y_pos, mx_v=hi_sw, col_b=color, lbls=obj.bar_labels):
                    u = max(0.0, min(1.0, float(t.get_value())))
                    idx = min(len(frames) - 1, int(round(u * max(1, len(frames) - 1))))
                    vals_k = frames[idx].bar_values or vals
                    g_b = m.VGroup()
                    nn = len(vals_k)
                    w_i = cw / max(1, nn)
                    for i_b, vv in enumerate(vals_k):
                        hh = max(0.03, abs(float(vv)) / mx_v * mh)
                        rb = m.Rectangle(width=w_i * 0.62, height=hh, fill_color=col_b, fill_opacity=0.88, stroke_width=0)
                        xp = b_ctr[0] - 0.5 * cw + (i_b + 0.5) * w_i
                        rb.move_to(np.array([xp, by + hh * 0.5, 0.0]))
                        g_b.add(rb)
                        if i_b < len(lbls) and lbls[i_b]:
                            tl = m.Text(str(lbls[i_b]), font_size=24, font=CAPTION_FONT, color=T.text_muted, **_FONT_KWARGS)
                            fit_mobject(tl, max_width=w_i * 0.9, min_height=L.min_text_h)
                            tl.move_to(np.array([xp, by - 0.30, 0.0]))
                            g_b.add(tl)
                    return g_b
                dyn_b = m.always_redraw(_mk_sw_bars)
                dyn_b._param_tracker = tracker  # type: ignore[attr-defined]
                dyn_b._sweep_key = sw_key  # type: ignore[attr-defined]
                dyn_b._sweep_keyframes = sw_kfs  # type: ignore[attr-defined]
                return dyn_b
            return grp

        if obj.kind == "angle_arc":
            tracker = self.params.setdefault(obj.param_name or "theta", m.ValueTracker(obj.param_init))
            ax = self._resolve_axes(obj.axes_id)
            r_math = obj.radius or 1.0
            r_scene = float(np.linalg.norm(ax.c2p(r_math, 0.0) - ax.c2p(0.0, 0.0))) if isinstance(ax, m.Axes) else r_math
            center = ax.c2p(0.0, 0.0) if isinstance(ax, m.Axes) else L.stage_center
            def _mk_arc(t=tracker, rad=r_scene, ctr=center, c=color):
                ang = float(t.get_value())
                if abs(ang) < 1e-3:
                    return m.Dot(ctr + np.array([rad, 0.0, 0.0]), radius=0.01, color=c)
                return m.Arc(radius=rad, start_angle=0.0, angle=ang, arc_center=ctr, color=c, stroke_width=4.5)
            mob = m.always_redraw(_mk_arc)
            mob._param_tracker = tracker  # type: ignore[attr-defined]
            return mob

        if obj.kind == "projection":
            tracker = self.params.setdefault(obj.param_name or "theta", m.ValueTracker(obj.param_init))
            ax = self._resolve_axes(obj.axes_id)
            p0, p1 = (obj.expr or "cos(theta); sin(theta)").split(";", 1) if (obj.expr and ";" in obj.expr) else ("cos(theta)", "sin(theta)")
            pn = obj.param_name or "theta"
            fx, fy = compile_expr(p0, (pn,)), compile_expr(p1, (pn,))
            def _mk_proj(t=tracker, a=ax, c=color, f1=fx, f2=fy, p_name=pn):
                ang = float(t.get_value())
                cx, cy = float(f1(**{p_name: ang})), float(f2(**{p_name: ang}))
                if not isinstance(a, m.Axes):
                    return m.VGroup()
                p_pt = a.c2p(cx, cy)
                p_x = a.c2p(cx, 0.0)
                return m.VGroup(
                    m.DashedLine(p_pt, p_x, color=c, stroke_width=2.2),
                    m.DashedLine(p_pt, a.c2p(0.0, cy), color=T.accent_tertiary, stroke_width=2.2),
                )
            mob = m.always_redraw(_mk_proj)
            mob._param_tracker = tracker  # type: ignore[attr-defined]
            return mob

        if obj.kind == "matrix":
            rows = obj.matrix_rows or [["a", "b"], ["c", "d"]]
            str_rows = [[str(cell) for cell in r] for r in rows]

            is_port = L.frame_h > L.frame_w
            def _mk_cell(s: str, c=color, p_mode=is_port) -> m.Mobject:
                txt = str(s)
                fs = 36 if p_mode else 32
                if any(tok in txt for tok in ("\\", "_", "^")):
                    try:
                        return m.MathTex(txt, font_size=fs, color=c)
                    except Exception:
                        pass
                return m.Text(txt, font_size=fs, font=MATH_SERIF_FONT, color=c, **_FONT_KWARGS)

            max_cell_chars = max((len(c) for r in str_rows for c in r), default=3)
            h_gap = max(1.45, max_cell_chars * 0.28 + 0.45)
            mat_mob = m.Matrix(
                str_rows,
                v_buff=0.84,
                h_buff=h_gap,
                bracket_h_buff=0.32,
                bracket_v_buff=0.28,
                element_to_mobject=_mk_cell,
                bracket_config={"color": T.text_primary},
            )
            brs = mat_mob.get_brackets()
            brs.set_color(T.text_primary)
            brs.set_stroke(color=T.text_primary, width=2.0)
            max_w = L.stage_w * (0.44 if obj.panel_slot in ("left", "right") else (0.76 if is_port else 0.56))
            max_h = L.stage_h * (0.34 if (is_port and obj.panel_slot in ("left", "right", "top", "bottom")) else (0.44 if obj.panel_slot in ("top", "bottom") else 0.56))
            fit_mobject(mat_mob, max_width=max_w, max_height=max_h)
            mat_mob.move_to(self._slot_center(obj, getattr(self, "_cur_scene_comp", "auto")))
            raw_box = self._manim_to_px_box(mat_mob, role="matrix")
            placed_box = resolve_non_overlapping_boxes([raw_box], self._scene_obstacles, self._cur_zones, gap_px=20.0)[0]
            px_cx, px_cy, _, _ = px_box_to_manim(placed_box, self._cur_zones, L.frame_w, L.frame_h)
            mat_mob.move_to(np.array([px_cx, px_cy, 0.0]))
            self._obj_boxes[obj.id] = placed_box
            self._scene_obstacles.append(placed_box)
            return mat_mob

        if obj.kind == "table":
            all_rows = ([obj.table_headers] if obj.table_headers else []) + list(obj.table_rows)
            n_r = max(1, len(all_rows))
            n_c = max(1, max((len(r) for r in all_rows), default=1))
            ctr = self._slot_center(obj, getattr(self, "_cur_scene_comp", "auto"))

            grid_cells: list[list[m.Mobject]] = []
            col_widths = [0.6] * n_c
            row_heights = [0.35] * n_r
            for ri, row in enumerate(all_rows):
                is_hdr = bool(obj.table_headers and ri == 0)
                c_mobs: list[m.Mobject] = []
                for ci in range(n_c):
                    val = str(row[ci]) if ci < len(row) else ""
                    if any(tok in val for tok in ("\\", "_", "^")):
                        try:
                            mob: m.Mobject = m.MathTex(val, font_size=26 if is_hdr else 24, color=color if is_hdr else T.text_primary)
                        except Exception:
                            mob = m.Text(val, font_size=24 if is_hdr else 22, font=CAPTION_FONT, weight="BOLD" if is_hdr else "NORMAL", color=color if is_hdr else T.text_primary, **_FONT_KWARGS)
                    else:
                        mob = m.Text(val, font_size=24 if is_hdr else 22, font=CAPTION_FONT, weight="BOLD" if is_hdr else "NORMAL", color=color if is_hdr else T.text_primary, **_FONT_KWARGS)
                    col_widths[ci] = max(col_widths[ci], mob.width + 0.48)
                    row_heights[ri] = max(row_heights[ri], mob.height + 0.28)
                    c_mobs.append(mob)
                grid_cells.append(c_mobs)

            total_w = sum(col_widths)
            total_h = sum(row_heights)
            is_port_tbl = L.frame_h > L.frame_w
            max_w = L.stage_w * (0.46 if (not is_port_tbl and obj.panel_slot in ("left", "right")) else 0.88)
            max_h = L.stage_h * (0.36 if (is_port_tbl and obj.panel_slot in ("left", "right", "top", "bottom")) else (0.44 if obj.panel_slot in ("top", "bottom") else 0.72))
            s_fit = min(1.0, max_w / max(total_w, 1e-6), max_h / max(total_h, 1e-6))

            grp = m.VGroup()
            x_cur = -0.5 * total_w
            col_centers = []
            for cw in col_widths:
                col_centers.append(x_cur + 0.5 * cw)
                x_cur += cw
            y_cur = 0.5 * total_h
            row_centers = []
            for rh in row_heights:
                row_centers.append(y_cur - 0.5 * rh)
                y_cur -= rh

            for ri, c_mobs in enumerate(grid_cells):
                for ci, mob in enumerate(c_mobs):
                    mob.move_to(np.array([col_centers[ci], row_centers[ri], 0.0]))
                    grp.add(mob)
            if obj.table_headers and n_r > 1:
                y_div = row_centers[0] - 0.5 * row_heights[0]
                grp.add(m.Line(np.array([-0.5 * total_w, y_div, 0.0]), np.array([0.5 * total_w, y_div, 0.0]), color=T.surface_border, stroke_width=1.6))
            border = m.RoundedRectangle(corner_radius=0.12, width=total_w + 0.18, height=total_h + 0.18, stroke_color=T.surface_border, stroke_width=1.8, fill_opacity=0.0)
            tbl_grp = m.VGroup(border, grp).scale(s_fit).move_to(ctr)
            pbox = self._manim_to_px_box(tbl_grp, role="table")
            placed_box = resolve_non_overlapping_boxes([pbox], self._scene_obstacles, self._cur_zones, gap_px=20.0)[0]
            px_cx, px_cy, _, _ = px_box_to_manim(placed_box, self._cur_zones, L.frame_w, L.frame_h)
            tbl_grp.move_to(np.array([px_cx, px_cy, 0.0]))
            self._obj_boxes[obj.id] = placed_box
            self._scene_obstacles.append(placed_box)
            return tbl_grp

        if obj.kind in ("data_series", "waveform"):
            return self._build_data_series_mob(obj, color)

        if obj.kind == "particle_system":
            grp = m.VGroup()
            for pt in (obj.points_list or [[0.0, 0.0]]):
                grp.add(m.Dot(self._to_point(pt[:2], obj.axes_id), radius=0.055, color=color, fill_opacity=0.85))
            return grp

        if obj.kind == "graph_network":
            grp = m.VGroup()
            node_pts = [self._to_point(p[:2], obj.axes_id) for p in obj.points_list]
            for i, pos in enumerate(node_pts):
                circ = m.Circle(radius=0.28, color=color, fill_color=T.surface, fill_opacity=0.92, stroke_width=2.6).move_to(pos)
                grp.add(circ)
                lbl_str = obj.node_labels[i] if i < len(obj.node_labels) else f"S{i}"
                lbl = m.Text(lbl_str, font_size=22, font=CAPTION_FONT, color=T.text_primary, **_FONT_KWARGS)
                fit_mobject(lbl, max_width=0.44, max_height=0.34)
                lbl.move_to(pos)
                grp.add(lbl)
            for edge in obj.edge_pairs:
                if len(edge) >= 2:
                    try:
                        s_i, d_i = int(edge[0]), int(edge[1])
                        if 0 <= s_i < len(node_pts) and 0 <= d_i < len(node_pts) and s_i != d_i:
                            arr = m.Arrow(node_pts[s_i], node_pts[d_i], buff=0.30, color=T.accent_secondary, stroke_width=2.4, tip_length=0.16)
                            grp.add(arr)
                    except ValueError:
                        pass
            return grp

        if obj.kind == "probability_region":
            pts = [self._to_point(v[:2], obj.axes_id) for v in (obj.vertices or [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
            return m.Polygon(*pts, color=color, fill_color=color, fill_opacity=0.35, stroke_width=2.8)

        if obj.kind == "vector_field":
            from studio.ai.simulation import sample_vector_field_grid as _sample_vf

            ax_ref = self.registry.get(obj.axes_id or "")
            ax = getattr(ax_ref, "_wrapped_axes", ax_ref)
            xr, yr = self.axes_ranges.get(obj.axes_id or "", (obj.x_range or [-3.0, 3.0], obj.y_range or [-3.0, 3.0]))
            segs = _sample_vf(
                (obj.dx_expr or obj.expr or "y").strip(),
                (obj.dy_expr or "-x - 0.25*y").strip(),
                xr,
                yr,
                var_names=getattr(obj, "var_names", None),
            )
            grp = m.VGroup()
            for p0_d, p1_d in segs:
                sx_p = ax.c2p(p0_d[0], p0_d[1]) if isinstance(ax, m.Axes) else self._to_point(p0_d, obj.axes_id)
                ex_p = ax.c2p(p1_d[0], p1_d[1]) if isinstance(ax, m.Axes) else self._to_point(p1_d, obj.axes_id)
                if np.linalg.norm(ex_p - sx_p) >= 0.04:
                    grp.add(m.Arrow(sx_p, ex_p, buff=0, color=T.text_muted if not getattr(T, "is_light", False) else color, stroke_width=1.8, tip_length=0.10))
            return grp

        if obj.kind == "mechanical_body":
            ax_ref = self.registry.get(obj.axes_id or "")
            ax = getattr(ax_ref, "_wrapped_axes", ax_ref)
            slot_ctr = self._slot_center(obj, getattr(self, "_cur_scene_comp", "auto"))
            xr_b, yr_b = self.axes_ranges.get(obj.axes_id or "", (obj.x_range or [-2.6, 2.6], obj.y_range or [-1.6, 1.6]))
            conn = getattr(obj, "connector_kind", "spring") or "spring"
            anc_raw = obj.anchor_coords or ([-2.3, 0.0] if conn != "rod" else [0.0, 1.3])
            pos_raw = obj.coords or [0.0, 0.0]
            sw_key = f"__sweep_{obj.model_id or obj.id}"
            kfs = list(obj.sweep_keyframes) if getattr(obj, "sweep_keyframes", None) else []
            init_u = float(getattr(obj, "param_init", 0.0) or 0.0)
            tracker = self.params.setdefault(sw_key, m.ValueTracker(init_u))
            if kfs:
                self._model_selected_u.setdefault(sw_key, init_u)
                self._model_keyframes.setdefault(sw_key, kfs)

            s_cx, s_cy = float(L.stage_center[0]), float(L.stage_center[1])
            s_left, s_right = s_cx - 0.5 * L.stage_w + 0.35, s_cx + 0.5 * L.stage_w - 0.35
            s_bot, s_top = s_cy - 0.5 * L.stage_h + 0.35, s_cy + 0.5 * L.stage_h - 0.35

            def _map_coord(xy: list[float]) -> np.ndarray:
                if isinstance(ax, m.Axes):
                    p_raw = ax.c2p(float(xy[0]), float(xy[1]))
                    return np.array([
                        max(s_left, min(s_right, float(p_raw[0]))),
                        max(s_bot, min(s_top, float(p_raw[1]))),
                        0.0,
                    ])
                span_x = max(1e-6, float(xr_b[1]) - float(xr_b[0]))
                span_y = max(1e-6, float(yr_b[1]) - float(yr_b[0]))
                w_sc = L.stage_w * (0.42 if obj.panel_slot in ("left", "right") else 0.72)
                h_sc = L.stage_h * (0.38 if obj.panel_slot in ("top", "bottom") else 0.56)
                nx = (float(xy[0]) - 0.5 * (float(xr_b[0]) + float(xr_b[1]))) / span_x * w_sc
                ny = (float(xy[1]) - 0.5 * (float(yr_b[0]) + float(yr_b[1]))) / span_y * h_sc
                p_raw = slot_ctr + np.array([nx, ny, 0.0])
                return np.array([
                    max(s_left, min(s_right, float(p_raw[0]))),
                    max(s_bot, min(s_top, float(p_raw[1]))),
                    0.0,
                ])

            def _spring_pts(p0: np.ndarray, p1: np.ndarray, coils: int = 8, amp: float = 0.16) -> list[np.ndarray]:
                vec = p1 - p0
                dist_v = float(np.linalg.norm(vec))
                if dist_v < 1e-4:
                    return [p0, p1]
                u_v = vec / dist_v
                n_v = np.array([-u_v[1], u_v[0], 0.0])
                pts = [p0, p0 + 0.10 * vec]
                for c_i in range(coils):
                    t_c = 0.12 + 0.76 * (c_i + 0.5) / float(coils)
                    sgn = 1.0 if (c_i % 2 == 0) else -1.0
                    pts.append(p0 + t_c * vec + sgn * amp * n_v)
                pts.extend([p0 + 0.90 * vec, p1])
                return pts

            p_anchor = _map_coord(anc_raw)
            lbl_s = (obj.label or "m").strip()

            def _mk_mech(t=tracker, frames=kfs, pa=p_anchor, p_init=pos_raw, ck=conn, c_mob=color):
                cur_xy = p_init
                if frames:
                    u = max(0.0, min(1.0, float(t.get_value())))
                    idx = min(len(frames) - 1, int(round(u * max(1, len(frames) - 1))))
                    cur_xy = getattr(frames[idx], "body_pos", None) or frames[idx].point or p_init
                pb = _map_coord(cur_xy)
                g_mech = m.VGroup()
                if ck == "rod":
                    g_mech.add(m.Line(pa + np.array([-0.45, 0.0, 0.0]), pa + np.array([0.45, 0.0, 0.0]), color=T.text_primary, stroke_width=4.0))
                    body = m.Circle(radius=0.28, color=c_mob, fill_color=c_mob, fill_opacity=0.88, stroke_width=2.8).move_to(pb)
                    half_span = 0.28
                else:
                    g_mech.add(m.Line(pa + np.array([0.0, -0.45, 0.0]), pa + np.array([0.0, 0.55, 0.0]), color=T.text_primary, stroke_width=4.0))
                    body = m.RoundedRectangle(corner_radius=0.10, width=0.72, height=0.60, color=c_mob, fill_color=c_mob, fill_opacity=0.90, stroke_width=2.6).move_to(pb)
                    half_span = 0.36
                vec_ap = pb - pa
                dist_ap = float(np.linalg.norm(vec_ap))
                p_edge = (pb - (vec_ap / dist_ap) * half_span) if dist_ap > half_span + 0.04 else pb
                if ck in ("spring", "spring_damper"):
                    sp = m.VMobject()
                    sp.set_points_as_corners(_spring_pts(pa, p_edge, coils=10, amp=0.18))
                    sp.set_stroke(color=T.text_primary, width=3.4)
                    g_mech.add(sp)
                if ck in ("damper", "spring_damper"):
                    off = np.array([0.0, -0.28, 0.0]) if ck == "spring_damper" else np.zeros(3)
                    mid = 0.5 * (pa + p_edge) + off
                    g_mech.add(m.Line(pa + off, mid, color=T.accent_secondary, stroke_width=3.2))
                    g_mech.add(m.RoundedRectangle(corner_radius=0.04, width=0.38, height=0.20, fill_color=T.surface, fill_opacity=0.85, stroke_color=T.accent_secondary, stroke_width=2.4).move_to(mid))
                    g_mech.add(m.Line(mid, p_edge + off, color=T.accent_secondary, stroke_width=2.8))
                if ck == "rod":
                    g_mech.add(m.Line(pa, p_edge, color=T.text_primary, stroke_width=3.2))
                g_mech.add(m.Dot(pa, radius=0.075, color=T.text_primary))
                g_mech.add(body)
                if lbl_s:
                    bt_fs = px_to_manim_font_size(max(28, int(self._px_scale.body_px)), self.profile.pixel_height)
                    bt = m.Text(lbl_s[:5], font_size=bt_fs, font=CAPTION_FONT, weight="BOLD", color=T.background, **_FONT_KWARGS)
                    fit_mobject(bt, max_width=half_span * 1.5, max_height=0.36, min_height=max(L.min_text_h, 0.24))
                    bt.move_to(body.get_center())
                    g_mech.add(bt)
                return g_mech

            mob_m = m.always_redraw(_mk_mech) if kfs else _mk_mech()
            self._obj_boxes[obj.id] = self._manim_to_px_box(_mk_mech(), role="mechanical_body")
            if kfs:
                mob_m._param_tracker = tracker  # type: ignore[attr-defined]
                mob_m._sweep_key = sw_key  # type: ignore[attr-defined]
                mob_m._sweep_keyframes = kfs  # type: ignore[attr-defined]
            return mob_m

        return m.Dot(L.stage_center, color=color)

    def _build_data_series_mob(self, obj: SceneObject, color: str) -> m.Mobject:
        """Render polyline points_list or animated sweep_keyframes cleanly onto axes or stage."""
        L = self.layout
        if getattr(obj, "sweep_keyframes", None):
            kfs = [kf for kf in obj.sweep_keyframes if kf.curve_pts and len(kf.curve_pts) >= 2]
            if kfs:
                sw_key = f"__sweep_{obj.model_id or obj.id}"
                init_u = float(getattr(obj, "param_init", 0.0) or 0.0)
                tracker = self.params.setdefault(sw_key, m.ValueTracker(init_u))
                self._model_selected_u.setdefault(sw_key, init_u)
                self._model_keyframes.setdefault(sw_key, kfs)
                ax_ref = self.registry.get(obj.axes_id or "")
                ax = getattr(ax_ref, "_wrapped_axes", ax_ref)
                ax_b = self.axes_ranges.get(obj.axes_id or "")

                def _mk_sw_curve(t=tracker, frames=kfs, a=ax, bounds=ax_b, c=color):
                    u = max(0.0, min(1.0, float(t.get_value())))
                    idx = min(len(frames) - 1, int(round(u * max(1, len(frames) - 1))))
                    raw_pts = frames[idx].curve_pts
                    if bounds is not None and isinstance(a, m.Axes):
                        runs = _clip_polyline_runs(raw_pts, bounds[0], bounds[1])
                        use_pts = runs[0] if runs else raw_pts
                        scene_pts = [a.c2p(p[0], p[1]) for p in use_pts]
                    elif isinstance(a, m.Axes):
                        scene_pts = [a.c2p(p[0], p[1]) for p in raw_pts]
                    else:
                        scene_pts = [L.stage_center + np.array([p[0], p[1], 0.0]) for p in raw_pts]
                    curve = m.VMobject()
                    curve.set_points_as_corners(scene_pts)
                    curve.set_stroke(color=c, width=3.4)
                    return curve

                mob = m.always_redraw(_mk_sw_curve)
                mob._param_tracker = tracker  # type: ignore[attr-defined]
                mob._sweep_key = sw_key  # type: ignore[attr-defined]
                mob._sweep_keyframes = kfs  # type: ignore[attr-defined]
                return mob
        pts_raw = obj.points_list
        if not pts_raw and obj.kind == "waveform" and obj.expr:
            fn_w = compile_expr(obj.expr, ("x",))
            xr = obj.x_range or [-3.14, 3.14]
            xs_w = np.linspace(xr[0], xr[1], 120)
            pts_raw = [[float(xw), float(fn_w(x=float(xw)))] for xw in xs_w]
        if pts_raw and len(pts_raw) >= 2:
            ax_bounds = self.axes_ranges.get(obj.axes_id or "")
            if ax_bounds is not None:
                runs = _clip_polyline_runs(pts_raw, ax_bounds[0], ax_bounds[1])
                if not runs:
                    return m.VGroup()
                if len(runs) == 1:
                    curve = m.VMobject()
                    curve.set_points_as_corners([self._to_point(p[:2], obj.axes_id) for p in runs[0]])
                    curve.set_stroke(color=color, width=3.2)
                    return curve
                grp = m.VGroup()
                for run in runs:
                    sub = m.VMobject()
                    sub.set_points_as_corners([self._to_point(p[:2], obj.axes_id) for p in run])
                    sub.set_stroke(color=color, width=3.2)
                    grp.add(sub)
                return grp
            scene_pts = [self._to_point(p[:2], obj.axes_id) for p in pts_raw]
            curve = m.VMobject()
            curve.set_points_as_corners(scene_pts)
            curve.set_stroke(color=color, width=3.2)
            return curve
        return m.Dot(self._slot_center(obj), color=color)
