"""Composable Object-and-Event ManimGX renderer for VideoPlan V3/V3.1."""

from __future__ import annotations

import os
from pathlib import Path
import manimgx as m
import numpy as np

from scenes.multi_scene import plot_clipped_segments
from studio.ai.expr import compile_expr
from studio.ai.v3_schema import AnimationEvent, SceneObject, VideoPlanV3
from studio.ai.world_model import materialize_constructions
from studio.audio.tts import BeatSpec
from studio.config import CAPTION_FONT
from studio.layout import CaptionCard, fit_mobject
from studio.registry.compositions import adapt_composition_for_aspect
from studio.scene import StoryScene

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


class V3Scene(StoryScene):
    """Persistent object-and-event educational animation engine (VideoPlan V3/V3.1)."""

    BEATS = [
        BeatSpec(id=s.scene_id, narration=s.narration, caption=s.purpose, pause_after=0.30, speaker=s.speaker)
        for s in V3_PLAN.scenes
    ]

    def construct(self) -> None:
        self.setup_studio()
        L, T = self.layout, self.theme
        if not V3_PLAN.scenes:
            return
        self.caption_card = CaptionCard(L, T, V3_PLAN.scenes[0].narration)
        self.add(self.caption_card.group)

        self.registry: dict[str, m.Mobject] = {}
        self.obj_kinds: dict[str, str] = {}
        self.obj_axes: dict[str, str | None] = {}
        self._obj_specs: dict[str, SceneObject] = {}
        self._hidden_persisted: dict[str, m.Mobject] = {}
        self.axes_ranges: dict[str, tuple[list[float], list[float]]] = {}
        self.params: dict[str, m.ValueTracker] = {}
        self._active_eq_id: str | None = None
        self._active_ann_id: str | None = None
        self._geom_frame: tuple[float, np.ndarray] | None = None
        self._live_unaxed_polys: set[str] = set()

        for idx, sc in enumerate(V3_PLAN.scenes):
            self._cur_scene_idx = idx
            self._cur_scene_comp = getattr(sc, "composition", "auto")
            # Auto-assign side-by-side / stacked slots if multiple conflicting apparatus/un-axed polygons co-occur
            new_app = [
                o for o in sc.objects
                if o.kind in ("axes", "table", "matrix", "bar_chart", "graph_network")
                or (o.kind == "polygon" and not o.axes_id)
            ]
            if len(new_app) >= 2:
                used_slots = {o.panel_slot for o in new_app if o.panel_slot in ("left", "right", "top", "bottom")}
                for cand in new_app:
                    if cand.panel_slot in ("auto", "center"):
                        if "left" not in used_slots:
                            cand.panel_slot = "left"
                            used_slots.add("left")
                        elif "right" not in used_slots:
                            cand.panel_slot = "right"
                            used_slots.add("right")
            self._cur_ann_index = 0
            self._cur_ann_total = sum(1 for o in sc.objects if o.kind == "annotation")
            self._cur_eq_index = 0
            self._cur_eq_total = sum(1 for o in sc.objects if o.kind == "equation")

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

            main_anims: list[m.Animation] = []
            # Restore any hidden persisted axes if the current scene attaches objects/events to it
            for ref_id in list(referenced_ids):
                if ref_id in self._hidden_persisted and ref_id not in self.registry:
                    restored_mob = self._hidden_persisted.pop(ref_id)
                    self.registry[ref_id] = restored_mob
                    if ref_id in self._obj_specs:
                        self.obj_kinds[ref_id] = self._obj_specs[ref_id].kind
                        self.obj_axes[ref_id] = self._obj_specs[ref_id].axes_id
                    main_anims.append(m.FadeIn(restored_mob))

            # Retire or temporarily hide superseded center-stage apparatus so scenes never pile up
            if any(o.kind in ("axes", "table", "matrix", "bar_chart", "graph_network") for o in sc.objects):
                retire_axes: set[str] = set()
                for lid in list(self.registry.keys()):
                    lkind = self.obj_kinds.get(lid, "")
                    laxes = self.obj_axes.get(lid)
                    if lid in referenced_ids or laxes in referenced_ids:
                        continue
                    if lkind in ("axes", "table", "matrix", "bar_chart", "graph_network", "annotation", "function_plot", "parametric_curve", "data_series", "waveform", "particle_system"):
                        old_mob = self.registry.pop(lid)
                        self.obj_kinds.pop(lid, None)
                        self.obj_axes.pop(lid, None)
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
                            if l_spec and l_spec.persist:
                                self._hidden_persisted[lid] = old_mob
                            main_anims.append(m.FadeOut(old_mob))

            if new_ann_ids and self._active_ann_id and self._active_ann_id in self.registry:
                if self._active_ann_id not in referenced_ids:
                    main_anims.append(m.FadeOut(self.registry.pop(self._active_ann_id)))
                self._active_ann_id = None

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

            transform_srcs = {e.source_id for e in sc.events if e.op in ("transform", "morph") and e.source_id}
            if new_eq_ids and self._active_eq_id and self._active_eq_id in self.registry:
                if self._active_eq_id not in transform_srcs:
                    old_eq = self.registry.pop(self._active_eq_id)
                    main_anims.append(m.FadeOut(old_eq))
                self._active_eq_id = None

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

            if idx == 0 and sc.objects and anchored_create_ids and all(o.id in anchored_create_ids for o in sc.objects):
                first_obj = sc.objects[0]
                main_anims.append(m.FadeIn(self.registry[first_obj.id]))
                for mk, evl in list(cues.items()):
                    cues[mk] = [e for e in evl if not (e.target_id == first_obj.id and e.op in ("create", "draw", "reveal"))]
                    if not cues[mk]:
                        cues.pop(mk)
                cue_callables = {mark: (lambda evs=ev_list: _compile_cue_group(evs)) for mark, ev_list in cues.items()}

            self.play_beat(sc.scene_id, *main_anims, cues=cue_callables or None, sfx=sc.sfx_cue)
            if new_eq_ids:
                self._active_eq_id = new_eq_ids[-1]
            if new_ann_ids:
                self._active_ann_id = new_ann_ids[-1]

            to_drop = [self.registry.pop(o.id) for o in sc.objects if not o.persist and o.id in self.registry]
            for o in sc.objects:
                if not o.persist:
                    self._live_unaxed_polys.discard(o.id)
                    self.obj_kinds.pop(o.id, None)
                    self.obj_axes.pop(o.id, None)
            if to_drop:
                self.remove(*to_drop)
            # Reset composition frame when no un-axed polygons remain live
            if not any(oid in self.registry for oid in self._live_unaxed_polys):
                self._live_unaxed_polys.clear()
                self._geom_frame = None

    def _slot_center(self, obj: SceneObject, scene_comp: str = "auto") -> np.ndarray:
        L = self.layout
        is_port = L.frame_h > L.frame_w
        aspect_lbl = "9:16" if is_port else "16:9"
        eff_comp = adapt_composition_for_aspect(scene_comp, aspect_lbl)
        slot = obj.panel_slot
        if slot == "auto" and eff_comp == "side_by_side_comparison" and obj.kind in ("axes", "matrix", "bar_chart", "graph_network", "table"):
            slot = "auto"
        dx = L.stage_w * 0.24
        dy = L.stage_h * 0.22
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
        if ev.target_id not in self.registry:
            raise RuntimeError(f"AnimationEvent target_id {ev.target_id!r} not found in registry.")
        mob = self.registry[ev.target_id]

        if ev.op in ("create", "draw", "reveal"):
            return m.Create(mob) if isinstance(mob, m.VMobject) else m.FadeIn(mob)
        if ev.op in ("transform", "morph"):
            if not ev.source_id or ev.source_id not in self.registry:
                raise RuntimeError(f"{ev.op} source_id {ev.source_id!r} not found in registry.")
            src = self.registry[ev.source_id]
            if self.obj_kinds.get(ev.source_id) != self.obj_kinds.get(ev.target_id):
                # Cross-kind visual morph (e.g. matrix -> polygon) should highlight source and reveal target without erasing source
                return m.AnimationGroup(m.Indicate(src, scale_factor=1.06), m.FadeIn(mob))
            self.registry[ev.target_id] = src
            if ev.source_id in self._live_unaxed_polys:
                self._live_unaxed_polys.discard(ev.source_id)
                self._live_unaxed_polys.add(ev.target_id)
            return m.Transform(src, mob)
        if ev.op in ("highlight", "group"):
            return m.Indicate(mob, scale_factor=1.08)
        if ev.op == "move" and ev.shift:
            s = self._get_geom_scale() if not getattr(mob, "_uses_axes", False) else 1.0
            vec = np.array([ev.shift[0] * s, ev.shift[1] * s, (ev.shift[2] * s) if len(ev.shift) > 2 else 0.0])
            return mob.animate.shift(vec)
        if ev.op == "scale" and ev.scale_factor:
            return mob.animate.scale(ev.scale_factor)
        if ev.op == "zoom":
            return mob.animate.scale(float(ev.scale_factor or 1.18))
        if ev.op == "rotate" and ev.angle_deg:
            # Never turn a matrix, table, equation, annotation, axes, or bar_chart diagonally sideways
            if self.obj_kinds.get(ev.target_id, "") in ("matrix", "table", "equation", "axes", "annotation", "bar_chart"):
                return m.Indicate(mob, scale_factor=1.06)
            return m.Rotate(mob, angle=float(np.deg2rad(ev.angle_deg)))
        if ev.op == "remove":
            self.registry.pop(ev.target_id, None)
            self._live_unaxed_polys.discard(ev.target_id)
            return m.FadeOut(mob)
        if ev.op == "focus":
            return m.Circumscribe(mob, color=self.theme.accent_highlight)
        if ev.op == "trace":
            return m.Create(mob) if isinstance(mob, m.VMobject) else m.FadeIn(mob)
        if ev.op == "sequence" and isinstance(mob, m.VGroup) and len(mob) > 1:
            return m.LaggedStart(*(m.FadeIn(sub) for sub in mob), lag_ratio=0.18)
        if ev.op in ("parallel",):
            return m.AnimationGroup(*(m.FadeIn(sub) for sub in (mob if isinstance(mob, m.VGroup) else [mob])))
        if ev.op in ("animate_param", "iterate_states", "interpolate_values") and ev.param_range and len(ev.param_range) == 2:
            tracker = getattr(mob, "_param_tracker", None)
            if tracker is None:
                return m.Indicate(mob, scale_factor=1.06)
            tracker.set_value(float(ev.param_range[0]))
            return tracker.animate.set_value(float(ev.param_range[1]))
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

    def _to_point(self, coords: list[float], axes_id: str | None) -> np.ndarray:
        ax = self.registry.get(axes_id or "")
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
            is_port = L.frame_h > L.frame_w
            comp_now = getattr(self, "_cur_scene_comp", "auto")
            eff_comp = adapt_composition_for_aspect(comp_now, "9:16" if is_port else "16:9")
            slot_scale = 0.48 if (obj.panel_slot in ("left", "right", "top", "bottom") or eff_comp in ("side_by_side_comparison", "stacked_comparison")) else 1.0
            x_len = L.stage_w * (0.44 if slot_scale < 1.0 and not is_port else 0.90)
            if obj.equal_aspect:
                y_len = min(L.stage_h * 0.66 * slot_scale, x_len * (yr[1] - yr[0]) / max(xr[1] - xr[0], 1e-6))
                x_len = y_len * (xr[1] - xr[0]) / max(yr[1] - yr[0], 1e-6)
            else:
                y_len = L.stage_h * (0.32 if slot_scale < 1.0 and is_port else 0.62)
            ax = m.Axes(
                x_range=[xr[0], xr[1], (xr[1] - xr[0]) / 4.0],
                y_range=[yr[0], yr[1], (yr[1] - yr[0]) / 4.0],
                x_length=x_len,
                y_length=y_len,
                axis_config={"color": T.text_muted, "stroke_width": 1.8, "include_tip": True},
                tips=True,
            ).move_to(self._slot_center(obj, comp_now))
            self.axes_ranges[obj.id] = (xr, yr)
            return ax

        if obj.kind == "function_plot":
            ax = self.registry.get(obj.axes_id or "")
            if isinstance(ax, m.Axes) and obj.axes_id in self.axes_ranges:
                xr, yr = self.axes_ranges[obj.axes_id]
            else:
                xr, yr = obj.x_range or [-3.5, 3.5], obj.y_range or [-1.5, 1.5]
                ax = m.Axes(x_range=[xr[0], xr[1], 1.0], y_range=[yr[0], yr[1], 1.0], x_length=L.stage_w * 0.88, y_length=L.stage_h * 0.65).move_to(L.stage_center)
            fn = compile_expr(obj.expr or "0", ("x",))
            return plot_clipped_segments(ax, fn, xr, yr, color=color, stroke_width=3.5)

        if obj.kind == "parametric_curve":
            ax = self.registry.get(obj.axes_id or "")
            if not isinstance(ax, m.Axes):
                xr, yr = obj.x_range or [-2.0, 2.0], obj.y_range or [-2.0, 2.0]
                ax = m.Axes(x_range=[xr[0], xr[1], 1.0], y_range=[yr[0], yr[1], 1.0], x_length=L.stage_w * 0.8, y_length=L.stage_h * 0.6).move_to(L.stage_center)
            p0, p1 = (obj.expr or "cos(t); sin(t)").split(";", 1)
            fx, fy = compile_expr(p0, ("t",)), compile_expr(p1, ("t",))
            tr = obj.t_range or [0.0, float(2 * np.pi)]
            ts = np.linspace(tr[0], tr[1], 240)
            pts = [ax.c2p(float(fx(t=t)), float(fy(t=t))) for t in ts]
            curve = m.VMobject()
            curve.set_points_smoothly(pts)
            curve.set_stroke(color=color, width=3.5)
            return curve

        if obj.kind == "equation":
            eq = m.MathTex(obj.tex or "x", font_size=40, color=color)
            eq_i = getattr(self, "_cur_eq_index", 0)
            eq_tot = max(1, getattr(self, "_cur_eq_total", 1))
            is_port = L.frame_h > L.frame_w
            eq_center = L.header_center.copy()
            max_w_ratio = 0.84
            max_h_ratio = 0.56
            if eq_tot >= 2:
                if not is_port:
                    slot_w = L.safe_w / eq_tot
                    eq_center[0] = (eq_i - 0.5 * (eq_tot - 1)) * slot_w
                    max_w_ratio = min(0.84, (slot_w * 0.86) / L.safe_w)
                else:
                    slot_h = L.header_max_h / eq_tot
                    eq_center[1] = L.header_center[1] + (0.5 * (eq_tot - 1) - eq_i) * slot_h
                    max_h_ratio = min(0.56, (slot_h * 0.72) / L.header_max_h)
            if obj.card or obj.color_role == "highlight":
                fit_mobject(eq, max_width=L.safe_w * max_w_ratio, max_height=L.header_max_h * max_h_ratio)
                eq.move_to(eq_center)
                pill = m.RoundedRectangle(
                    corner_radius=0.16,
                    width=min(L.safe_w * (max_w_ratio + 0.08), eq.width + 0.48),
                    height=min(L.header_max_h * (max_h_ratio + 0.16), eq.height + 0.28),
                    fill_color=T.surface,
                    fill_opacity=0.90,
                    stroke_color=color,
                    stroke_opacity=0.45,
                    stroke_width=1.8,
                ).move_to(eq_center)
                return m.VGroup(pill, eq)
            fit_mobject(eq, max_width=L.safe_w * (max_w_ratio + 0.04), max_height=L.header_max_h * (max_h_ratio + 0.08))
            eq.move_to(eq_center)
            return eq

        if obj.kind == "annotation":
            raw_lbl = (obj.text or obj.label or "").strip()
            has_tex = any(tok in raw_lbl for tok in ("\\", "_", "^"))
            if has_tex:
                try:
                    txt = m.MathTex(raw_lbl, font_size=30, color=color)
                except Exception:
                    txt = m.Text(raw_lbl, font_size=26, font=CAPTION_FONT, weight="BOLD", color=color)
            else:
                txt = m.Text(raw_lbl, font_size=26, font=CAPTION_FONT, weight="BOLD", color=color)
            fit_mobject(txt, max_width=L.safe_w * 0.85, max_height=0.55)
            ax_mob = self.registry.get(obj.axes_id or "")
            if not isinstance(ax_mob, m.Axes):
                ax_mob = next((mob for mob in self.registry.values() if isinstance(mob, m.Axes)), None)
            ann_i = getattr(self, "_cur_ann_index", 0)
            ann_tot = max(1, getattr(self, "_cur_ann_total", 1))
            is_port = L.frame_h > L.frame_w
            if isinstance(ax_mob, m.Axes) and ann_tot >= 2 and not is_port:
                fit_mobject(txt, max_width=L.safe_w * 0.34, max_height=0.55)
                is_left = (ann_i % 2 == 0)
                if is_left:
                    side_x = 0.5 * (-L.safe_w * 0.48 + ax_mob.get_left()[0])
                else:
                    side_x = 0.5 * (L.safe_w * 0.48 + ax_mob.get_right()[0])
                row_off = (0.5 - (ann_i // 2)) * 0.72
                pos = np.array([side_x, ax_mob.get_center()[1] + row_off, 0.0])
            elif isinstance(ax_mob, m.Axes) and is_port and ann_tot >= 2:
                fit_mobject(txt, max_width=L.safe_w * 0.86, max_height=0.55)
                if ann_i % 2 == 0:
                    pos = np.array([ax_mob.get_center()[0], ax_mob.get_top()[1] + txt.height * 0.65 + 0.25 + (ann_i // 2) * 0.60, 0.0])
                else:
                    pos = np.array([ax_mob.get_center()[0], ax_mob.get_bottom()[1] - txt.height * 0.65 - 0.25 - (ann_i // 2) * 0.60, 0.0])
            elif isinstance(ax_mob, m.Axes):
                max_y = L.header_center[1] - L.header_max_h * 0.56 - txt.height * 0.52
                pref_y = ax_mob.get_top()[1] + txt.height * 0.55 + 0.22 - (ann_i * 0.55 if ann_tot > 1 else 0.0)
                pos = np.array([ax_mob.get_center()[0], min(max_y, pref_y), 0.0])
                if pos[1] - txt.height * 0.55 <= ax_mob.get_top()[1] + 0.08:
                    side_x = min(L.safe_w * 0.45 - txt.width * 0.5, ax_mob.get_right()[0] + txt.width * 0.5 + 0.35)
                    pos = np.array([side_x, ax_mob.get_top()[1] - txt.height * 0.6 - ann_i * 0.58, 0.0])
            elif obj.coords and len(obj.coords) >= 2:
                pos = self._to_point(obj.coords, obj.axes_id)
            else:
                y_shift = (0.5 * (ann_tot - 1) - ann_i) * 0.72
                pos = L.stage_center + np.array([0.0, L.stage_h * 0.18 + y_shift, 0.0])
            txt.move_to(pos)
            return txt

        if obj.kind == "point":
            if obj.param_name and obj.expr and ";" in obj.expr:
                tracker = self.params.setdefault(obj.param_name, m.ValueTracker(obj.param_init))
                p0, p1 = obj.expr.split(";", 1)
                fx, fy = compile_expr(p0, (obj.param_name,)), compile_expr(p1, (obj.param_name,))
                ax = self.registry.get(obj.axes_id or "")
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
            if obj.kind == "vector" and obj.param_name and obj.expr and ";" in obj.expr:
                tracker = self.params.setdefault(obj.param_name, m.ValueTracker(obj.param_init))
                p0, p1 = obj.expr.split(";", 1)
                fx, fy = compile_expr(p0, (obj.param_name,)), compile_expr(p1, (obj.param_name,))
                ax = self.registry.get(obj.axes_id or "")
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
            return poly

        if obj.kind == "circle":
            pos = self._to_point(obj.coords or [0.0, 0.0], obj.axes_id)
            r = obj.radius or 1.0
            ax = self.registry.get(obj.axes_id or "")
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
            if not obj.axes_id:
                for v in self.registry.values():
                    if isinstance(v, m.Axes) or getattr(v, "_is_bar_chart", False):
                        self.remove(v)
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
            w = L.stage_w * 0.78 / max(1, n)
            base_y = L.stage_center[1] - L.stage_h * 0.10
            max_h = L.stage_h * 0.40
            lbl_step = max(1, (n + 7) // 8)
            for i, frac in enumerate(norm):
                h = max(0.0, abs(frac) * max_h)
                if h > 0.0:
                    bar = m.Rectangle(width=w * 0.62, height=h, fill_color=color, fill_opacity=0.88, stroke_width=0)
                    x_pos = L.stage_center[0] - L.stage_w * 0.39 + (i + 0.5) * w
                    y_off = h * 0.5 if frac >= 0 else -h * 0.5
                    bar.move_to(np.array([x_pos, base_y + y_off, 0.0]))
                    grp.add(bar)
                if i < len(obj.bar_labels) and obj.bar_labels[i] and (i % lbl_step == 0 or (i == n - 1 and (n - 1) % lbl_step >= max(2, lbl_step // 2))):
                    lbl = m.Text(obj.bar_labels[i], font_size=24, font=CAPTION_FONT, color=T.text_muted)
                    fit_mobject(lbl, max_width=w * lbl_step * 0.85, min_height=L.min_text_h)
                    x_pos = L.stage_center[0] - L.stage_w * 0.39 + (i + 0.5) * w
                    lbl.move_to(np.array([x_pos, base_y - 0.32, 0.0]))
                    grp.add(lbl)
            grp._is_bar_chart = True  # type: ignore[attr-defined]
            return grp

        if obj.kind == "angle_arc":
            tracker = self.params.setdefault(obj.param_name or "theta", m.ValueTracker(obj.param_init))
            ax = self.registry.get(obj.axes_id or "")
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
            ax = self.registry.get(obj.axes_id or "")
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
                fs = 34 if p_mode else 30
                if any(tok in txt for tok in ("\\", "_", "^")):
                    try:
                        return m.MathTex(txt, font_size=fs, color=c)
                    except Exception:
                        pass
                return m.Text(txt, font_size=fs, font=CAPTION_FONT, color=c)

            mat_mob = m.Matrix(str_rows, v_buff=0.68, h_buff=0.95, element_to_mobject=_mk_cell)
            max_w = L.stage_w * (0.44 if obj.panel_slot in ("left", "right") else (0.78 if is_port else 0.58))
            fit_mobject(mat_mob, max_width=max_w, max_height=L.stage_h * 0.54)
            mat_mob.move_to(self._slot_center(obj, getattr(self, "_cur_scene_comp", "auto")))
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
                            mob = m.Text(val, font_size=24 if is_hdr else 22, font=CAPTION_FONT, weight="BOLD" if is_hdr else "NORMAL", color=color if is_hdr else T.text_primary)
                    else:
                        mob = m.Text(val, font_size=24 if is_hdr else 22, font=CAPTION_FONT, weight="BOLD" if is_hdr else "NORMAL", color=color if is_hdr else T.text_primary)
                    col_widths[ci] = max(col_widths[ci], mob.width + 0.48)
                    row_heights[ri] = max(row_heights[ri], mob.height + 0.28)
                    c_mobs.append(mob)
                grid_cells.append(c_mobs)

            total_w = sum(col_widths)
            total_h = sum(row_heights)
            max_w = L.stage_w * (0.46 if obj.panel_slot in ("left", "right") else 0.88)
            max_h = L.stage_h * (0.44 if obj.panel_slot in ("top", "bottom") else 0.72)
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
            return tbl_grp

        if obj.kind in ("data_series", "waveform"):
            pts_raw = obj.points_list
            if not pts_raw and obj.kind == "waveform" and obj.expr:
                fn_w = compile_expr(obj.expr, ("x",))
                xr = obj.x_range or [-3.14, 3.14]
                xs_w = np.linspace(xr[0], xr[1], 120)
                pts_raw = [[float(xw), float(fn_w(x=float(xw)))] for xw in xs_w]
            if pts_raw and len(pts_raw) >= 2:
                scene_pts = [self._to_point(p[:2], obj.axes_id) for p in pts_raw]
                curve = m.VMobject()
                curve.set_points_as_corners(scene_pts)
                curve.set_stroke(color=color, width=3.2)
                return curve
            return m.Dot(self._slot_center(obj), color=color)

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
                lbl = m.Text(lbl_str, font_size=20, font=CAPTION_FONT, color=T.text_primary)
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

        return m.Dot(L.stage_center, color=color)
