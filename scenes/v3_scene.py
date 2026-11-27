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
        BeatSpec(id=s.scene_id, narration=s.narration, caption=s.purpose, pause_after=0.30)
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
        self.axes_ranges: dict[str, tuple[list[float], list[float]]] = {}
        self.params: dict[str, m.ValueTracker] = {}
        self._active_eq_id: str | None = None
        self._geom_frame: tuple[float, np.ndarray] | None = None
        self._live_unaxed_polys: set[str] = set()

        for idx, sc in enumerate(V3_PLAN.scenes):
            self._cur_scene_idx = idx
            new_eq_ids = [o.id for o in sc.objects if o.kind == "equation"]
            for obj in sc.objects:
                if obj.id in self.registry:
                    raise RuntimeError(f"Scene {sc.scene_id}: duplicate live object {obj.id!r}")
                self.registry[obj.id] = self._build_object(obj)
                if obj.kind == "polygon" and not obj.axes_id:
                    self._live_unaxed_polys.add(obj.id)

            transform_srcs = {e.source_id for e in sc.events if e.op == "transform" and e.source_id}
            main_anims: list[m.Animation] = []
            if new_eq_ids and self._active_eq_id and self._active_eq_id in self.registry:
                if self._active_eq_id not in transform_srcs:
                    old_eq = self.registry.pop(self._active_eq_id)
                    main_anims.append(m.FadeOut(old_eq))
                self._active_eq_id = None

            cues: dict[str, list[AnimationEvent]] = {}
            anchored_ids: set[str] = set()
            for ev in sc.events:
                if ev.anchor_mark:
                    cues.setdefault(ev.anchor_mark, []).append(ev)
                    anchored_ids.add(ev.target_id)
                else:
                    main_anims.append(self._compile_event(ev))

            cue_callables = {
                mark: (lambda evs=ev_list: [self._compile_event(e) for e in evs])
                for mark, ev_list in cues.items()
            }

            if not main_anims and sc.objects:
                # Only auto-reveal objects that don't have a bookmark-anchored event
                unanchored = [self.registry[o.id] for o in sc.objects if o.id not in anchored_ids and o.id in self.registry]
                main_anims = [m.FadeIn(mob) for mob in unanchored]

            self.play_beat(sc.scene_id, *main_anims, cues=cue_callables or None, sfx=sc.sfx_cue)
            if new_eq_ids:
                self._active_eq_id = new_eq_ids[-1]

            to_drop = [self.registry.pop(o.id) for o in sc.objects if not o.persist and o.id in self.registry]
            for o in sc.objects:
                if not o.persist:
                    self._live_unaxed_polys.discard(o.id)
            if to_drop:
                self.remove(*to_drop)
            # Reset composition frame when no un-axed polygons remain live
            if not any(oid in self.registry for oid in self._live_unaxed_polys):
                self._live_unaxed_polys.clear()
                self._geom_frame = None

    def _compile_event(self, ev: AnimationEvent) -> m.Animation:
        if ev.target_id not in self.registry:
            raise RuntimeError(f"AnimationEvent target_id {ev.target_id!r} not found in registry.")
        mob = self.registry[ev.target_id]

        if ev.op in ("create", "draw"):
            return m.Create(mob) if isinstance(mob, m.VMobject) else m.FadeIn(mob)
        if ev.op == "transform":
            if not ev.source_id or ev.source_id not in self.registry:
                raise RuntimeError(f"Transform source_id {ev.source_id!r} not found in registry.")
            src = self.registry[ev.source_id]
            self.registry[ev.target_id] = src
            return m.Transform(src, mob)
        if ev.op == "highlight":
            return m.Indicate(mob, scale_factor=1.08)
        if ev.op == "move" and ev.shift:
            s = self._get_geom_scale() if not getattr(mob, "_uses_axes", False) else 1.0
            vec = np.array([ev.shift[0] * s, ev.shift[1] * s, (ev.shift[2] * s) if len(ev.shift) > 2 else 0.0])
            return mob.animate.shift(vec)
        if ev.op == "scale" and ev.scale_factor:
            return mob.animate.scale(ev.scale_factor)
        if ev.op == "rotate" and ev.angle_deg:
            return m.Rotate(mob, angle=float(np.deg2rad(ev.angle_deg)))
        if ev.op == "remove":
            self.registry.pop(ev.target_id, None)
            self._live_unaxed_polys.discard(ev.target_id)
            return m.FadeOut(mob)
        if ev.op == "focus":
            return m.Circumscribe(mob, color=self.theme.accent_highlight)
        if ev.op == "animate_param" and ev.param_range and len(ev.param_range) == 2:
            tracker = getattr(mob, "_param_tracker", None)
            if tracker is None:
                raise RuntimeError(f"Object {ev.target_id!r} has no bound parameter tracker.")
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
            x_len = L.stage_w * 0.92
            if obj.equal_aspect:
                y_len = min(L.stage_h * 0.78, x_len * (yr[1] - yr[0]) / max(xr[1] - xr[0], 1e-6))
                x_len = y_len * (xr[1] - xr[0]) / max(yr[1] - yr[0], 1e-6)
            else:
                y_len = L.stage_h * 0.72
            ax = m.Axes(
                x_range=[xr[0], xr[1], (xr[1] - xr[0]) / 4.0],
                y_range=[yr[0], yr[1], (yr[1] - yr[0]) / 4.0],
                x_length=x_len,
                y_length=y_len,
                axis_config={"color": T.text_muted, "stroke_width": 1.8, "include_tip": True},
                tips=True,
            ).move_to(L.stage_center)
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
            if obj.card or obj.color_role == "highlight":
                fit_mobject(eq, max_width=L.safe_w * 0.84, max_height=L.header_max_h * 0.56)
                eq.move_to(L.header_center)
                pill = m.RoundedRectangle(
                    corner_radius=0.16,
                    width=min(L.safe_w * 0.94, eq.width + 0.56),
                    height=min(L.header_max_h * 0.82, eq.height + 0.34),
                    fill_color=T.surface,
                    fill_opacity=0.90,
                    stroke_color=color,
                    stroke_opacity=0.45,
                    stroke_width=1.8,
                ).move_to(L.header_center)
                return m.VGroup(pill, eq)
            fit_mobject(eq, max_width=L.safe_w * 0.90, max_height=L.header_max_h * 0.65)
            eq.move_to(L.header_center)
            return eq

        if obj.kind == "annotation":
            txt = m.Text(obj.text or obj.label, font_size=26, font=CAPTION_FONT, weight="BOLD", color=color)
            fit_mobject(txt, max_width=L.safe_w * 0.85, min_height=L.min_text_h)
            pos = self._to_point(obj.coords, obj.axes_id) if obj.coords and len(obj.coords) >= 2 else (L.stage_center + np.array([0.0, L.stage_h * 0.36, 0.0]))
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
                    if isinstance(v, m.Axes):
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
            for i, frac in enumerate(norm):
                h = max(0.0, abs(frac) * max_h)
                if h > 0.0:
                    bar = m.Rectangle(width=w * 0.62, height=h, fill_color=color, fill_opacity=0.88, stroke_width=0)
                    x_pos = L.stage_center[0] - L.stage_w * 0.39 + (i + 0.5) * w
                    y_off = h * 0.5 if frac >= 0 else -h * 0.5
                    bar.move_to(np.array([x_pos, base_y + y_off, 0.0]))
                    grp.add(bar)
                if i < len(obj.bar_labels) and obj.bar_labels[i]:
                    lbl = m.Text(obj.bar_labels[i], font_size=24, font=CAPTION_FONT, color=T.text_muted)
                    fit_mobject(lbl, max_width=w * 1.15, min_height=L.min_text_h)
                    x_pos = L.stage_center[0] - L.stage_w * 0.39 + (i + 0.5) * w
                    lbl.move_to(np.array([x_pos, base_y - 0.25, 0.0]))
                    grp.add(lbl)
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

        return m.Dot(L.stage_center, color=color)
