"""Plan-driven template scenes: render any validated AI `ShortPlan` (studio/ai/schema.py).

The plan file comes from STUDIO_PLAN (set by `studio.pipeline --topic/--prompt/--plan`);
without it, the bundled sample plan renders so `manimgx inspect` works directly.

Templates:
  graph       y = f(x) morphing beat to beat, optional fixed reference curve + error ribbon
  parametric  (X(t), Y(t)) curves with a moving point and velocity vector
  sequence    terms a(n) revealed as dots converging on a reference/limit line
  surface     z = f(x, y) in 3D (AIShort3D) with an orbiting gradient probe and slices

Every element is positioned from SafeLayout, every timing comes from play_beat, and each
template keeps a continuously moving tracer + live readout so no beat tail is ever static.
"""

from __future__ import annotations

import os
from pathlib import Path

import manimgx as m
import numpy as np

from studio.ai.expr import compile_expr
from studio.ai.schema import ShortPlan, strip_markup
from studio.audio.tts import BeatSpec
from studio.layout import (
    CaptionCard,
    HeaderBanner,
    StatBadge,
    adaptive_tex,
    ambient_grid,
    fit_mobject,
    glow,
    precise_tex,
)
from studio.scene import Story3DScene, StoryScene

DEFAULT_PLAN = Path(__file__).resolve().parent.parent / "plans" / "zero_point_nine.json"
PLAN_PATH = Path(os.environ.get("STUDIO_PLAN", str(DEFAULT_PLAN)))
PLAN = ShortPlan.model_validate_json(PLAN_PATH.read_text(encoding="utf-8"))

ROLE_TO_THEME = {
    "primary": "accent_primary",
    "secondary": "accent_secondary",
    "tertiary": "accent_tertiary",
    "success": "success",
    "highlight": "accent_highlight",
    "danger": "danger",
}
PAUSE_AFTER = 0.25
FINAL_PAUSE = 0.45
TRACER_PERIOD_S = 6.0  # one full back-and-forth sweep of the live tracer


def plan_beats(plan: ShortPlan) -> list[BeatSpec]:
    return [
        BeatSpec(
            id=b.beat_id,
            narration=b.narration,
            caption=b.caption,
            pause_after=FINAL_PAUSE if i == len(plan.beats) - 1 else PAUSE_AFTER,
        )
        for i, b in enumerate(plan.beats)
    ]


def largest_inside_run(xs: np.ndarray, ys: np.ndarray, lo: float, hi: float) -> tuple[float, float] | None:
    """Widest contiguous x-interval where f stays finite and inside [lo, hi] (curves never leave the plot)."""
    ok = np.isfinite(ys) & (ys >= lo) & (ys <= hi)
    best, start = None, None
    for i, flag in enumerate(np.append(ok, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best is None or best[1] - best[0] < 3:
        return None
    return float(xs[best[0]]), float(xs[best[1] - 1])


class _PlanMixin:
    """Shared plan plumbing for the 2D and 3D templates."""

    PLAN = PLAN
    BEATS = plan_beats(PLAN)

    def color(self, role: str) -> str:
        return getattr(self.theme, ROLE_TO_THEME.get(role, "accent_primary"))

    def chrome(self) -> HeaderBanner:
        plan = self.PLAN
        header = HeaderBanner(
            layout=self.layout,
            theme=self.theme,
            tag_text=plan.tag,
            title_text=plan.title,
            initial_formula=plan.beats[0].formula,
        )
        header.formula.set_color(self.color(plan.beats[0].color_role))
        self.caption_card = CaptionCard(
            layout=self.layout, theme=self.theme, initial_text=strip_markup(plan.beats[0].narration)
        )
        return header


class AIShort(_PlanMixin, StoryScene):
    """2D template: graph, parametric and sequence plans."""

    def construct(self) -> None:
        self.setup_studio()
        plan, th, lay = self.PLAN, self.theme, self.layout
        kind = plan.template
        grid = ambient_grid(lay, th)
        header = self.chrome()

        # ------------------------------------------------------------------ stage
        (x0, x1), (y0, y1) = plan.x_range, plan.y_range
        axes_h = lay.stage_h * 0.58
        axes_w = min(lay.stage_w * 0.92, axes_h * 1.95)
        stage_top = lay.stage_center[1] + lay.stage_h * 0.5
        stage_bot = lay.stage_center[1] - lay.stage_h * 0.5
        axes = m.Axes(
            x_range=[x0, x1, (x1 - x0) / 4.0],
            y_range=[y0, y1, (y1 - y0) / 4.0],
            x_length=axes_w,
            y_length=axes_h,
            axis_config={"color": th.text_muted, "stroke_width": 2.2, "include_ticks": False, "tip_length": 0.16},
        ).move_to(np.array([0.0, stage_top - axes_h * 0.5 - 0.18, 0.0]))
        plot_bottom = axes.c2p(x0, y0)[1]

        landmarks = m.VGroup()
        for lbl in plan.x_labels[:3]:
            x_px = axes.c2p(lbl.value, 0.0)[0]
            guide = m.DashedLine(
                np.array([x_px, axes.c2p(0.0, y1 - (y1 - y0) * 0.08)[1], 0.0]),
                np.array([x_px, plot_bottom - 0.06, 0.0]),
                dash_length=0.10,
                color=th.surface_border,
                stroke_width=1.8,
            )
            tex = m.MathTex(lbl.tex, font_size=32, color=th.text_muted)
            fit_mobject(tex, max_width=axes_w / 4.0, min_height=lay.min_text_h * 0.95)
            tex.next_to(guide, m.DOWN, buff=0.08)
            # A label under the x-axis arrowhead would collide with it: drop it below the tip
            # (measured from the real tip mobject; it renders larger than `tip_length`)
            tips = [s for s in axes.x_axis.submobjects if "Tip" in type(s).__name__]
            if tips:
                tip = tips[-1]
                overlaps_x = tex.get_right()[0] > tip.get_left()[0] - 0.05 and tex.get_left()[0] < tip.get_right()[0] + 0.05
                if overlaps_x and tex.get_top()[1] > tip.get_bottom()[1] - 0.06:
                    tex.shift(m.DOWN * (tex.get_top()[1] - (tip.get_bottom()[1] - 0.08)))
            landmarks.add(guide, tex)

        # View window (graph/sequence): plot values y map to base-axes y through an affine map
        # a*y + b that can zoom per beat. Objects are rebuilt under the target map and morph,
        # while the live tracer reads the interpolated map, so everything moves in lockstep.
        def affine(v0: float, v1: float) -> tuple[float, float]:
            a = (y1 - y0) / (v1 - v0)
            return a, y0 - a * v0

        view: dict = {"from": affine(y0, y1), "to": affine(y0, y1), "win": (y0, y1)}
        zoom = m.ValueTracker(1.0)

        def cur_ab() -> tuple[float, float]:
            s_ = zoom.get_value()
            (a0, b0), (a1, b1) = view["from"], view["to"]
            return a0 + (a1 - a0) * s_, b0 + (b1 - b0) * s_

        def vy(y: float, final: bool = False) -> float:
            a, b = view["to"] if final else cur_ab()
            return a * y + b

        def c2p(x: float, y: float, final: bool = False) -> np.ndarray:
            return axes.c2p(x, vy(y, final))

        def win_pad() -> tuple[float, float]:
            w0, w1 = view["win"]
            pad = (w1 - w0) * 0.02
            return w0 + pad, w1 - pad

        # ------------------------------------------------------------------ math objects
        ref_fn = compile_expr(plan.reference_expr, ("x",)) if plan.reference_expr else None
        state: dict = {"fn": None, "fn2": None, "t_range": (0.0, 2.0 * np.pi), "bounds": (x0, x1), "n_max": 10,
                       "readout": plan.beats[0].visual_intent.readout}

        def scalar(fn, **kw) -> float:
            return float(fn(**kw))

        def plot_fn(fn, color: str, width: float = 4.2):
            xs = np.linspace(x0, x1, 900)
            run = largest_inside_run(xs, fn(x=xs), *win_pad())
            if run is None:
                return None, None
            graph = axes.plot(lambda x: vy(scalar(fn, x=x), final=True),
                              x_range=[run[0], run[1], (run[1] - run[0]) / 400.0], color=color, stroke_width=width)
            return graph, run

        def build_object(expr: str, color: str, vi) -> tuple[m.VMobject, dict]:
            if kind == "graph":
                fn = compile_expr(expr, ("x",))
                graph, run = plot_fn(fn, color)
                return graph, {"fn": fn, "bounds": run, "expr": expr}
            if kind == "parametric":
                fx = compile_expr(expr.split(";")[0], ("t",))
                fy = compile_expr(expr.split(";")[1], ("t",))
                t_lo, t_hi = vi.t_range or (0.0, 2.0 * np.pi)
                curve = m.ParametricFunction(
                    lambda t: axes.c2p(np.clip(scalar(fx, t=t), x0, x1), np.clip(scalar(fy, t=t), y0, y1)),
                    t_range=[t_lo, t_hi, (t_hi - t_lo) / 600.0],
                    color=color,
                    stroke_width=4.2,
                )
                return curve, {"fn": fx, "fn2": fy, "t_range": (t_lo, t_hi), "expr": expr}
            # sequence
            fn = compile_expr(expr, ("n",))
            n_max = int(vi.n_max or state["n_max"])
            ns = np.arange(1, n_max + 1, dtype=float)
            vals = fn(n=ns)
            lo_v, hi_v = win_pad()
            pts = [c2p(n, v, final=True) for n, v in zip(ns, vals)
                   if np.isfinite(v) and lo_v <= v <= hi_v and x0 <= n <= x1]
            dot_r = min(0.085, axes_w / max(n_max, 1) * 0.22)
            group = m.VGroup()
            if len(pts) >= 2:
                path = m.VMobject(color=color, stroke_width=2.2, stroke_opacity=0.55)
                path.set_points_as_corners(pts)
                group.add(path)
            group.add(*[m.Dot(p, radius=dot_r, color=color) for p in pts])
            return group, {"fn": fn, "n_max": n_max, "expr": expr}

        # ------------------------------------------------------------------ live tracer & readout
        def tracer_param() -> float:
            return np.sin(2.0 * np.pi * self.now() / TRACER_PERIOD_S)  # in [-1, 1]

        def tracer_point() -> tuple[np.ndarray, float, float] | None:
            """(scene point, primary value, slope-ish value) for the moving probe."""
            s = tracer_param()
            fn = state["fn"]
            if kind == "graph":
                active = fn or ref_fn
                if active is None:
                    return None
                lo, hi = state["bounds"] or (x0, x1)
                mid, half = (lo + hi) * 0.5, (hi - lo) * 0.5 * 0.9
                x = mid + half * s
                y = scalar(active, x=x)
                h = (x1 - x0) * 1e-4
                slope = (scalar(active, x=x + h) - scalar(active, x=x - h)) / (2 * h)
                return c2p(x, float(np.clip(y, *view["win"]))), y, slope
            if kind == "parametric":
                if fn is None:
                    return None
                t_lo, t_hi = state["t_range"]
                t = t_lo + (t_hi - t_lo) * (0.5 + 0.5 * s)
                x, y = scalar(fn, t=t), scalar(state["fn2"], t=t)
                h = (t_hi - t_lo) * 1e-4
                vel_x = (scalar(fn, t=t + h) - scalar(fn, t=t - h)) / (2 * h)
                vel_y = (scalar(state["fn2"], t=t + h) - scalar(state["fn2"], t=t - h)) / (2 * h)
                return axes.c2p(np.clip(x, x0, x1), np.clip(y, y0, y1)), t, float(np.hypot(vel_x, vel_y))
            # sequence: hop along the revealed terms
            if fn is None:
                return None
            ns = np.arange(1, max(1, state["n_max"]) + 1, dtype=float)
            vals = fn(n=ns)
            lo_v, hi_v = win_pad()
            visible = [int(n) for n, v in zip(ns, vals) if np.isfinite(v) and lo_v <= v <= hi_v and n <= x1]
            if not visible:
                return None
            n = visible[int(self.now() * 2.5) % len(visible)]
            v = scalar(fn, n=float(n))
            prev = scalar(fn, n=float(n - 1)) if n > 1 else v
            return c2p(n, v), v, v - prev

        def readout_value() -> float:
            got = tracer_point()
            if got is None:
                return 0.0
            p, value, slope = got
            mode = state["readout"]
            if mode == "slope":
                return slope
            if mode == "error" and ref_fn is not None:
                x_plot = axes.p2c(p)[0]
                return abs(value - scalar(ref_fn, x=x_plot))
            if kind == "parametric" and mode == "value":
                return value  # the parameter t
            return value

        prefix_for = {
            "graph": {"value": r"f(x)=", "slope": r"f'(x)=", "error": r"|\varepsilon|=", "none": r"x="},
            "parametric": {"value": r"t=", "slope": r"|v|=", "error": r"r=", "none": r"t="},
            "sequence": {"value": r"a_n=", "slope": r"\Delta a_n=", "error": r"|a_n-L|=", "none": r"a_n="},
        }[kind]

        badge_h = min(0.84, lay.stage_h * 0.15)
        # The badge hangs below whatever actually sits lowest under the plot: landmark labels,
        # the x-axis arrowhead, or the plot frame itself
        content_bottom = min([plot_bottom, axes.x_axis.get_bottom()[1]]
                             + ([landmarks.get_bottom()[1]] if len(landmarks) else []))
        label_floor = content_bottom - 0.14 - badge_h * 0.5
        badge = StatBadge(
            layout=lay,
            theme=th,
            center=np.array([0.0, max(stage_bot + badge_h * 0.5 + 0.06, label_floor), 0.0]),
            width=min(lay.stage_w * 0.94, 7.4),
            height=badge_h,
            label=plan.beats[0].visual_intent.badge_label,
            prefix_tex=prefix_for[state["readout"]],
            value_fn=readout_value,
            color=self.color(plan.beats[0].color_role),
            # Sequence values near a limit show their digits (0.99999999); gaps/errors go scientific
            formatter=lambda v: precise_tex(v) if (kind == "sequence" and state["readout"] == "value")
            else adaptive_tex(v, 4 if kind == "sequence" else 3),
            widest_samples=(r"8.8\times10^{-8}", "-8.888", "0.99999999" if kind == "sequence" else "0"),
            labels=tuple(b.visual_intent.badge_label for b in plan.beats),
            prefixes=tuple(prefix_for[b.visual_intent.readout] for b in plan.beats),
        )
        # A stacked (two-row) badge keeps its top edge where a one-row badge would sit
        badge_top = max(stage_bot + badge_h + 0.06, label_floor + badge_h * 0.5)
        badge.group.move_to(np.array([0.0, max(stage_bot + badge.bg.height * 0.5 + 0.06,
                                                badge_top - badge.bg.height * 0.5), 0.0]))

        def make_tracer() -> m.VGroup:
            got = tracer_point()
            if got is None:
                return m.VGroup()
            p, _, slope = got
            parts: list[m.Mobject] = []
            if kind == "graph":
                vec = c2p(1.0, slope) - c2p(0.0, 0.0)
                unit = vec / max(np.linalg.norm(vec), 1e-6)
                half = axes_w * 0.10
                parts.append(m.Line(p - unit * half, p + unit * half, color=th.accent_highlight, stroke_width=3.0))
                if state["readout"] == "error" and ref_fn is not None:
                    x_plot = axes.p2c(p)[0]
                    q = c2p(x_plot, float(np.clip(scalar(ref_fn, x=x_plot), *view["win"])))
                    if np.linalg.norm(q - p) > 0.02:
                        parts.append(m.Line(p, q, color=th.danger, stroke_width=3.5))
            elif kind == "parametric":
                base = axes.c2p(0.0, 0.0) if (x0 <= 0 <= x1 and y0 <= 0 <= y1) else axes.get_center()
                parts.append(m.Line(base, p, color=th.text_muted, stroke_width=1.6, stroke_opacity=0.6))
            else:
                w0, w1 = view["win"]
                floor_pt = np.array([p[0], c2p(x0, max(w0, min(0.0, w1)))[1], 0.0])
                parts.append(m.Line(floor_pt, p, color=th.accent_highlight, stroke_width=2.2, stroke_opacity=0.6))
            parts.append(m.Dot(p, radius=0.085, color=th.accent_highlight))
            return m.VGroup(*parts)

        tracer = m.always_redraw(make_tracer)

        # ------------------------------------------------------------------ frame 0: hook already on screen
        ref_graph = None
        if ref_fn is not None:
            ref_graph, _ = plot_fn(ref_fn, th.accent_primary, width=3.6)
        first = plan.beats[0]
        main = None
        main_glow = None
        if first.visual_intent.expr:
            main, info = build_object(first.visual_intent.expr, self.color(first.color_role), first.visual_intent)
            state.update(info)
            main_glow = glow(main) if kind != "sequence" else None

        statics: list[m.Mobject] = [grid]
        if ref_graph is not None:
            statics += [glow(ref_graph, layers=2), ref_graph]
        statics += [axes, landmarks]
        if main is not None:
            statics += [mob for mob in (main_glow, main) if mob is not None]
        statics += [badge.group, header.group, self.caption_card.group]
        self.add(*statics)

        frame = self.camera.frame  # type: ignore[attr-defined]
        frame.scale(0.86).move_to(axes.get_center())

        # ------------------------------------------------------------------ helpers for beats
        def sweep() -> list[m.Animation]:
            target = main if main is not None else ref_graph
            if target is None or kind == "sequence":
                return [m.Indicate(badge.label, color=th.accent_highlight)]
            flash = target.copy().set_stroke(color=th.accent_highlight, width=7.0, opacity=1.0)
            return [m.ShowPassingFlash(flash, time_width=0.35)]

        def cue_factory(action: str):
            def make() -> list[m.Animation]:
                if action == "flash_point":
                    got = tracer_point()
                    spot = m.Dot(got[0] if got else axes.get_center(), radius=0.01).set_opacity(0.0)
                    return [m.Flash(spot, color=th.accent_highlight, flash_radius=0.45)]
                if action == "indicate_formula":
                    return [m.Indicate(header.formula, scale_factor=1.08, color=th.accent_highlight)]
                if action == "pulse_badge":
                    return [m.Indicate(badge.label, color=th.accent_highlight)]
                return sweep()

            return make

        # ------------------------------------------------------------------ beats
        area_mob: m.Mobject | None = None
        for i, beat in enumerate(plan.beats):
            vi = beat.visual_intent
            color = self.color(beat.color_role)
            is_last = i == len(plan.beats) - 1
            anims: list[m.Animation] = []
            target_obj = main
            if i == 0:
                anims += [frame.animate.scale(1.0 / 0.86).move_to(np.zeros(3)), m.FadeIn(tracer, scale=0.6),
                          m.Indicate(header.formula, color=th.accent_highlight)]
            else:
                anims.append(m.Transform(header.formula, header.build_formula(beat.formula, color=color)))
                zooming = (vi.y_range is not None and kind in ("graph", "sequence")
                           and tuple(vi.y_range) != tuple(view["win"]))
                if zooming:
                    view["from"], view["to"], view["win"] = cur_ab(), affine(*vi.y_range), tuple(vi.y_range)
                    zoom.set_value(0.0)
                    anims.append(zoom.animate.set_value(1.0))
                    # Camera punch-in sells the dive (there-and-back, so framing is unchanged after)
                    anims.append(frame.animate(rate_func=m.there_and_back).scale(0.93))
                    if ref_graph is not None and ref_fn is not None:
                        new_ref, _ = plot_fn(ref_fn, th.accent_primary, width=3.6)
                        if new_ref is not None:
                            anims.append(m.Transform(ref_graph, new_ref))
                    if main is not None and not vi.expr:
                        rebuilt, info = build_object(state["expr"], color, vi)
                        state.update(info)
                        target_obj = rebuilt
                        anims.append(m.Transform(main, rebuilt))
                if vi.expr:
                    new_obj, info = build_object(vi.expr, color, vi)
                    target_obj = new_obj
                    if main is None:
                        main = new_obj
                        if kind != "sequence":
                            main_glow = glow(main)
                            self.add(main_glow)
                        anims.append(m.Create(main))
                    else:
                        anims.append(m.Transform(main, new_obj))
                    state.update(info)
                elif main is not None and not zooming:
                    anims.append(main.animate.set_color(color))
            state["readout"] = vi.readout
            anims += badge.swap(vi.badge_label, color, prefix_tex=prefix_for[vi.readout])

            # Error ribbon between the current curve and the reference (replaced beat to beat)
            if vi.show_error_area and target_obj is not None and ref_fn is not None and kind == "graph":
                lo, hi = state["bounds"]
                ref_sub = axes.plot(lambda x: scalar(ref_fn, x=x), x_range=[lo, hi], stroke_width=0.0)
                new_area = axes.get_area(ref_sub, x_range=[lo, hi], bounded_graph=target_obj,
                                         color=th.danger, opacity=0.22)
                if area_mob is None:
                    area_mob = new_area
                    anims.append(m.FadeIn(area_mob))
                else:
                    anims.append(m.Transform(area_mob, new_area))
            elif area_mob is not None:
                anims.append(m.FadeOut(area_mob))
                area_mob = None

            self.play_beat(
                beat.beat_id,
                *anims,
                caption_color=color if is_last else None,
                border_color=color,
                sfx=beat.sfx_cue,
                rate_func=m.ease_out_cubic if i == 0 else None,
                hold=sweep,
                cues={cue.mark: cue_factory(cue.action) for cue in vi.cues},
            )


class AIShort3D(_PlanMixin, Story3DScene):
    """3D template: surface plans z = f(x, y) with orbiting camera, gradient probe and slices."""

    def construct(self) -> None:
        self.setup_studio()
        plan, th, lay = self.PLAN, self.theme, self.layout
        header = self.chrome()
        (x0, x1), (y0, y1) = plan.x_range, plan.y_range
        z0, z1 = plan.z_range or [-2.0, 2.0]
        state: dict = {"fn": None, "readout": plan.beats[0].visual_intent.readout}

        def scalar(fn, **kw) -> float:
            return float(fn(**kw))

        def probe_xy() -> tuple[float, float]:
            a = 2.0 * np.pi * self.now() / 7.0
            rx, ry = (x1 - x0) * 0.30, (y1 - y0) * 0.30
            return (x0 + x1) * 0.5 + rx * np.cos(a), (y0 + y1) * 0.5 + ry * np.sin(a)

        def grad(fn, x: float, y: float) -> tuple[float, float]:
            h = 1e-4 * max(x1 - x0, y1 - y0)
            return ((scalar(fn, x=x + h, y=y) - scalar(fn, x=x - h, y=y)) / (2 * h),
                    (scalar(fn, x=x, y=y + h) - scalar(fn, x=x, y=y - h)) / (2 * h))

        def readout_value() -> float:
            fn = state["fn"]
            if fn is None:
                return 0.0
            x, y = probe_xy()
            if state["readout"] == "slope":
                return float(np.hypot(*grad(fn, x, y)))
            return float(np.clip(scalar(fn, x=x, y=y), z0, z1))

        prefix_for = {"value": r"f(x,y)=", "slope": r"|\nabla f|=", "error": r"f(x,y)=", "none": r"f(x,y)="}
        stage_bot = lay.stage_center[1] - lay.stage_h * 0.5
        badge_h = min(0.84, lay.stage_h * 0.13)
        badge = StatBadge(
            layout=lay,
            theme=th,
            center=np.array([0.0, stage_bot + badge_h * 0.5 + 0.06, 0.0]),
            width=min(lay.stage_w * 0.94, 7.4),
            height=badge_h,
            label=plan.beats[0].visual_intent.badge_label,
            prefix_tex=prefix_for[state["readout"]],
            value_fn=readout_value,
            color=self.color(plan.beats[0].color_role),
        )
        self.add_fixed_in_frame_mobjects(header.group, self.caption_card.group, badge.group)
        self.set_camera_orientation(phi=66 * m.DEGREES, theta=-45 * m.DEGREES)

        span = min(lay.stage_w, lay.stage_h) * 0.68
        axes3d = m.ThreeDAxes(
            x_range=[x0, x1, (x1 - x0) / 4.0], y_range=[y0, y1, (y1 - y0) / 4.0], z_range=[z0, z1, (z1 - z0) / 4.0],
            x_length=span, y_length=span, z_length=span * 0.75,
        )
        floor = m.NumberPlane(
            x_range=[x0, x1, (x1 - x0) / 8.0], y_range=[y0, y1, (y1 - y0) / 8.0],
            x_length=span * 1.15, y_length=span * 1.15,
            background_line_style={"stroke_color": th.surface_border, "stroke_width": 1.4, "stroke_opacity": 0.55},
            faded_line_ratio=2, axis_config={"stroke_opacity": 0.0, "stroke_width": 0.0},
        ).move_to(axes3d.c2p((x0 + x1) / 2, (y0 + y1) / 2, z0))
        shift = np.array([0.0, lay.stage_center[1] * 0.35, 0.0])

        def build_surface(expr: str, role: str) -> tuple[m.Surface, object]:
            fn = compile_expr(expr, ("x", "y"))
            surf = m.Surface(
                lambda u, v: axes3d.c2p(u, v, float(np.clip(scalar(fn, x=u, y=v), z0, z1))),
                u_range=[x0 * 0.85, x1 * 0.85] if x0 < 0 < x1 else [x0, x1],
                v_range=[y0 * 0.85, y1 * 0.85] if y0 < 0 < y1 else [y0, y1],
                resolution=(24, 24), fill_opacity=0.68, stroke_width=0.6,
                checkerboard_colors=[self.color(role), th.surface_border],
            ).shift(shift)
            return surf, fn

        def make_probe() -> m.VGroup:
            fn = state["fn"]
            if fn is None:
                return m.VGroup()
            x, y = probe_xy()
            z = float(np.clip(scalar(fn, x=x, y=y), z0, z1))
            gx, gy = grad(fn, x, y)
            p = axes3d.c2p(x, y, z)
            tip = axes3d.c2p(x + 0.3 * gx, y + 0.3 * gy, z + 0.3 * (gx * gx + gy * gy))
            d = tip - p
            d = d / max(np.linalg.norm(d), 1e-6) * span * 0.16
            return m.VGroup(m.Line(p, p + d, color=th.accent_highlight, stroke_width=4.0),
                            m.Dot3D(p, radius=0.07, color=th.accent_highlight)).shift(shift)

        probe = m.always_redraw(make_probe)
        m.Group(floor, axes3d).shift(shift)
        origin = m.Dot3D(axes3d.c2p((x0 + x1) / 2, (y0 + y1) / 2, 0.0), radius=0.08, color=th.accent_highlight)

        surface = None
        first = plan.beats[0]
        statics: list[m.Mobject] = [floor, axes3d, origin]
        if first.visual_intent.expr:
            surface, state["fn"] = build_surface(first.visual_intent.expr, first.color_role)
            statics.append(surface)
        self.add(*statics)
        self.begin_ambient_camera_rotation(rate=0.22)

        def slice_curve(fn, axis: str, color: str) -> m.ParametricFunction:
            if axis == "x":
                f3 = lambda t: axes3d.c2p(t, 0.0, float(np.clip(scalar(fn, x=t, y=0.0), z0, z1)))
                rng = [x0 * 0.85, x1 * 0.85]
            else:
                f3 = lambda t: axes3d.c2p(0.0, t, float(np.clip(scalar(fn, x=0.0, y=t), z0, z1)))
                rng = [y0 * 0.85, y1 * 0.85]
            return m.ParametricFunction(f3, t_range=[rng[0], rng[1], (rng[1] - rng[0]) / 80],
                                        color=color, stroke_width=6.0).shift(shift)

        def cue_factory(action: str):
            def make() -> list[m.Animation]:
                if action in ("indicate_formula", "sweep_curve"):
                    return [m.Indicate(header.formula, scale_factor=1.08, color=th.accent_highlight)]
                if action == "pulse_badge":
                    return [m.Indicate(badge.label, color=th.accent_highlight)]
                return [m.Flash(origin, color=th.accent_highlight)]

            return make

        for i, beat in enumerate(plan.beats):
            vi = beat.visual_intent
            color = self.color(beat.color_role)
            anims: list[m.Animation] = []
            if i == 0:
                anims += [m.Flash(origin, color=th.accent_highlight), m.Indicate(header.formula, color=th.accent_highlight)]
                if surface is not None:
                    anims.append(m.FadeIn(probe))
            else:
                anims.append(m.Transform(header.formula, header.build_formula(beat.formula, color=color)))
                if vi.expr:
                    new_surface, fn = build_surface(vi.expr, beat.color_role)
                    if surface is None:
                        surface = new_surface
                        anims += [m.Create(surface), m.FadeIn(probe)]
                    else:
                        anims.append(m.Transform(surface, new_surface))
                    state["fn"] = fn
            if vi.slice_axis and state["fn"] is not None:
                anims.append(m.Create(slice_curve(state["fn"], vi.slice_axis, color)))
            state["readout"] = vi.readout
            anims += badge.swap(vi.badge_label, color, prefix_tex=prefix_for[vi.readout])
            self.play_beat(
                beat.beat_id,
                *anims,
                caption_color=color if i == len(plan.beats) - 1 else None,
                border_color=color,
                sfx=beat.sfx_cue,
                hold=lambda: [m.Indicate(header.formula, scale_factor=1.04, color=th.accent_highlight)],
                cues={cue.mark: cue_factory(cue.action) for cue in vi.cues},
            )
        self.stop_ambient_camera_rotation()


class AIShortPolytope(_PlanMixin, Story3DScene):
    """3D template for 4D polytopes: true R^4 rotations, perspective-projected to R^3 ("the shadow").

    Edges are classified every frame by their rotated w-coordinates: both ends at w < 0 form the
    inner cell, both at w > 0 the outer cell, mixed ones are connectors, so rotating through the
    4th dimension visibly turns the shape inside out. Brightness tracks nearness to the 4D eye.
    """

    def construct(self) -> None:
        from studio.geometry4d import max_projected_radius, plane_rotation, polytope

        self.setup_studio()
        plan, th, lay = self.PLAN, self.theme, self.layout
        header = self.chrome()
        verts, edges = polytope(plan.polytope or "tesseract")

        first_vi = plan.beats[0].visual_intent
        state: dict = {
            "M": np.eye(4),
            "plane": first_vi.rotation_plane or "xw",
            "speed": 0.0,
            "target_speed": float(first_vi.rotation_speed if first_vi.rotation_speed is not None else 0.08),
            "angle": 0.0,
            "last_t": 0.0,
            "highlight": first_vi.highlight or "none",
            "readout": first_vi.readout,
        }

        def step() -> np.ndarray:
            """Advance the 4D rotation to the current scene time; returns rotated vertices."""
            t = self.now()
            dt = max(0.0, t - state["last_t"])
            state["last_t"] = t
            if dt > 0:
                # Speed eases toward its target (no jerks when a beat changes it)
                state["speed"] += (state["target_speed"] - state["speed"]) * (1.0 - np.exp(-dt / 0.4))
                d_angle = 2.0 * np.pi * state["speed"] * dt
                state["M"] = plane_rotation(state["plane"], d_angle) @ state["M"]
                state["angle"] += d_angle
            return verts @ state["M"].T

        # The shadow stays inside a sphere of radius span/2 for every 4D rotation, so it can fill the stage
        span = min(lay.stage_w * 0.90, lay.stage_h * 0.78)
        scale = span * 0.5 / max_projected_radius(verts)
        shift = np.array([0.0, lay.stage_center[1] * 0.35, 0.0])
        role_color = {"inner": th.accent_secondary, "outer": th.accent_primary, "connectors": th.accent_tertiary}

        def make_shadow() -> m.VGroup:
            rotated = step()
            w = rotated[:, 3]
            pts = rotated[:, :3] / (3.0 - w)[:, None] * scale + shift
            group = m.VGroup()
            hl = state["highlight"]
            for i, j in edges:
                kind = "inner" if (w[i] < 0 and w[j] < 0) else "outer" if (w[i] > 0 and w[j] > 0) else "connectors"
                near = float(np.clip((w[i] + w[j]) * 0.25 + 0.5, 0.0, 1.0))  # 0 far .. 1 near the 4D eye
                focus = hl == "none" or hl == kind
                group.add(
                    m.Line(
                        pts[i], pts[j], color=role_color[kind],
                        stroke_width=(6.0 if hl == kind else 3.4),
                        stroke_opacity=(0.45 + 0.55 * near) if focus else 0.22,
                    )
                )
            for k in range(len(pts)):
                group.add(m.Dot3D(pts[k], radius=0.045, resolution=(6, 6), color=th.text_primary))
            return group

        def readout_value() -> float:
            if state["readout"] == "slope":
                return 360.0 * state["speed"]
            return float(np.degrees(state["angle"]) % 360.0)

        prefix_for = {"value": r"\theta=", "slope": r"\omega=", "error": r"\theta=", "none": r"\theta="}
        stage_bot = lay.stage_center[1] - lay.stage_h * 0.5
        badge_h = min(0.84, lay.stage_h * 0.13)
        badge = StatBadge(
            layout=lay,
            theme=th,
            center=np.array([0.0, stage_bot + badge_h * 0.5 + 0.06, 0.0]),
            width=min(lay.stage_w * 0.94, 7.4),
            height=badge_h,
            label=first_vi.badge_label,
            prefix_tex=prefix_for[state["readout"]],
            value_fn=readout_value,
            color=self.color(plan.beats[0].color_role),
            decimals=1,
            labels=tuple(b.visual_intent.badge_label for b in plan.beats),
            prefixes=tuple(prefix_for[b.visual_intent.readout] for b in plan.beats),
        )
        badge.group.move_to(np.array([0.0, stage_bot + badge.bg.height * 0.5 + 0.06, 0.0]))
        self.add_fixed_in_frame_mobjects(header.group, self.caption_card.group, badge.group)
        self.set_camera_orientation(phi=68 * m.DEGREES, theta=-38 * m.DEGREES)

        shadow = m.always_redraw(make_shadow)
        core = m.Dot3D(shift, radius=0.06, color=th.accent_highlight)
        self.add(shadow, core)  # frame 0: the full shadow is already on screen
        self.begin_ambient_camera_rotation(rate=0.12)

        def cue_factory(action: str):
            def make() -> list[m.Animation]:
                if action == "pulse_badge":
                    return [m.Indicate(badge.label, color=th.accent_highlight)]
                if action == "flash_point":
                    return [m.Flash(core, color=th.accent_highlight, flash_radius=span * 0.3)]
                return [m.Indicate(header.formula, scale_factor=1.08, color=th.accent_highlight)]

            return make

        for i, beat in enumerate(plan.beats):
            vi = beat.visual_intent
            color = self.color(beat.color_role)
            anims: list[m.Animation] = []
            if i == 0:
                anims += [m.Flash(core, color=th.accent_highlight, flash_radius=span * 0.3),
                          m.Indicate(header.formula, color=th.accent_highlight)]
            else:
                anims.append(m.Transform(header.formula, header.build_formula(beat.formula, color=color)))
            if vi.rotation_plane:
                state["plane"] = vi.rotation_plane
            if vi.rotation_speed is not None:
                state["target_speed"] = float(np.clip(vi.rotation_speed, 0.0, 0.5))
            state["highlight"] = vi.highlight or "none"
            state["readout"] = vi.readout
            anims += badge.swap(vi.badge_label, color, prefix_tex=prefix_for[vi.readout])
            self.play_beat(
                beat.beat_id,
                *anims,
                caption_color=color if i == len(plan.beats) - 1 else None,
                border_color=color,
                sfx=beat.sfx_cue,
                hold=lambda: [m.Indicate(header.formula, scale_factor=1.04, color=th.accent_highlight)],
                cues={cue.mark: cue_factory(cue.action) for cue in vi.cues},
            )
        self.stop_ambient_camera_rotation()
