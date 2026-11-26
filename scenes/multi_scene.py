"""Multi-scene composable ManimGX renderer for VideoPlan V2/V2.1."""

from __future__ import annotations

import os
from pathlib import Path
import manimgx as m
import numpy as np

from studio.ai.expr import compile_expr
from studio.ai.multi_schema import VideoPlan
from studio.audio.tts import BeatSpec
from studio.config import CAPTION_FONT
from studio.layout import CaptionCard, fit_mobject, glow
from studio.scene import StoryScene

DEFAULT_V2_PLAN = Path(__file__).resolve().parent.parent / "plans" / "fourier_v2_benchmark.json"
V2_PLAN_PATH = Path(os.environ.get("STUDIO_V2_PLAN", str(DEFAULT_V2_PLAN)))
V2_PLAN = VideoPlan.model_validate_json(V2_PLAN_PATH.read_text(encoding="utf-8"))

ROLE_TO_THEME = {
    "primary": "accent_primary",
    "secondary": "accent_secondary",
    "tertiary": "accent_tertiary",
    "success": "success",
    "highlight": "accent_highlight",
    "danger": "danger",
}


def plot_clipped_segments(
    ax: m.Axes,
    fn,
    x_range: list[float],
    y_range: list[float],
    color: str,
    stroke_width: float = 3.5,
    n_samples: int = 360,
) -> m.VGroup:
    """Plot y=f(x) splitting at viewport bounds and jump discontinuities (never np.clip flat lines)."""
    xs = np.linspace(x_range[0], x_range[1], n_samples)
    ys = np.asarray(fn(x=xs), dtype=float)
    y_lo, y_hi = y_range[0], y_range[1]
    jump_thresh = 0.65 * (y_hi - y_lo)

    segments: list[list[np.ndarray]] = []
    cur: list[np.ndarray] = []
    prev_y: float | None = None
    for x_val, y_val in zip(xs, ys):
        in_bounds = bool(np.isfinite(y_val) and (y_lo <= y_val <= y_hi))
        is_jump = prev_y is not None and abs(float(y_val) - prev_y) > jump_thresh
        if in_bounds and not is_jump:
            cur.append(ax.c2p(float(x_val), float(y_val)))
            prev_y = float(y_val)
        else:
            if len(cur) >= 2:
                segments.append(cur)
            cur = []
            prev_y = float(y_val) if in_bounds else None
    if len(cur) >= 2:
        segments.append(cur)

    grp = m.VGroup()
    for pts in segments:
        curve = m.VMobject()
        curve.set_points_smoothly(pts)
        curve.set_stroke(color=color, width=stroke_width)
        grp.add(curve)
    return grp


class MultiSceneVideo(StoryScene):
    """Composable multi-scene educational video renderer driven by VideoPlan V2/V2.1."""

    BEATS = [
        BeatSpec(id=s.scene_id, narration=s.narration, caption=s.purpose, pause_after=0.30)
        for s in V2_PLAN.scenes
    ]

    def construct(self) -> None:
        self.setup_studio()
        L, T = self.layout, self.theme
        self.caption_card = CaptionCard(L, T, V2_PLAN.scenes[0].narration)
        self.add(self.caption_card.group)

        stage_group: m.VGroup | None = None
        formula_mob: m.MathTex | None = None

        for sc in V2_PLAN.scenes:
            color = getattr(T, ROLE_TO_THEME[sc.color_role])
            new_stage, anim_mode, anim_targets = self._build_primitive(sc, color)
            anims: list[m.Animation] = []
            if stage_group is not None:
                anims.append(m.FadeOut(stage_group))
            if anim_mode == "grow_bars" and anim_targets:
                anims.append(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in anim_targets], lag_ratio=0.12))
                for extra in new_stage[1:]:
                    anims.append(m.FadeIn(extra))
            elif anim_mode == "create_curves" and anim_targets:
                anims.append(m.FadeIn(new_stage[0]))
                for t_mob in anim_targets:
                    anims.append(m.Create(t_mob))
            else:
                anims.append(m.FadeIn(new_stage))

            if sc.formula:
                new_f = m.MathTex(sc.formula, font_size=40, color=color)
                fit_mobject(new_f, max_width=L.safe_w * 0.90, max_height=L.header_max_h * 0.65)
                new_f.move_to(L.header_center)
                if formula_mob is not None:
                    anims.append(m.Transform(formula_mob, new_f))
                else:
                    formula_mob = new_f
                    anims.append(m.FadeIn(formula_mob))
            elif formula_mob is not None:
                anims.append(m.FadeOut(formula_mob))
                formula_mob = None

            self.play_beat(sc.scene_id, *anims, sfx=sc.sfx_cue)
            stage_group = new_stage

    def _axes(self, y_range: list[float] | None = None) -> m.Axes:
        L, T = self.layout, self.theme
        xr = V2_PLAN.x_range
        yr = y_range or V2_PLAN.y_range
        return m.Axes(
            x_range=[xr[0], xr[1], (xr[1] - xr[0]) / 4.0],
            y_range=[yr[0], yr[1], (yr[1] - yr[0]) / 4.0],
            x_length=L.stage_w * 0.92,
            y_length=L.stage_h * 0.68,
            axis_config={"color": T.text_muted, "stroke_width": 1.6, "include_tip": True},
            tips=True,
        ).move_to(L.stage_center)

    def _build_bars(self, values: list[float], labels: list[str], color: str, log_scale: bool) -> tuple[m.VGroup, str, list[m.Mobject]]:
        L, T = self.layout, self.theme
        if log_scale:
            # Honest logarithmic magnitude mapping above min_val
            logs = [float(np.log10(max(v, 1e-12))) for v in values]
            lo = min(logs) - 0.5
            hi = max(logs)
            span = max(hi - lo, 1e-6)
            norm = [(v - lo) / span for v in logs]
        else:
            hi = max(values) if values else 1.0
            norm = [max(0.0, v) / max(hi, 1e-6) for v in values]

        bars = m.VGroup()
        lbl_grp = m.VGroup()
        n = len(norm)
        w = L.stage_w * 0.78 / max(1, n)
        base_y = L.stage_center[1] - L.stage_h * 0.22
        max_h = L.stage_h * 0.48
        for i, frac in enumerate(norm):
            h = max(0.03, frac * max_h)
            bar = m.Rectangle(width=w * 0.62, height=h, fill_color=color, fill_opacity=0.88, stroke_width=0)
            x_pos = L.stage_center[0] - L.stage_w * 0.39 + (i + 0.5) * w
            bar.move_to(np.array([x_pos, base_y + h * 0.5, 0.0]))
            bars.add(bar)
            if i < len(labels) and labels[i]:
                lbl = m.Text(labels[i], font_size=24, font=CAPTION_FONT, color=T.text_muted)
                fit_mobject(lbl, max_width=w * 1.18, min_height=L.min_text_h * 1.05)
                lbl.next_to(bar, m.DOWN, buff=0.14)
                lbl_grp.add(lbl)
        return m.VGroup(bars, lbl_grp), "grow_bars", list(bars)

    def _build_primitive(self, sc, color: str) -> tuple[m.VGroup, str, list[m.Mobject]]:
        T = self.theme
        prim = sc.primitive

        if prim.kind == "bar_chart" and prim.values:
            return self._build_bars(prim.values, prim.bar_labels, color, prim.log_scale)
        if prim.kind == "spectrum_bars" and prim.harmonics:
            vals = [1.0 / k for k in prim.harmonics]
            lbls = prim.bar_labels or [str(k) for k in prim.harmonics]
            return self._build_bars(vals, lbls, color, False)

        yr = sc.y_range or V2_PLAN.y_range
        ax = self._axes(yr)
        group = m.VGroup(ax)
        curves: list[m.Mobject] = []

        if prim.expr:
            fn = compile_expr(prim.expr, ("x",))
            seg_grp = plot_clipped_segments(ax, fn, V2_PLAN.x_range, yr, color=color, stroke_width=3.5)
            group.add(glow(seg_grp, color=color), seg_grp)
            curves.append(seg_grp)
            if prim.highlight_x is not None:
                hx = float(prim.highlight_x)
                hy = float(fn(x=hx))
                if np.isfinite(hy) and yr[0] <= hy <= yr[1]:
                    dot = m.Dot(ax.c2p(hx, hy), radius=0.09, color=T.accent_highlight)
                    group.add(dot)

        if prim.secondary_expr:
            fn2 = compile_expr(prim.secondary_expr, ("x",))
            c2 = T.accent_primary if color != T.accent_primary else T.accent_secondary
            seg2 = plot_clipped_segments(ax, fn2, V2_PLAN.x_range, yr, color=c2, stroke_width=2.4)
            group.add(seg2)
            curves.append(seg2)

        if prim.reference_y is not None and yr[0] <= prim.reference_y <= yr[1]:
            p_left = ax.c2p(V2_PLAN.x_range[0], prim.reference_y)
            p_right = ax.c2p(V2_PLAN.x_range[1], prim.reference_y)
            group.add(m.DashedLine(p_left, p_right, color=T.accent_highlight, stroke_width=2.0))

        return group, "create_curves", curves
