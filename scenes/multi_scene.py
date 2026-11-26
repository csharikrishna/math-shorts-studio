"""Multi-scene composable ManimGX renderer for VideoPlan V2."""

from __future__ import annotations

import os
from pathlib import Path
import manimgx as m
import numpy as np

from studio.ai.expr import compile_expr
from studio.ai.multi_schema import VideoPlan
from studio.audio.tts import BeatSpec
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


class MultiSceneVideo(StoryScene):
    """Composable multi-scene educational video renderer driven by VideoPlan V2."""

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

        for idx, sc in enumerate(V2_PLAN.scenes):
            color = getattr(T, ROLE_TO_THEME[sc.color_role])
            new_stage = self._build_primitive(sc, color)
            anims: list[m.Animation] = []
            if stage_group is not None:
                anims.append(m.FadeOut(stage_group))
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

    def _build_primitive(self, sc, color: str) -> m.VGroup:
        T = self.theme
        prim = sc.primitive
        ax = self._axes(sc.y_range)
        group = m.VGroup(ax)

        yr = sc.y_range or V2_PLAN.y_range
        if prim.expr:
            fn = compile_expr(prim.expr, ("x",))
            curve = ax.plot(
                lambda x, f=fn, lo=yr[0], hi=yr[1]: float(np.clip(f(x=x), lo, hi)),
                x_range=V2_PLAN.x_range,
                color=color,
                stroke_width=3.5,
            )
            group.add(glow(curve, color=color), curve)

        if prim.secondary_expr:
            fn2 = compile_expr(prim.secondary_expr, ("x",))
            curve2 = ax.plot(
                lambda x, f=fn2, lo=yr[0], hi=yr[1]: float(np.clip(f(x=x), lo, hi)),
                x_range=V2_PLAN.x_range,
                color=T.accent_primary if color != T.accent_primary else T.accent_secondary,
                stroke_width=2.5,
            )
            group.add(curve2)

        if prim.kind == "spectrum_bars" and prim.harmonics:
            bars = m.VGroup()
            n = len(prim.harmonics)
            w = self.layout.stage_w * 0.70 / max(1, n)
            for i, k in enumerate(prim.harmonics):
                h = (1.0 / k) * self.layout.stage_h * 0.45
                bar = m.Rectangle(width=w * 0.65, height=max(0.08, h), fill_color=color, fill_opacity=0.85, stroke_width=0)
                x_pos = self.layout.stage_center[0] - self.layout.stage_w * 0.35 + (i + 0.5) * w
                y_pos = self.layout.stage_center[1] - self.layout.stage_h * 0.22 + h * 0.5
                bar.move_to(np.array([x_pos, y_pos, 0.0]))
                bars.add(bar)
            return m.VGroup(bars)

        return group
