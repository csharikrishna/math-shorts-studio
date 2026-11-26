"""Higher-Order Calculus Showcase: Taylor Series Reconstruction of sin(x)."""

from __future__ import annotations

import math
import manimgx as m
import numpy as np
from studio.audio.tts import BeatSpec
from studio.layout import CaptionCard, HeaderBanner, StatBadge, ambient_grid, fit_mobject, glow
from studio.scene import StoryScene


def taylor_sin(x: float | np.ndarray, max_power: int) -> float | np.ndarray:
    """Compute Maclaurin polynomial for sin(x) up to odd power `max_power`."""
    total = np.zeros_like(x, dtype=float)
    sign = 1.0
    for k in range(1, max_power + 1, 2):
        total = total + sign * (x ** k) / math.factorial(k)
        sign = -sign
    return total


def safe_x_bounds(
    fn,
    x_min: float,
    x_max: float,
    y_limit: float,
    samples: int = 400,
) -> list[float]:
    """Find symmetric/contiguous [lo, hi] where |fn(x)| <= y_limit so curves never exit the plot frame."""
    xs = np.linspace(x_min, x_max, samples)
    ys = np.abs(fn(xs))
    # Walk outward from x=0 so we keep the main connected branch inside the box
    mid = samples // 2
    lo_idx = mid
    while lo_idx > 0 and ys[lo_idx - 1] <= y_limit:
        lo_idx -= 1
    hi_idx = mid
    while hi_idx < samples - 1 and ys[hi_idx + 1] <= y_limit:
        hi_idx += 1
    return [float(xs[lo_idx]), float(xs[hi_idx])]


class TaylorSeriesCalculus(StoryScene):
    """Visualizes how higher-order derivatives at x=0 reconstruct sin(x) across space."""

    BEATS = [
        BeatSpec(
            id="hook",
            narration="How can simple polynomials draw an endless smooth wave like sine of x?",
            caption="How do simple polynomials rebuild sin(x)?",
            pause_after=0.25,
        ),
        BeatSpec(
            id="order_1",
            narration="Match the first derivative at zero, and you only get a straight tangent line.",
            caption="Order 1 matches slope at x = 0",
            pause_after=0.25,
        ),
        BeatSpec(
            id="order_3",
            narration="Add the third derivative, and the curve bends to hug the first wave.",
            caption="Order 3 bends with curvature",
            pause_after=0.25,
        ),
        BeatSpec(
            id="order_5",
            narration="Push to the fifth derivative, and it wraps tightly around the peaks.",
            caption="Order 5 locks onto the peaks",
            pause_after=0.25,
        ),
        BeatSpec(
            id="order_7",
            narration="At order seven, each higher derivative pulls the error closer to zero.",
            caption="Order 7 crushes approximation error",
            pause_after=0.25,
        ),
        BeatSpec(
            id="order_11",
            narration="By order eleven, the polynomial and the sine wave become virtually identical!",
            caption="Order 11 matches sin(x) across [-π, π]",
            pause_after=0.30,
        ),
        BeatSpec(
            id="outro",
            narration="Every higher-order derivative carries the DNA of the entire function.",
            caption="Higher derivatives encode the whole curve.",
            pause_after=0.35,
        ),
    ]

    def construct(self) -> None:
        self.setup_studio()
        th = self.theme
        lay = self.layout

        # ------------------------------------------------------------------
        # 1. UI chrome + living backdrop
        # ------------------------------------------------------------------
        grid = ambient_grid(lay, th)
        header = HeaderBanner(
            layout=lay,
            theme=th,
            tag_text="Higher-Order Calculus",
            title_text="Taylor Series Expansion",
            initial_formula=r"f(x) = \sin(x)",
        )
        self.caption_card = CaptionCard(layout=lay, theme=th, initial_text=self.BEATS[0].narration)

        # ------------------------------------------------------------------
        # 2. Coordinate system inside the safe stage (aspect-ratio agnostic)
        # ------------------------------------------------------------------
        x_lim = 4.2
        y_lim = 2.1
        axes_h = lay.stage_h * 0.58
        axes_w = min(lay.stage_w * 0.92, axes_h * 1.95)
        stage_top_y = lay.stage_center[1] + lay.stage_h * 0.5
        stage_bot_y = lay.stage_center[1] - lay.stage_h * 0.5

        axes = m.Axes(
            x_range=[-x_lim, x_lim, np.pi / 2],
            y_range=[-y_lim, y_lim, 1.0],
            x_length=axes_w,
            y_length=axes_h,
            axis_config={
                "color": th.text_muted,
                "stroke_width": 2.2,
                "include_ticks": False,
                "tip_length": 0.16,
            },
        ).move_to(np.array([0.0, stage_top_y - axes_h * 0.5 - 0.18, 0.0]))

        pi_markers = m.VGroup()
        plot_bottom_y = axes.c2p(0.0, -y_lim)[1]
        for val, tex in [(-np.pi, r"-\pi"), (np.pi, r"\pi")]:
            x_px = axes.c2p(val, 0.0)[0]
            guide = m.DashedLine(
                np.array([x_px, axes.c2p(0.0, y_lim * 0.85)[1], 0.0]),
                np.array([x_px, plot_bottom_y - 0.06, 0.0]),
                dash_length=0.10,
                color=th.surface_border,
                stroke_width=1.8,
            )
            lbl = m.MathTex(tex, font_size=32, color=th.text_muted)
            fit_mobject(lbl, min_height=lay.min_text_h * 0.95)
            lbl.next_to(guide, m.DOWN, buff=0.08)
            pi_markers.add(guide, lbl)

        # ------------------------------------------------------------------
        # 3. Live tracer state: a point sweeping x, with slope and error readouts
        # ------------------------------------------------------------------
        state = {"order": 0, "amp": np.pi * 0.85, "t": 0.0}
        sweep_period = 6.0

        def poly(x: float) -> float:
            return float(taylor_sin(np.array(x), state["order"])) if state["order"] else float(np.sin(x))

        def tracer_x() -> float:
            t = self.now()
            # Ease the sweep amplitude toward the current polynomial's in-frame bounds
            if state["order"]:
                fn = lambda xs, p=state["order"]: taylor_sin(xs, p)
                lo, hi = safe_x_bounds(fn, -4.0, 4.0, y_limit=y_lim * 0.92)
                target = min(np.pi, -lo, hi) * 0.88
            else:
                target = np.pi * 0.85
            k = 1.0 - np.exp(-max(0.0, t - state["t"]) * 3.0)
            state["amp"] += (target - state["amp"]) * k
            state["t"] = t
            return state["amp"] * np.sin(2.0 * np.pi * t / sweep_period)

        def readout_value() -> float:
            x = tracer_x()
            return abs(np.sin(x) - poly(x)) if state["order"] else float(np.cos(x))

        badge_h = min(0.84, lay.stage_h * 0.15)
        badge_y = max(stage_bot_y + badge_h * 0.5 + 0.06, plot_bottom_y - 0.98)
        badge = StatBadge(
            layout=lay,
            theme=th,
            center=np.array([0.0, badge_y, 0.0]),
            width=min(lay.stage_w * 0.94, 7.4),
            height=badge_h,
            label="Target: sin(x)",
            prefix_tex=r"\text{slope}=",
            value_fn=readout_value,
            color=th.accent_primary,
        )

        sin_curve = axes.plot(lambda x: np.sin(x), x_range=[-4.0, 4.0], color=th.accent_primary, stroke_width=4.5)
        sin_glow = glow(sin_curve)
        anchor_dot = m.Dot(axes.c2p(0.0, 0.0), radius=0.09, color=th.accent_highlight)

        tangent_len = axes_w * 0.20

        def make_tracer() -> m.VGroup:
            x = tracer_x()
            p_sin = axes.c2p(x, np.sin(x))
            slope_vec = axes.c2p(1.0, np.cos(x)) - axes.c2p(0.0, 0.0)
            unit = slope_vec / max(np.linalg.norm(slope_vec), 1e-6)
            tangent = m.Line(
                p_sin - unit * tangent_len * 0.5,
                p_sin + unit * tangent_len * 0.5,
                color=th.accent_highlight,
                stroke_width=3.0,
            )
            parts = [tangent]
            if state["order"]:
                p_poly = axes.c2p(x, poly(x))
                if np.linalg.norm(p_poly - p_sin) > 0.02:
                    parts.append(m.Line(p_sin, p_poly, color=th.danger, stroke_width=3.5))
                parts.append(m.Dot(p_poly, radius=0.075, color=th.text_primary))
            parts.append(m.Dot(p_sin, radius=0.085, color=th.accent_highlight))
            return m.VGroup(*parts)

        tracer = m.always_redraw(make_tracer)

        def build_approx(order: int, color: str) -> tuple[m.Mobject, m.Mobject]:
            fn = lambda x, p=order: taylor_sin(x, p)
            x_bounds = safe_x_bounds(fn, -4.0, 4.0, y_limit=y_lim * 0.92)
            poly_graph = axes.plot(fn, x_range=x_bounds, color=color, stroke_width=4.2)
            sin_sub = axes.plot(lambda x: np.sin(x), x_range=x_bounds, color=th.accent_primary, stroke_width=0.0)
            err_area = axes.get_area(sin_sub, x_range=x_bounds, bounded_graph=poly_graph, color=th.danger, opacity=0.22)
            return poly_graph, err_area

        def sweep(curve: m.VMobject, color: str) -> list[m.Animation]:
            """Highlight pass: a bright pulse runs along the curve during the beat tail."""
            flash = curve.copy().set_stroke(color=color, width=curve.get_stroke_width() * 1.8, opacity=1.0)
            return [m.ShowPassingFlash(flash, time_width=0.35)]

        # ------------------------------------------------------------------
        # 4. Frame 0: the full hook is on screen before the first word
        # ------------------------------------------------------------------
        self.add(grid, sin_glow, axes, pi_markers, sin_curve, anchor_dot, badge.group, header.group,
                 self.caption_card.group)
        frame = self.camera.frame  # type: ignore[attr-defined]
        frame.scale(0.86).move_to(axes.get_center())

        self.play_beat(
            "hook",
            frame.animate.scale(1.0 / 0.86).move_to(np.zeros(3)),
            m.FadeIn(tracer, scale=0.6),
            m.Indicate(header.formula, color=th.accent_highlight),
            rate_func=m.ease_out_cubic,
            hold=lambda: sweep(sin_curve, th.accent_highlight),
            sfx="whoosh_soft",
        )

        # Order 1
        p1, err1 = build_approx(1, th.accent_secondary)
        p_glow = glow(p1)
        state["order"] = 1
        self.add(p_glow)
        self.play_beat(
            "order_1",
            m.Transform(header.formula, header.build_formula(r"P_1(x) = x", color=th.accent_secondary)),
            m.Create(p1),
            m.FadeIn(err1),
            *badge.swap("Order 1", th.accent_secondary, prefix_tex=r"|\varepsilon|="),
            border_color=th.accent_secondary,
            hold=lambda: sweep(p1, th.accent_highlight),
            sfx="pop_marimba",
        )

        stages = [
            ("order_3", 3, th.accent_secondary, r"P_3(x) = x - \frac{x^3}{3!}", "Order 3", "tick_soft"),
            ("order_5", 5, th.accent_tertiary, r"P_5(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!}", "Order 5", "tick_soft"),
            (
                "order_7",
                7,
                th.accent_tertiary,
                r"P_7(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!}",
                "Order 7",
                "chime_reveal",
            ),
            (
                "order_11",
                11,
                th.success,
                r"P_{11}(x) = \sum_{k=0}^{5} \frac{(-1)^k x^{2k+1}}{(2k+1)!}",
                "Order 11",
                "chord_epiphany",
            ),
        ]
        for beat_id, order, color, tex, label, sfx_name in stages:
            p_next, err_next = build_approx(order, color)
            state["order"] = order
            epiphany = order == 11
            self.play_beat(
                beat_id,
                m.Transform(header.formula, header.build_formula(tex, color=color)),
                m.Transform(p1, p_next),
                m.Transform(err1, err_next),
                *badge.swap(label, color),
                caption_color=color if epiphany else None,
                border_color=color,
                hold=(lambda: [*sweep(p1, th.accent_highlight), m.Flash(anchor_dot, color=th.success)])
                if epiphany
                else (lambda: sweep(p1, th.accent_highlight)),
                sfx=sfx_name,
            )

        # Outro: the infinite series, with a slow cinematic push-in
        self.play_beat(
            "outro",
            m.Transform(
                header.formula,
                header.build_formula(
                    r"\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}",
                    color=th.accent_highlight,
                ),
            ),
            m.FadeOut(err1),
            *badge.swap("Order ∞", th.accent_highlight),
            caption_color=th.accent_highlight,
            border_color=th.accent_highlight,
            hold=lambda: [*sweep(p1, th.accent_highlight), m.Indicate(header.formula, scale_factor=1.06,
                                                                      color=th.accent_highlight)],
            sfx="chime_reveal",
        )
