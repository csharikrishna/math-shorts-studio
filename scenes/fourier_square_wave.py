"""Higher-Order Harmonic Calculus: Fourier Series Building a Square Wave from Pure Sine Waves."""

from __future__ import annotations

import manimgx as m
import numpy as np
from studio.audio.tts import BeatSpec
from studio.layout import CaptionCard, HeaderBanner, StatBadge, ambient_grid, fit_mobject, glow
from studio.scene import StoryScene


def fourier_square(x: float | np.ndarray, max_odd_n: int) -> float | np.ndarray:
    """Compute partial Fourier sum of unit square wave up to harmonic `max_odd_n`."""
    total = np.zeros_like(x, dtype=float)
    for k in range(1, max_odd_n + 1, 2):
        total = total + np.sin(k * x) / k
    return (4.0 / np.pi) * total


class FourierSquareWave(StoryScene):
    """Visualizes higher-order Fourier harmonics transforming smooth waves into sharp square edges."""

    BEATS = [
        BeatSpec(
            id="hook",
            narration="Can infinitely smooth sine waves ever build a razor-sharp square edge?",
            caption="Can smooth sine waves build a sharp square edge?",
            pause_after=0.25,
        ),
        BeatSpec(
            id="n_1",
            narration="Start with a single sine wave. Smooth, curved, and nowhere near flat.",
            caption="1st Harmonic: one smooth sine wave",
            pause_after=0.25,
        ),
        BeatSpec(
            id="n_3",
            narration="Add the third harmonic at one-third amplitude, and the peaks begin to flatten.",
            caption="Add 3rd harmonic: peaks begin to flatten",
            pause_after=0.25,
        ),
        BeatSpec(
            id="n_7",
            narration="Layer up to the seventh harmonic, and steep vertical cliffs suddenly emerge!",
            caption="Up to 7th harmonic: vertical walls form",
            pause_after=0.25,
        ),
        BeatSpec(
            id="n_19",
            narration="Push to nineteen harmonics, and smooth sine waves sculpt a digital square wave!",
            caption="19th harmonic: smooth waves sculpt a square!",
            pause_after=0.30,
        ),
        BeatSpec(
            id="outro",
            narration="From music to Wi-Fi, Fourier math turns pure waves into everything.",
            caption="Fourier math powers audio, images & Wi-Fi.",
            pause_after=0.35,
        ),
    ]

    def construct(self) -> None:
        self.setup_studio()
        th = self.theme
        lay = self.layout
        grid = ambient_grid(lay, th)

        header = HeaderBanner(
            layout=self.layout,
            theme=self.theme,
            tag_text="Fourier Analysis",
            title_text="Waves Into Square Edges",
            initial_formula=r"f(x) = \operatorname{sgn}(\sin x)",
        )
        self.caption_card = CaptionCard(
            layout=self.layout,
            theme=self.theme,
            initial_text=self.BEATS[0].narration,
        )

        # Aspect-ratio-agnostic stage coordinates
        x_lim = 3.6
        y_lim = 1.65
        axes_h = self.layout.stage_h * 0.58
        axes_w = min(self.layout.stage_w * 0.92, axes_h * 1.95)

        stage_top_y = self.layout.stage_center[1] + self.layout.stage_h * 0.5
        stage_bot_y = self.layout.stage_center[1] - self.layout.stage_h * 0.5

        axes = m.Axes(
            x_range=[-x_lim, x_lim, np.pi / 2],
            y_range=[-y_lim, y_lim, 1.0],
            x_length=axes_w,
            y_length=axes_h,
            axis_config={
                "color": self.theme.text_muted,
                "stroke_width": 2.2,
                "include_ticks": False,
                "tip_length": 0.16,
            },
        ).move_to(np.array([0.0, stage_top_y - axes_h * 0.5 - 0.18, 0.0]))

        # Domain labels strictly below plot bottom edge
        pi_markers = m.VGroup()
        plot_bottom_y = axes.c2p(0.0, -y_lim)[1]
        for val, tex in [(-np.pi, r"-\pi"), (np.pi, r"\pi")]:
            x_px = axes.c2p(val, 0.0)[0]
            guide = m.DashedLine(
                np.array([x_px, axes.c2p(0.0, 1.35)[1], 0.0]),
                np.array([x_px, plot_bottom_y - 0.06, 0.0]),
                dash_length=0.10,
                color=self.theme.surface_border,
                stroke_width=1.8,
            )
            lbl = m.MathTex(tex, font_size=32, color=self.theme.text_muted)
            fit_mobject(lbl, min_height=self.layout.min_text_h * 0.95)
            lbl.next_to(guide, m.DOWN, buff=0.08)
            pi_markers.add(guide, lbl)

        # Ideal target square wave in electric cyan
        sq_pts = [
            axes.c2p(-np.pi, 0.0),
            axes.c2p(-np.pi, -1.0),
            axes.c2p(0.0, -1.0),
            axes.c2p(0.0, 1.0),
            axes.c2p(np.pi, 1.0),
            axes.c2p(np.pi, 0.0),
        ]
        target_square = m.VMobject(color=th.accent_primary, stroke_width=3.2, stroke_opacity=0.65)
        target_square.set_points_as_corners(sq_pts)
        square_glow = glow(target_square, layers=2)

        # Live scan probe state: sweeps x across the domain, reading the current partial sum
        state = {"n": 0}
        sweep_period = 6.5

        def probe_x() -> float:
            return 0.92 * np.pi * np.sin(2.0 * np.pi * self.now() / sweep_period)

        def probe_y() -> float:
            x = probe_x()
            return float(fourier_square(np.array(x), state["n"])) if state["n"] else float(np.sign(np.sin(x)))

        badge_h = min(0.84, lay.stage_h * 0.15)
        badge_y = max(stage_bot_y + badge_h * 0.5 + 0.06, plot_bottom_y - 0.98)
        badge = StatBadge(
            layout=lay,
            theme=th,
            center=np.array([0.0, badge_y, 0.0]),
            width=min(lay.stage_w * 0.94, 7.4),
            height=badge_h,
            label="Target: Square",
            prefix_tex=r"f(x)=",
            value_fn=probe_y,
            color=th.accent_primary,
        )

        def make_probe() -> m.VGroup:
            x, y = probe_x(), probe_y()
            top = axes.c2p(x, y)
            base = axes.c2p(x, 0.0)
            parts: list[m.Mobject] = []
            if abs(y) > 0.02:
                parts.append(m.Line(base, top, color=th.accent_highlight, stroke_width=2.4, stroke_opacity=0.7))
            parts.append(m.Dot(top, radius=0.085, color=th.accent_highlight))
            return m.VGroup(*parts)

        probe = m.always_redraw(make_probe)

        def build_harmonic(max_n: int, color: str) -> m.Mobject:
            return axes.plot(
                lambda x, n=max_n: fourier_square(x, n),
                x_range=[-np.pi, np.pi, 0.015],
                color=color,
                stroke_width=4.2,
            )

        def sweep(curve: m.VMobject, color: str) -> list[m.Animation]:
            flash = curve.copy().set_stroke(color=color, width=curve.get_stroke_width() * 1.8, opacity=1.0)
            return [m.ShowPassingFlash(flash, time_width=0.35)]

        # Frame 0: complete hook on screen before the first word
        self.add(grid, square_glow, axes, pi_markers, target_square, badge.group, header.group, self.caption_card.group)
        frame = self.camera.frame  # type: ignore[attr-defined]
        frame.scale(0.86).move_to(axes.get_center())

        self.play_beat(
            "hook",
            frame.animate.scale(1.0 / 0.86).move_to(np.zeros(3)),
            m.FadeIn(probe, scale=0.6),
            m.Indicate(header.formula, color=th.accent_highlight),
            rate_func=m.ease_out_cubic,
            hold=lambda: sweep(target_square, th.accent_highlight),
            sfx="whoosh_soft",
        )

        # 1st harmonic
        w1 = build_harmonic(1, th.accent_secondary)
        state["n"] = 1
        self.add(glow(w1))
        self.play_beat(
            "n_1",
            m.Transform(header.formula, header.build_formula(r"S_1(x) = \frac{4}{\pi}\sin(x)", color=th.accent_secondary)),
            m.Create(w1),
            *badge.swap("N = 1", th.accent_secondary, prefix_tex=r"S(x)="),
            border_color=th.accent_secondary,
            hold=lambda: sweep(w1, th.accent_highlight),
            sfx="pop_marimba",
        )

        stages = [
            ("n_3", 3, th.accent_secondary, r"S_3(x) = \frac{4}{\pi}\left(\sin x + \frac{\sin 3x}{3}\right)", "N = 3", "tick_soft"),
            ("n_7", 7, th.accent_tertiary, r"S_7(x) = \frac{4}{\pi}\sum_{k=1,3,5,7}\frac{\sin(kx)}{k}", "N = 7", "chime_reveal"),
            (
                "n_19",
                19,
                th.success,
                r"S_{19}(x) = \frac{4}{\pi}\sum_{n=0}^{9}\frac{\sin((2n+1)x)}{2n+1}",
                "N = 19",
                "chord_epiphany",
            ),
        ]
        for beat_id, n, color, tex, label, sfx_name in stages:
            w_next = build_harmonic(n, color)
            state["n"] = n
            epiphany = n == 19
            self.play_beat(
                beat_id,
                m.Transform(header.formula, header.build_formula(tex, color=color)),
                m.Transform(w1, w_next),
                *badge.swap(label, color),
                caption_color=color if epiphany else None,
                border_color=color,
                hold=(lambda: [*sweep(w1, th.accent_highlight), m.Indicate(badge.label, color=th.success)])
                if epiphany
                else (lambda: sweep(w1, th.accent_highlight)),
                sfx=sfx_name,
            )

        # Outro
        self.play_beat(
            "outro",
            m.Transform(
                header.formula,
                header.build_formula(
                    r"f(x) = \frac{4}{\pi}\sum_{n=0}^{\infty}\frac{\sin((2n+1)x)}{2n+1}",
                    color=th.accent_highlight,
                ),
            ),
            *badge.swap("N = ∞", th.accent_highlight),
            caption_color=th.accent_highlight,
            border_color=th.accent_highlight,
            hold=lambda: [*sweep(w1, th.accent_highlight), m.Indicate(header.formula, scale_factor=1.06,
                                                                      color=th.accent_highlight)],
            sfx="chime_reveal",
        )
