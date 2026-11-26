"""3D Multivariable Calculus Showcase: Second-Order Partial Derivatives & Saddle Curvature."""

from __future__ import annotations

import manimgx as m
import numpy as np
from studio.audio.tts import BeatSpec
from studio.layout import CaptionCard, HeaderBanner, StatBadge
from studio.scene import Story3DScene


class HessianSaddle3D(Story3DScene):
    """Visualizes f(x, y) = x^2 - y^2, zero gradient at origin, and opposing second derivatives."""

    BEATS = [
        BeatSpec(
            id="hook",
            narration="In calculus, when the slope hits zero, is it always a peak or a valley?",
            caption="Zero slope doesn't always mean a peak or valley!",
            pause_after=0.25,
        ),
        BeatSpec(
            id="surface",
            narration="Meet the hyperbolic paraboloid: z equals x squared minus y squared.",
            caption="Meet the 3D saddle: z = x² - y²",
            pause_after=0.25,
        ),
        BeatSpec(
            id="x_slice",
            narration="Slice along the x axis, and curvature goes up. A local minimum!",
            caption="Along x: f_xx = +2 (curves UP)",
            pause_after=0.25,
        ),
        BeatSpec(
            id="y_slice",
            narration="Slice along y, and curvature flips downward. A local maximum!",
            caption="Along y: f_yy = -2 (curves DOWN)",
            pause_after=0.25,
        ),
        BeatSpec(
            id="hessian",
            narration="Because the second derivatives fight each other, the Hessian determinant is negative: a saddle point!",
            caption="det(H) < 0  =>  Classic Saddle Point!",
            pause_after=0.35,
        ),
    ]

    def construct(self) -> None:
        self.setup_studio()
        th = self.theme
        lay = self.layout

        header = HeaderBanner(
            layout=lay,
            theme=th,
            tag_text="Multivariable Calculus",
            title_text="Second-Order Curvature",
            initial_formula=r"\nabla f(0, 0) = \mathbf{0}",
        )
        self.caption_card = CaptionCard(layout=lay, theme=th, initial_text=self.BEATS[0].narration)

        # Live probe orbiting the origin on the surface: (x, y) = r(cos θ, sin θ)
        probe_r = 1.0
        probe_period = 7.0

        def probe_uv() -> tuple[float, float]:
            a = 2.0 * np.pi * self.now() / probe_period
            return probe_r * np.cos(a), probe_r * np.sin(a)

        def probe_f() -> float:
            u, v = probe_uv()
            return u * u - v * v

        stage_bot_y = lay.stage_center[1] - lay.stage_h * 0.5
        badge_h = min(0.84, lay.stage_h * 0.13)
        badge = StatBadge(
            layout=lay,
            theme=th,
            center=np.array([0.0, stage_bot_y + badge_h * 0.5 + 0.06, 0.0]),
            width=min(lay.stage_w * 0.94, 7.4),
            height=badge_h,
            label="∇f = 0",
            prefix_tex=r"f(x,y)=",
            value_fn=probe_f,
            color=th.accent_primary,
        )

        # Lock 2D HUD chrome in screen space while the 3D camera orbits
        self.add_fixed_in_frame_mobjects(header.group, self.caption_card.group, badge.group)
        self.set_camera_orientation(phi=66 * m.DEGREES, theta=-45 * m.DEGREES)

        stage_span = min(lay.stage_w, lay.stage_h) * 0.68
        axes3d = m.ThreeDAxes(
            x_range=[-1.6, 1.6, 0.8],
            y_range=[-1.6, 1.6, 0.8],
            z_range=[-2.2, 2.2, 1.1],
            x_length=stage_span,
            y_length=stage_span,
            z_length=stage_span * 0.75,
        )

        # Faint floor grid under the saddle for spatial depth
        floor = m.NumberPlane(
            x_range=[-2.0, 2.0, 0.5],
            y_range=[-2.0, 2.0, 0.5],
            x_length=stage_span * 1.25,
            y_length=stage_span * 1.25,
            background_line_style={"stroke_color": th.surface_border, "stroke_width": 1.4, "stroke_opacity": 0.55},
            faded_line_ratio=2,
            axis_config={"stroke_opacity": 0.0, "stroke_width": 0.0},
        )
        floor.move_to(axes3d.c2p(0.0, 0.0, -2.0))

        saddle_surface = m.Surface(
            lambda u, v: axes3d.c2p(u, v, u**2 - v**2),
            u_range=[-1.35, 1.35],
            v_range=[-1.35, 1.35],
            resolution=(22, 22),
            fill_opacity=0.68,
            stroke_width=0.6,
            checkerboard_colors=[th.accent_primary, th.surface_border],
        )
        origin_dot = m.Dot3D(axes3d.c2p(0.0, 0.0, 0.0), radius=0.09, color=th.accent_highlight)

        x_parabola = m.ParametricFunction(
            lambda t: axes3d.c2p(t, 0.0, t**2), t_range=[-1.35, 1.35, 0.04], color=th.success, stroke_width=6.0
        )
        y_parabola = m.ParametricFunction(
            lambda t: axes3d.c2p(0.0, t, -(t**2)), t_range=[-1.35, 1.35, 0.04], color=th.danger, stroke_width=6.0
        )

        shift = np.array([0.0, lay.stage_center[1] * 0.35, 0.0])

        def make_probe() -> m.VGroup:
            u, v = probe_uv()
            p = axes3d.c2p(u, v, u * u - v * v)
            # Steepest-ascent direction on the surface: (f_x, f_y, f_x^2 + f_y^2)
            gx, gy = 2.0 * u, -2.0 * v
            tip = axes3d.c2p(u + 0.35 * gx, v + 0.35 * gy, u * u - v * v + 0.35 * (gx * gx + gy * gy))
            tail = axes3d.c2p(u, v, u * u - v * v)
            d = tip - tail
            d = d / max(np.linalg.norm(d), 1e-6) * stage_span * 0.16
            grad = m.Line(p, p + d, color=th.accent_highlight, stroke_width=4.0)
            dot = m.Dot3D(p, radius=0.07, color=th.accent_highlight)
            return m.VGroup(grad, dot).shift(shift)

        probe = m.always_redraw(make_probe)

        world = m.Group(floor, axes3d, saddle_surface, origin_dot, x_parabola, y_parabola)
        world.shift(shift)

        # Frame 0: the axes, floor, critical point and HUD are visible, and the orbit is already moving
        self.add(floor, axes3d, origin_dot)
        self.begin_ambient_camera_rotation(rate=0.22)

        self.play_beat(
            "hook",
            m.Flash(origin_dot, color=th.accent_highlight),
            m.Indicate(header.formula, color=th.accent_highlight),
            sfx="whoosh_soft",
        )

        self.play_beat(
            "surface",
            m.Transform(header.formula, header.build_formula(r"f(x, y) = x^2 - y^2", color=th.accent_primary)),
            m.Create(saddle_surface),
            m.FadeIn(probe),
            *badge.swap("Probe", th.accent_primary),
            border_color=th.accent_primary,
            sfx="chime_reveal",
        )

        self.play_beat(
            "x_slice",
            m.Transform(
                header.formula,
                header.build_formula(r"\frac{\partial^2 f}{\partial x^2} = +2 > 0", color=th.success),
            ),
            m.Create(x_parabola),
            *badge.swap("x: min", th.success),
            caption_color=th.success,
            border_color=th.success,
            hold=lambda: [m.Indicate(x_parabola, color=th.accent_highlight, scale_factor=1.0)],
            sfx="pop_marimba",
        )

        self.play_beat(
            "y_slice",
            m.Transform(
                header.formula,
                header.build_formula(r"\frac{\partial^2 f}{\partial y^2} = -2 < 0", color=th.danger),
            ),
            m.Create(y_parabola),
            *badge.swap("y: max", th.danger),
            caption_color=th.danger,
            border_color=th.danger,
            hold=lambda: [m.Indicate(y_parabola, color=th.accent_highlight, scale_factor=1.0)],
            sfx="pop_marimba",
        )

        self.play_beat(
            "hessian",
            m.Transform(
                header.formula,
                header.build_formula(r"\det(H) = f_{xx}f_{yy} - f_{xy}^2 = -4 < 0", color=th.accent_highlight),
            ),
            *badge.swap("Saddle", th.accent_highlight),
            m.Flash(origin_dot, color=th.accent_highlight),
            caption_color=th.accent_highlight,
            border_color=th.accent_highlight,
            hold=lambda: [m.Indicate(header.formula, scale_factor=1.06, color=th.accent_highlight)],
            sfx="chord_epiphany",
        )
        self.stop_ambient_camera_rotation()
