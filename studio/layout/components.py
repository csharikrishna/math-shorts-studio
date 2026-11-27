"""Safe-zone layout math, auto-wrapping text, and reusable UI components."""

from __future__ import annotations

from dataclasses import dataclass
import os
import manimgx as m
import numpy as np
from studio.config import FormatProfile, Theme, CAPTION_FONT, HEADER_FONT


@dataclass(frozen=True)
class SafeLayout:
    """Computes non-overlapping header, center stage, and caption regions for any aspect ratio."""

    frame_w: float
    frame_h: float
    safe_w: float
    header_center: np.ndarray
    header_max_h: float
    stage_center: np.ndarray
    stage_w: float
    stage_h: float
    caption_center: np.ndarray
    caption_h: float
    min_text_h: float

    @classmethod
    def from_config(cls, profile: FormatProfile) -> SafeLayout:
        fw = float(m.config.frame_width)
        fh = float(m.config.frame_height)

        side_pad = fw * profile.side_safe_ratio
        top_pad = fh * profile.top_safe_ratio
        bot_pad = fh * profile.bottom_safe_ratio

        safe_w = fw - 2.0 * side_pad
        usable_h = fh - top_pad - bot_pad
        usable_top = fh * 0.5 - top_pad
        usable_bot = -fh * 0.5 + bot_pad

        if profile.is_vertical:
            header_h = usable_h * 0.23
            caption_h = usable_h * 0.16
            gap = usable_h * 0.025
        else:
            header_h = usable_h * 0.24
            caption_h = usable_h * 0.14
            gap = usable_h * 0.030

        stage_h = usable_h - header_h - caption_h - 2.0 * gap
        header_y = usable_top - header_h * 0.5
        caption_y = usable_bot + caption_h * 0.5
        stage_y = (usable_top - header_h - gap + usable_bot + caption_h + gap) * 0.5

        # ManimGX flags text smaller than ~21px at 1080p normalized height; enforce 26px+ floor
        min_text_h = fh * (26.0 / 1080.0)

        return cls(
            frame_w=fw,
            frame_h=fh,
            safe_w=safe_w,
            header_center=np.array([0.0, header_y, 0.0]),
            header_max_h=header_h,
            stage_center=np.array([0.0, stage_y, 0.0]),
            stage_w=safe_w,
            stage_h=stage_h,
            caption_center=np.array([0.0, caption_y, 0.0]),
            caption_h=caption_h,
            min_text_h=min_text_h,
        )


def fit_mobject(
    mob: m.Mobject,
    max_width: float | None = None,
    max_height: float | None = None,
    min_height: float | None = None,
) -> m.Mobject:
    """Proportionally fit a Mobject within maximum bounds while respecting optional min height."""
    if max_width is not None and mob.width > max_width and mob.width > 1e-6:
        mob.scale(max_width / mob.width)
    if max_height is not None and mob.height > max_height and mob.height > 1e-6:
        mob.scale(max_height / mob.height)
    if min_height is not None and mob.height < min_height and mob.height > 1e-6:
        scale_up = min_height / mob.height
        if max_width is None or mob.width * scale_up <= max_width:
            mob.scale(scale_up)
    return mob


def wrap_text(
    text: str,
    font_size: int,
    max_width: float,
    color: str = m.WHITE,
    weight: str = "BOLD",
    min_height: float | None = None,
    font: str | None = None,
) -> m.Text:
    """Wrap prose across lines so it fits cleanly inside max_width without horizontal clipping."""
    _font = font or HEADER_FONT
    words = text.split()
    if not words:
        return m.Text("", font_size=font_size, font=_font, color=color, weight=weight)

    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        probe = m.Text(candidate, font_size=font_size, font=_font, weight=weight)
        if probe.width <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))

    if len(lines) == 2 and len(words) >= 3:
        # Balance 2-line splits evenly to prevent awkward single-word orphan wraps
        best_split = None
        best_diff = float("inf")
        for idx in range(1, len(words)):
            l1 = " ".join(words[:idx])
            l2 = " ".join(words[idx:])
            w1 = m.Text(l1, font_size=font_size, font=_font, weight=weight).width
            w2 = m.Text(l2, font_size=font_size, font=_font, weight=weight).width
            if w1 <= max_width and w2 <= max_width and abs(w1 - w2) < best_diff:
                best_diff = abs(w1 - w2)
                best_split = [l1, l2]
        if best_split is not None:
            lines = best_split

    rendered = m.Text("\n".join(lines), font_size=font_size, font=_font, color=color, weight=weight)
    return fit_mobject(rendered, max_width=max_width, min_height=min_height)


class HeaderBanner:
    """Structured top banner with category tag, main title, and top-anchored swappable formula slot."""

    def __init__(
        self,
        layout: SafeLayout,
        theme: Theme,
        tag_text: str,
        title_text: str,
        initial_formula: str,
    ) -> None:
        self.layout = layout
        self.theme = theme

        tag = m.Text(tag_text.upper(), font_size=28, font=HEADER_FONT, weight="BOLD", color=theme.accent_primary)
        fit_mobject(tag, max_width=layout.safe_w * 0.90, min_height=layout.min_text_h)

        title = wrap_text(
            title_text,
            font_size=38,
            max_width=layout.safe_w * 0.95,
            color=theme.text_primary,
            weight="BOLD",
            min_height=layout.min_text_h * 1.25,
            font=HEADER_FONT,
        )

        self.top_stack = m.VGroup(tag, title).arrange(m.DOWN, buff=0.14)
        self.formula_buff = 0.20 if layout.frame_w > layout.frame_h else 0.26

        # Position top_stack near the top of header zone, then bound max_formula_h strictly to header bottom
        header_top_y = layout.header_center[1] + layout.header_max_h * 0.5
        header_bot_y = layout.header_center[1] - layout.header_max_h * 0.5
        self.top_stack.move_to(np.array([0.0, header_top_y - self.top_stack.height * 0.5 - 0.04, 0.0]))
        self.max_formula_h = max(0.35, (self.top_stack.get_bottom()[1] - self.formula_buff) - header_bot_y)

        self.formula = self.build_formula(initial_formula, color=theme.accent_secondary)
        self.group = m.VGroup(self.top_stack, self.formula)

    def build_formula(self, tex: str, color: str | None = None, font_size: int = 44) -> m.MathTex:
        """Create a properly scaled replacement formula anchored cleanly below top_stack."""
        mob = m.MathTex(tex, font_size=font_size, color=color or self.theme.accent_secondary)
        fit_mobject(
            mob,
            max_width=self.layout.safe_w * 0.92,
            max_height=self.max_formula_h,
        )
        mob.next_to(self.top_stack, m.DOWN, buff=self.formula_buff)
        return mob


def scene_time() -> float:
    """Current scene time in seconds (ManimGX's exact rational clock)."""
    from manimgx.animation import clock

    return float(clock.now)


def glow(curve: m.VMobject, color: str | None = None, layers: int = 3, spread: float = 2.6) -> m.VGroup:
    """Soft halo pass: wide, faint stroke copies that track `curve` every frame.

    ManimGX's bloom only lights material surfaces, so 2D glow is layered strokes. The halo
    follows the curve's points (and color unless fixed), so Create/Transform on the curve
    carry the glow with it.
    """
    base_w = max(1.0, float(curve.get_stroke_width()))
    halo = m.VGroup()
    for k in range(layers, 0, -1):
        layer = m.VMobject()
        layer.match_points(curve)
        layer.set_fill(opacity=0.0)
        layer.set_stroke(
            color=color or curve.get_stroke_color(),
            width=base_w * (1.0 + spread * k / layers),
            opacity=0.30 / (k + 1.0),
        )

        def _track(mob: m.VMobject, src: m.VMobject = curve, fixed: str | None = color) -> None:
            mob.match_points(src)
            if fixed is None:
                mob.set_stroke(color=src.get_stroke_color())

        layer.add_updater(_track)
        halo.add(layer)
    return halo


def ambient_grid(layout: SafeLayout, theme: Theme, drift_speed: float = 0.06) -> m.NumberPlane:
    """Full-bleed faint NumberPlane that drifts diagonally with a seamless wrap.

    Sized from the frame (any aspect ratio); grid lines are exempt from text-crossing QA,
    so it can live behind every element.
    """
    step = max(layout.frame_w, layout.frame_h) / 9.0
    half_w = layout.frame_w * 0.5 + 2.0 * step
    half_h = layout.frame_h * 0.5 + 2.0 * step
    plane = m.NumberPlane(
        x_range=[-half_w / step, half_w / step, 1.0],
        y_range=[-half_h / step, half_h / step, 1.0],
        x_length=2.0 * half_w,
        y_length=2.0 * half_h,
        background_line_style={
            "stroke_color": theme.surface_border,
            "stroke_width": 1.4,
            "stroke_opacity": 0.38,
        },
        faded_line_ratio=2,
        axis_config={"stroke_opacity": 0.0, "stroke_width": 0.0},
    )
    anchor = plane.get_center().copy()

    def _drift(mob: m.Mobject) -> None:
        offset = (scene_time() * drift_speed * step) % step
        mob.move_to(anchor + np.array([offset, offset, 0.0]))

    plane.add_updater(_drift)
    return plane


def ui_style(style: str | None = None) -> str:
    """Return the active UI style: 'minimal', 'card', or 'neon'."""
    chosen = style or os.environ.get("STUDIO_UI", "minimal")
    if chosen not in UI_STYLES:
        raise ValueError(f"Unknown UI style {chosen!r}. Options: {UI_STYLES}")
    return chosen


UI_STYLES = ("minimal", "card", "neon")


class CaptionCard:
    """Shorts-style word-synced captions: one punchy line at a time, karaoke-highlighted.

    Words come from edge-tts WordBoundary timings (absolute seconds). The captions drive
    themselves from the scene clock with an updater, so they stay locked to speech through
    plays and waits alike, with no hardcoded timing.

    Styles
    ------
    minimal (default)
        Box-free. Big, bold Segoe UI words with a soft dark outline — legible over any grid
        or curve. Upcoming words in muted grey; spoken words white; active word pops YELLOW
        with a small scale impulse; *emphasis* words pop CYAN with a damped-spring bounce.
        Font size is kept proportional (~30 % of caption zone height) so it never dominates
        the frame the way the old 2.3× multiplier did.

    card
        A frosted-glass rounded pill sits behind the entire caption line. The active word is
        highlighted with a solid accent fill (inverted text) — same energy as Instagram Reels
        captions. Clean and readable on complex backgrounds.

    neon
        A fullwidth semi-transparent gradient bar anchored to the bottom of the caption zone.
        Words render in the bar; the active word pulses with a layered neon glow (mimicking
        TikTok / YouTube Shorts high-energy style). Great for fast-paced mathematics.
    """

    MAX_WORDS = 6
    # Outline stroke width (scene units × 100) — keeps bare text crisp over grids and curves
    OUTLINE_WIDTH = 6.0
    # Caption text height as a fraction of caption zone height — governs overall text size.
    # Previously this was 2.3 × min_text_h which was huge; 1.5 × feels right for Shorts.
    LINE_H_SCALE = 1.45

    def __init__(self, layout: SafeLayout, theme: Theme, initial_text: str, style: str | None = None) -> None:
        self.layout = layout
        self.theme = theme
        self.style = ui_style(style)
        self.minimal = self.style == "minimal"
        self.card_w = layout.safe_w
        self.card_h = layout.caption_h

        # ---- Proportional line height ------------------------------------------
        # Keep captions at a sensible size: at most LINE_H_SCALE × min_text_h,
        # and at most 42 % of the caption zone height so they don't eat the frame.
        self.line_h = min(layout.min_text_h * self.LINE_H_SCALE, layout.caption_h * 0.42)

        if self.minimal or self.style == "neon":
            self.inner_w = self.card_w - 0.20
            self.inner_h = self.card_h
        else:
            self.inner_w = self.card_w - 0.70
            self.inner_h = self.card_h - 0.36

        # Reference height for scaling text glyphs to self.line_h
        self._ref_h = m.Text("Hg", font_size=48, font=CAPTION_FONT, weight="BOLD").height
        self.accent = theme.accent_highlight

        # Invisible layout anchor (always present for consistent VGroup structure)
        bg_opacity = 0.0 if (self.minimal or self.style == "neon") else 0.92
        self.bg = m.RoundedRectangle(
            corner_radius=0.22 if self.style == "card" else 0.0,
            width=self.card_w,
            height=self.card_h,
            fill_color=theme.surface if self.style == "card" else "#000000",
            fill_opacity=bg_opacity,
            stroke_color=theme.surface_border,
            stroke_width=0.0 if (self.minimal or self.style == "neon") else 2.5,
        ).move_to(layout.caption_center)

        # Neon style: opaque (>=0.92 for QA) cinema dock with a sleek glowing top accent rule
        if self.style == "neon":
            bar_h = self.card_h * 1.08
            dock = m.Rectangle(
                width=layout.frame_w,
                height=bar_h,
                fill_color=theme.surface,
                fill_opacity=0.93,
                stroke_width=0.0,
            ).move_to(layout.caption_center)
            top_y = layout.caption_center[1] + bar_h * 0.5
            glow_line = m.Line(
                np.array([-layout.frame_w * 0.5, top_y, 0.0]),
                np.array([layout.frame_w * 0.5, top_y, 0.0]),
                color=theme.accent_primary,
                stroke_width=3.0,
                stroke_opacity=0.85,
            )
            self._neon_bar = m.VGroup(dock, glow_line)
        else:
            self._neon_bar = m.VGroup()

        self.pill: m.Mobject = m.VGroup()
        self.line: m.Mobject = m.VGroup()
        self.group = m.VGroup(self._neon_bar, self.bg, self.pill, self.line)

        self._chunks: list[tuple[float, list[tuple[str, float, float]]]] = []
        self._built: dict[int, tuple[m.Text, list[m.VGroup]]] = {}
        self._shown: tuple[int, int] = (-1, -1)
        self._pill_born = 0.0
        self._pill_scale = 1.0
        self._pill_center = np.zeros(3)
        self._bounce_part = None
        self._bounce_center = np.zeros(3)
        self._bounce_scale = 1.0
        self._pop_amp = 0.10
        self._bounce_amp = 0.0
        self._spring = False
        self.set_words([(w, 0.0, 0.0, False) for w in initial_text.split()], 0.0)
        self._tick(0.0)
        self.group.add_updater(lambda _mob: self._tick(scene_time()))

    # -- chunking -------------------------------------------------------------
    def _text(self, s: str) -> m.Text:
        txt = m.Text(
            s,
            font_size=48,
            font=CAPTION_FONT,
            weight="BOLD",
            color=self.theme.text_muted,
            disable_ligatures=True,
        )
        txt.scale(self.line_h / max(self._ref_h, 1e-6))
        if self.minimal or self.style == "neon":
            # Soft dark outline drawn behind the fill keeps bare text crisp over grids and curves
            txt.set_stroke(color=self.theme.background, width=self.OUTLINE_WIDTH, opacity=0.92, background=True)
        return txt

    def _make_line(self, words: list[str]) -> tuple[m.Text, list[m.VGroup]]:
        line = self._text(" ".join(words))
        fit_mobject(line, max_width=self.inner_w)
        line.move_to(self.bg.get_center())
        glyphs = list(line.submobjects)
        parts: list[m.VGroup] = []
        cursor = 0
        for w in words:
            n = len(w.replace(" ", ""))
            parts.append(m.VGroup(*glyphs[cursor : cursor + n]))
            cursor += n
        return line, parts

    PHRASE_END = (".", ",", "!", "?", ";", ":", "—")
    SOFT_BREAK_WORDS = {"and", "or", "but", "so", "because", "while", "where", "when", "then", "which", "that", "plus", "equals"}

    def set_words(self, words: list[tuple], beat_start: float) -> None:
        """Pack timed words into grammatical phrase chunks (shared with SRT/VTT semantic segmentation).

        Each word is (text, start, end[, emphasis]).
        """
        self._chunks = []
        self._built = {}
        norm = []
        for w in words:
            raw_tok = str(w[0])
            breaks_phrase = raw_tok.endswith(self.PHRASE_END)
            disp_tok = raw_tok[:-1] if (raw_tok.endswith(".") and not raw_tok.endswith("...") and not raw_tok[:-1].isdigit()) else raw_tok
            norm.append((disp_tok, float(w[1]), float(w[2]), bool(w[3]) if len(w) > 3 else False, breaks_phrase))
        current: list[tuple[str, float, float, bool]] = []
        for idx, (disp_tok, s_t, e_t, emph_f, breaks_phrase) in enumerate(norm):
            word = (disp_tok, s_t, e_t, emph_f)
            cand = [*current, word]
            too_wide = self._text(" ".join(w[0] for w in cand)).width > self.inner_w * 0.94
            if current and (len(cand) > self.MAX_WORDS + 1 or too_wide):
                self._chunks.append((current[0][1], current))
                current = [word]
            else:
                current = cand
            next_tok = norm[idx + 1][0].lower().strip(".,!?;:") if idx + 1 < len(norm) else ""
            soft_break = next_tok in self.SOFT_BREAK_WORDS and len(current) >= 3 and self._text(" ".join(w[0] for w in current)).width > self.inner_w * 0.62
            if (breaks_phrase and len(current) >= 2) or soft_break:
                self._chunks.append((current[0][1], current))
                current = []
        if current:
            self._chunks.append((current[0][1], current))
        if self._chunks:
            self._chunks[0] = (beat_start, self._chunks[0][1])
        self._shown = (-1, -1)

    def set_beat(self, words: list, narration: str, start: float, speech_end: float) -> None:
        """Bind a beat: real WordBoundary timings, or length-weighted estimates as a fallback."""
        if words:
            timed = [tuple(w) for w in words]
        else:
            toks = narration.split()
            total = sum(len(t) + 1 for t in toks) or 1
            span = max(0.1, speech_end - start)
            timed, cur = [], start
            for tok in toks:
                d = span * (len(tok) + 1) / total
                timed.append((tok, cur, cur + d, False))
                cur += d
        self.set_words(timed, start)
        self._tick(scene_time())

    # -- per-frame drive ------------------------------------------------------
    def _tick(self, t: float) -> None:
        if not self._chunks:
            return
        ci = 0
        for idx, (c_start, _) in enumerate(self._chunks):
            if t + 1e-6 >= c_start:
                ci = idx
        words = self._chunks[ci][1]
        wi = 0
        for j, w in enumerate(words):
            if t + 1e-6 >= w[1]:
                wi = j
        if (ci, wi) != self._shown:
            self._show(ci, wi, words, t)
        self._animate(t)

    def _show(self, ci: int, wi: int, words: list[tuple], t: float) -> None:
        if ci not in self._built:
            self._built[ci] = self._make_line([w[0] for w in words])
        line, parts = self._built[ci]
        self._reset_bounce()
        emph = words[wi][3]
        for j, part in enumerate(parts):
            if self.style == "neon":
                part.set_stroke(color=self.theme.background, width=self.OUTLINE_WIDTH, opacity=0.92, background=True)
            if j == wi:
                if self.minimal or self.style == "neon":
                    part.set_fill(color=self.theme.accent_primary if emph else self.accent)
                else:
                    part.set_color(self.theme.background)
            elif j < wi:
                # Spoken emphasis words stay lit in the secondary (cyan) accent
                part.set_fill(color=self.theme.accent_primary if words[j][3] else self.theme.text_primary)
            else:
                part.set_fill(color=self.theme.text_muted)
        if self.minimal or self.style == "neon":
            self._show_minimal(ci, wi, line, parts, emph, t)
            return
        # Uniform pill height for every word (ascenders/descenders never change its size)
        pill_h = self.line_h * 1.30
        pad_y = max(0.0, (pill_h - parts[wi].height) * 0.5)
        pill = m.BackgroundRectangle(
            parts[wi],
            buff=(self.line_h * 0.16, pad_y),
            corner_radius=self.line_h * 0.18,
            color=self.theme.accent_primary if emph else self.accent,
            fill_opacity=0.95,
            stroke_width=0.0,
        )
        # Draw order: card, pill, words
        pill.move_to(np.array([pill.get_center()[0], line.get_center()[1], 0.0]))
        self.group.submobjects = [self.bg, pill, line]
        self.pill, self.line = pill, line
        self._pill_center = pill.get_center().copy()
        self._pill_born = t
        self._pill_scale = 1.0
        self._bounce_part = parts[wi] if emph else None
        self._bounce_center = parts[wi].get_center().copy()
        self._bounce_scale = 1.0
        # Pops are capped per word so growth never reaches the neighbouring words: the pill may
        # grow into at most a third of the inter-word space, and the bouncing word stays inside
        # its resting pill.
        space = self.line_h * 0.30
        self._pop_amp = min(0.20 if emph else 0.10, 2.0 * space / 3.0 / max(pill.width, 1e-6))
        self._bounce_amp = min(0.14, pill.width / max(parts[wi].width, 1e-6) - 1.0)
        self._shown = (ci, wi)

    def _show_minimal(self, ci: int, wi: int, line: m.Text, parts: list[m.VGroup], emph: bool, t: float) -> None:
        """Box-free or neon bar: no pill; the active word pops (emphasis words spring with glow)."""
        if self.style == "neon":
            self.group.submobjects = [self._neon_bar, self.bg, line]
        else:
            self.group.submobjects = [self._neon_bar, self.bg, line]
        self.pill, self.line = m.VGroup(), line
        self._pill_born = t
        self._pill_scale = 1.0
        self._bounce_part = parts[wi]
        self._bounce_center = parts[wi].get_center().copy()
        self._bounce_scale = 1.0
        # Growth is capped so the word never reaches its neighbours (~40 % of a space per side)
        space = self.line_h * 0.30
        cap = 0.8 * space / max(parts[wi].width, 1e-6)
        self._pop_amp = 0.0
        self._bounce_amp = min(0.16 if emph else 0.08, cap)
        self._spring = emph
        # Neon style: subtle colored halo strictly behind the crisp glyph fill
        if self.style == "neon":
            glow_color = self.theme.accent_primary if emph else self.accent
            parts[wi].set_stroke(color=glow_color, width=self.OUTLINE_WIDTH * 1.45, opacity=0.45, background=True)
        self._shown = (ci, wi)

    def _reset_bounce(self) -> None:
        part = getattr(self, "_bounce_part", None)
        if part is not None and abs(self._bounce_scale - 1.0) > 1e-6:
            part.scale(1.0 / self._bounce_scale)
            part.move_to(self._bounce_center)
        self._bounce_part = None
        self._bounce_scale = 1.0

    def _animate(self, t: float) -> None:
        """Pill: ease-out pop. Active word: ease-out pop, or a damped-spring bounce for emphasis."""
        age = max(0.0, t - self._pill_born)
        emph = self._bounce_part is not None
        if self.style == "card":
            target = 1.0 + self._pop_amp * float(np.exp(-age * 18.0))
            if abs(target - self._pill_scale) > 1e-4:
                self.pill.scale(target / self._pill_scale)
                self.pill.move_to(self._pill_center)
                self._pill_scale = target
        if emph:
            if (self.minimal or self.style == "neon") and not getattr(self, "_spring", False):
                spring = 1.0 + self._bounce_amp * float(np.exp(-age * 14.0))  # plain words: quick pop
            else:
                spring = 1.0 + self._bounce_amp * float(np.exp(-age * 9.0) * np.cos(age * 24.0))
            if abs(spring - self._bounce_scale) > 1e-4:
                self._bounce_part.scale(spring / self._bounce_scale)
                self._bounce_part.move_to(self._bounce_center)
                self._bounce_scale = spring

    def transition(self, text: str, color: str | None = None, border_color: str | None = None) -> list[m.Animation]:
        """Per-beat chrome accent (the words themselves are clock-driven)."""
        if color is not None:
            self.accent = color
        anims: list[m.Animation] = []
        if border_color is not None:
            if self.style == "card":
                anims.append(self.bg.animate.set_stroke(color=border_color, width=3.0))
            elif self.style == "neon" and len(self._neon_bar.submobjects) >= 2:
                anims.append(self._neon_bar[1].animate.set_stroke(color=border_color, width=3.0))
        return anims


def adaptive_tex(value: float, decimals: int = 3, sci_below: float = 1e-3) -> str:
    """Readable live number: fixed decimals normally, scientific notation for tiny magnitudes."""
    if not np.isfinite(value):
        return r"\infty"
    mag = abs(value)
    if 0.0 < mag < sci_below:
        exp = int(np.floor(np.log10(mag)))
        mant = round(value / 10.0**exp, 1)
        if abs(mant) >= 10.0:
            mant, exp = mant / 10.0, exp + 1
        if abs(abs(mant) - 1.0) < 1e-9:
            return ("-" if mant < 0 else "") + rf"10^{{{exp}}}"
        return rf"{mant:.1f}\times10^{{{exp}}}"
    return f"{value:.{decimals}f}"


def precise_tex(value: float, max_decimals: int = 8) -> str:
    """Up to `max_decimals` decimals with trailing zeros trimmed (shows 0.99999999 honestly)."""
    if not np.isfinite(value):
        return r"\infty"
    text = f"{value:.{max_decimals}f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


class StatBadge:
    """Live status line: a swappable label and a live numeric readout.

    Styles:
      minimal (default) - no box: one centered line `label · prefix number` with a small dot separator,
          outlined text like the captions. The label is right-aligned to the separator and the
          number left-anchored after its prefix, so nothing jitters while the value ticks.
      card - an opaque pill (fill >= 0.9, which also lets curves pass behind it QA-clean).

    `value_fn` is polled each frame. By default the number is a DecimalNumber; pass `formatter`
    (value -> LaTeX) for adaptive formats such as scientific notation, rendered from a glyph cache.
    The label gets exactly the width the readout leaves free, so the two can never touch.

    Responsive: given every label/prefix the badge will show (`labels`, `prefixes`), it checks
    whether one row fits at full text size; if not (narrow 9:16 frames, long readouts) it becomes
    a two-row card: label on top, readout centered below. `height` is the one-row height.
    """

    GAP = 0.10
    DIV_GAP = 0.20  # label/readout distance from the minimal divider (well beyond the QA touch margin)
    CACHE_LIMIT = 512

    def __init__(
        self,
        layout: SafeLayout,
        theme: Theme,
        center: np.ndarray,
        width: float,
        height: float,
        label: str,
        prefix_tex: str,
        value_fn,
        color: str,
        decimals: int = 3,
        formatter=None,
        widest_samples: tuple[str, ...] = (r"8.8\times10^{-8}", "-8.888"),
        labels: tuple[str, ...] = (),
        prefixes: tuple[str, ...] = (),
        style: str | None = None,
    ) -> None:
        self.layout = layout
        self.theme = theme
        self.minimal = ui_style(style) in ("minimal", "neon")
        self.value_fn = value_fn
        self.formatter = formatter
        self.pad = height * 0.32
        self.text_h = max(layout.min_text_h * 1.02, height * 0.40)
        self._cache: dict[str, m.MathTex] = {}
        self.stacked = False

        if formatter is None:
            widest = m.DecimalNumber(-8.0, num_decimal_places=decimals, font_size=36)
            widest.scale(self.text_h / max(widest.height, 1e-6))
            self.num_w = widest.width * 1.06
        else:
            self._digit_scale = self.text_h / max(m.MathTex("0").height, 1e-6)
            self.num_w = max(self._number_mob(t).width for t in widest_samples) * 1.04

        # One row must hold: pad | label | separator | prefix | gap | number | pad, at full text size
        sep = self.DIV_GAP * 2 if self.minimal else 3 * self.GAP
        label_ws = [self._label_mob(t, color).width for t in (*labels, label)]
        prefix_ws = [self._prefix_mob(t, color).width for t in (*prefixes, prefix_tex)]
        self.label_slot, self.prefix_slot = max(label_ws), max(prefix_ws)
        need = 2 * self.pad + self.label_slot + sep + self.prefix_slot + self.GAP + self.num_w
        if need > width:
            self.stacked = True
            height = height * 1.75

        # In minimal style the pill is an invisible layout anchor
        self.bg = m.RoundedRectangle(
            corner_radius=min(height, 0.84) * 0.24,
            width=width,
            height=height,
            fill_color=theme.surface,
            fill_opacity=0.0 if self.minimal else 0.96,
            stroke_color=color,
            stroke_width=0.0 if self.minimal else 2.2,
        ).move_to(center)
        # Minimal one-row: a fixed divider, composition centered for the widest label/prefix pair
        total = self.label_slot + sep + self.prefix_slot + self.GAP + self.num_w
        self._divider_dx = -total * 0.5 + self.label_slot + self.DIV_GAP
        # A centered dot, not a bar: a vertical line next to |...| would read as math notation
        self.divider = m.Dot(
            radius=self.text_h * 0.09,
            color=theme.text_muted,
            fill_opacity=0.9 if (self.minimal and not self.stacked) else 0.0,
        )
        self.divider.add_updater(
            lambda d: d.move_to(np.array([self.bg.get_center()[0] + self._divider_dx, self.bg.get_center()[1], 0.0]))
        )

        if formatter is None:
            self.number = m.DecimalNumber(0.0, num_decimal_places=decimals, include_sign=False, font_size=36,
                                          color=theme.accent_highlight)
            self.number.scale(self.text_h / max(self.number.height, 1e-6))
        else:
            self.number = self._number_mob(formatter(0.0)).copy()
        self.prefix = self._make_prefix(prefix_tex, color)
        self.label = self._make_label(label, color)
        self.number.add_updater(self._track)
        self._track(self.number)
        self.divider.update()
        self.group = m.VGroup(self.bg, self.divider, self.label, self.prefix, self.number)

    # -- pieces -------------------------------------------------------------------
    def _number_mob(self, tex: str) -> m.MathTex:
        mob = self._cache.get(tex)
        if mob is None:
            if len(self._cache) >= self.CACHE_LIMIT:
                self._cache.clear()
            mob = m.MathTex(tex, color=self.theme.accent_highlight)
            mob.scale(self._digit_scale)
            self._outline(mob)
            self._cache[tex] = mob
        return mob

    def _row_y(self, top: bool) -> float:
        cy = self.bg.get_center()[1]
        if not self.stacked:
            return cy
        return cy + (1 if top else -1) * self.bg.height * 0.22

    def _outline(self, mob: m.Mobject) -> m.Mobject:
        """Minimal style: the same soft dark outline as the captions, for legibility over the grid."""
        if self.minimal:
            mob.set_stroke(color=self.theme.background, width=5.0, opacity=0.9, background=True)
        return mob

    def _readout_left(self, prefix_w: float) -> float:
        if self.minimal and not self.stacked:  # prefix starts just right of the divider
            return self.bg.get_center()[0] + self._divider_dx + self.DIV_GAP
        if self.minimal:  # box-free two-row: center on the number as it is now (no reserved gap)
            current = self.number.width if hasattr(self, "number") else self.num_w
            return self.bg.get_center()[0] - (prefix_w + self.GAP + current) * 0.5
        if self.stacked:  # readout block centered on the bottom row (prefix anchored, number grows right)
            return self.bg.get_center()[0] - (prefix_w + self.GAP + self.num_w) * 0.5
        return self.bg.get_right()[0] - self.pad - self.num_w - self.GAP - prefix_w

    def _prefix_mob(self, tex: str, color: str) -> m.MathTex:
        pre = m.MathTex(tex, font_size=36, color=color)
        pre.scale(self.text_h * 1.05 / max(pre.height, 1e-6))
        return self._outline(pre)

    def _label_mob(self, text: str, color: str) -> m.Text:
        lbl = m.Text(text, font_size=36, font=CAPTION_FONT, weight="BOLD", color=color)
        lbl.scale(self.text_h / max(lbl.height, 1e-6))
        return self._outline(lbl)

    def _make_prefix(self, tex: str, color: str) -> m.MathTex:
        pre = self._prefix_mob(tex, color)
        room = self.bg.width - 2 * self.pad - self.num_w - self.GAP
        fit_mobject(pre, max_width=max(0.3, room if self.stacked else room * 0.6))
        if pre.height < self.layout.min_text_h:
            pre.scale(self.layout.min_text_h / pre.height)  # readability floor wins over width
        left = self._readout_left(pre.width)
        pre.move_to(np.array([left + pre.width * 0.5, self._row_y(top=False), 0.0]))
        return pre

    def _make_label(self, text: str, color: str) -> m.Text:
        lbl = self._label_mob(text, color)
        if self.stacked:
            fit_mobject(lbl, max_width=self.bg.width - 2 * self.pad)
            lbl.move_to(np.array([self.bg.get_center()[0], self._row_y(top=True), 0.0]))
            return lbl
        if self.minimal:  # right-aligned against the divider; may extend left to the safe edge
            right = self.bg.get_center()[0] + self._divider_dx - self.DIV_GAP
            fit_mobject(lbl, max_width=right - (self.bg.get_left()[0] + self.pad))
            lbl.move_to(np.array([right - lbl.width * 0.5, self._row_y(top=True), 0.0]))
            return lbl
        max_w = self.prefix.get_left()[0] - self.GAP * 2.0 - (self.bg.get_left()[0] + self.pad)
        fit_mobject(lbl, max_width=max_w)
        lbl.move_to(np.array([self.bg.get_left()[0] + self.pad + lbl.width * 0.5, self._row_y(top=True), 0.0]))
        return lbl

    def _track(self, num: m.Mobject) -> None:
        value = float(self.value_fn())
        if self.formatter is None:
            num.set_value(value)  # type: ignore[attr-defined]
        else:
            num.become(self._number_mob(self.formatter(value)))
        if self.minimal and self.stacked:
            # Keep `prefix number` visually centered; ease so width changes (0.0100 -> 10^-4) glide
            target_left = self.bg.get_center()[0] - (self.prefix.width + self.GAP + num.width) * 0.5
            current_left = self.prefix.get_left()[0]
            self.prefix.shift(m.RIGHT * (target_left - current_left) * 0.35)
        num.move_to(np.array([self.prefix.get_right()[0] + self.GAP + num.width * 0.5, self._row_y(top=False), 0.0]))

    def swap(self, label: str, color: str, prefix_tex: str | None = None) -> list[m.Animation]:
        anims: list[m.Animation] = [] if self.minimal else [self.bg.animate.set_stroke(color=color, width=2.2)]
        if prefix_tex is not None:
            new_prefix = self._make_prefix(prefix_tex, color)
            anims.append(m.Transform(self.prefix, new_prefix))
            # Lay the label out against the incoming prefix
            old_prefix, self.prefix = self.prefix, new_prefix
            new_label = self._make_label(label, color)
            self.prefix = old_prefix
        else:
            new_label = self._make_label(label, color)
        anims.append(m.Transform(self.label, new_label))
        return anims
