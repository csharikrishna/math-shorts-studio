"""Safe-zone layout math, auto-wrapping text, and reusable UI components."""

from __future__ import annotations

from dataclasses import dataclass
import os
import manimgx as m
import numpy as np
from studio.config import FormatProfile, Theme, CAPTION_FONT, HEADER_FONT
from studio.design.fonts import manimgx_font_paths


def _text_font_kwargs() -> dict:
    fp = manimgx_font_paths()
    return {"font_paths": fp} if fp else {}


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
    def from_config(
        cls,
        profile: FormatProfile,
        *,
        has_header_banner: bool = True,
        has_equation_band: bool = False,
        caption_style: str | None = None,
    ) -> SafeLayout:
        from studio.captions.model import CAPTION_PRESETS, normalize_caption_style

        fw = float(m.config.frame_width)
        fh = float(m.config.frame_height)

        side_pad = fw * profile.side_safe_ratio
        top_pad = fh * profile.top_safe_ratio
        norm_cap = normalize_caption_style(
            caption_style or os.environ.get("STUDIO_CAPTION_STYLE") or os.environ.get("STUDIO_UI"),
            default="editorial" if not profile.is_vertical else "kinetic",
        )
        preset = CAPTION_PRESETS[norm_cap]
        cap_ratio = preset.reserve_band_ratio_portrait if profile.is_vertical else preset.reserve_band_ratio_landscape
        bot_pad = fh * (profile.bottom_safe_ratio * 0.25 if norm_cap == "off" else 0.02)

        safe_w = fw - 2.0 * side_pad
        usable_top = fh * 0.5 - top_pad
        usable_bot = -fh * 0.5 + bot_pad

        caption_h = fh * cap_ratio
        if has_header_banner:
            usable_h = fh - top_pad - bot_pad
            header_h = usable_h * (0.22 if profile.is_vertical else 0.19)
            gap = usable_h * 0.020
        elif has_equation_band:
            header_h = fh * (0.105 if profile.is_vertical else 0.105)
            gap = fh * (0.022 if profile.is_vertical else 0.018)
        else:
            header_h = 0.0
            gap = fh * (0.020 if profile.is_vertical else 0.016)

        stage_top = usable_top - header_h - (gap if header_h > 0 else 0.0)
        stage_bot = usable_bot + caption_h + gap
        stage_h = max(fh * 0.42, stage_top - stage_bot)
        header_y = usable_top - header_h * 0.5
        caption_y = usable_bot + caption_h * 0.5
        stage_y = 0.5 * (stage_top + stage_bot)

        # Enforce 26px+ readable floor across 1080p vertical & widescreen formats
        min_text_h = fh * (26.0 / 1080.0)

        return cls(
            frame_w=fw,
            frame_h=fh,
            safe_w=safe_w,
            header_center=np.array([0.0, header_y, 0.0]),
            header_max_h=max(0.45, header_h),
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
    is_light = getattr(theme, "is_light", False)
    plane = m.NumberPlane(
        x_range=[-half_w / step, half_w / step, 1.0],
        y_range=[-half_h / step, half_h / step, 1.0],
        x_length=2.0 * half_w,
        y_length=2.0 * half_h,
        background_line_style={
            "stroke_color": theme.grid_line if is_light else theme.surface_border,
            "stroke_width": 1.1 if is_light else 1.4,
            "stroke_opacity": 0.55 if is_light else 0.34,
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

    MAX_WORDS = 9
    # Outline stroke width (scene units × 100) — keeps bare text crisp over grids and curves
    OUTLINE_WIDTH = 4.2
    # Proportional scale for 1-line vs 2-line captions
    LINE_H_SCALE = 1.48

    def __init__(
        self,
        layout: SafeLayout,
        theme: Theme,
        initial_text: str,
        style: str | None = None,
        caption_style: str | None = None,
    ) -> None:
        from studio.captions.model import CAPTION_PRESETS, normalize_caption_style

        self.layout = layout
        self.theme = theme
        is_port = layout.frame_h > layout.frame_w
        raw_style = caption_style or os.environ.get("STUDIO_CAPTION_STYLE") or style or os.environ.get("STUDIO_UI")
        default_cap = "kinetic" if (is_port and (style in ("card", "neon"))) else ("editorial" if not is_port else "minimal")
        self.caption_style = normalize_caption_style(raw_style, default=default_cap)
        self.preset = CAPTION_PRESETS[self.caption_style]
        self.legacy_style = ui_style(style if style in UI_STYLES else None)
        # Preserve card/neon pill feel strictly when kinetic/card/neon is active
        self.style = (
            "card"
            if (self.caption_style == "kinetic" and style == "card") or self.caption_style in ("accessible", "dialogue")
            else ("neon" if (self.caption_style == "kinetic" and style == "neon") else "minimal")
        )
        self.minimal = not self.preset.use_box and self.caption_style in ("editorial", "minimal", "off")
        self.card_w = layout.safe_w
        self.card_h = max(0.45, layout.caption_h)
        self._speaker = "narrator"

        # Two-line captions need slightly tighter single-line cap height so 2 spaced rows stay balanced
        two_line_factor = 0.31 if self.preset.max_lines >= 2 else 0.44
        self.line_h = min(layout.min_text_h * self.LINE_H_SCALE, self.card_h * two_line_factor)

        if self.minimal or self.style == "neon":
            self.inner_w = self.card_w * 0.94
            self.inner_h = self.card_h
        else:
            self.inner_w = self.card_w - 0.56
            self.inner_h = self.card_h - 0.24

        self._ref_h = m.Text("Hg", font_size=48, font=CAPTION_FONT, weight="BOLD", **_text_font_kwargs()).height
        self.accent = theme.accent_primary if self.caption_style == "editorial" else theme.accent_highlight

        bg_opacity = 0.0
        border_w = 0.0
        if self.caption_style == "accessible":
            bg_opacity = 0.96
            border_w = 1.6
        elif self.caption_style == "dialogue":
            bg_opacity = 0.90
            border_w = 1.4
        elif self.style == "card" and self.caption_style == "kinetic":
            bg_opacity = 0.90
            border_w = 2.0

        self.bg = m.RoundedRectangle(
            corner_radius=0.16,
            width=self.card_w,
            height=self.card_h,
            fill_color=theme.surface if bg_opacity > 0 else theme.background,
            fill_opacity=bg_opacity,
            stroke_color=theme.surface_border,
            stroke_width=border_w,
        ).move_to(layout.caption_center)

        if self.style == "neon" and self.caption_style == "kinetic":
            bar_h = self.card_h * 1.06
            dock = m.Rectangle(
                width=layout.frame_w,
                height=bar_h,
                fill_color=theme.surface,
                fill_opacity=0.92,
                stroke_width=0.0,
            ).move_to(layout.caption_center)
            top_y = layout.caption_center[1] + bar_h * 0.5
            glow_line = m.Line(
                np.array([-layout.frame_w * 0.5, top_y, 0.0]),
                np.array([layout.frame_w * 0.5, top_y, 0.0]),
                color=theme.accent_primary,
                stroke_width=2.5,
                stroke_opacity=0.80,
            )
            self._neon_bar = m.VGroup(dock, glow_line)
        else:
            self._neon_bar = m.VGroup()

        self.pill: m.Mobject = m.VGroup()
        self.line: m.Mobject = m.VGroup()
        self._speaker_badge: m.Mobject = m.VGroup()
        self.group = m.VGroup(self._neon_bar, self.bg, self._speaker_badge, self.pill, self.line)

        self._chunks: list[tuple[float, list[tuple[str, float, float]], list[str]]] = []
        self._built: dict[int, tuple[m.Mobject, list[m.VGroup]]] = {}
        self._row_meta: dict[int, tuple[list[int], list[list[int]]]] = {}
        self._shown: tuple[int, int] = (-1, -1)
        self._pill_born = 0.0
        self._pill_scale = 1.0
        self._pill_center = np.zeros(3)
        self._bounce_part = None
        self._bounce_center = np.zeros(3)
        self._bounce_scale = 1.0
        self._pop_amp = 0.06
        self._bounce_amp = 0.0
        self._spring = False
        if self.caption_style != "off":
            self.set_words([(w, 0.0, 0.0, False) for w in initial_text.split()], 0.0)
            self._tick(0.0)
        else:
            self.group.submobjects = []
        self.group.add_updater(lambda _mob: self._tick(scene_time()))

    def _text(self, s: str) -> m.Text:
        is_light = getattr(self.theme, "is_light", False)
        txt = m.Text(
            s,
            font_size=48,
            font=CAPTION_FONT,
            weight="SEMIBOLD" if self.caption_style in ("editorial", "minimal") else "BOLD",
            color=self.theme.text_muted if self.caption_style == "kinetic" else self.theme.text_primary,
            disable_ligatures=True,
            **_text_font_kwargs(),
        )
        txt.scale(self.line_h / max(self._ref_h, 1e-6))
        if not self.preset.use_box:
            # Ground-matched soft halo keeps bare editorial glyphs legible without muddy borders
            stroke_w = self.OUTLINE_WIDTH * (0.65 if is_light else 1.05)
            txt.set_stroke(color=self.theme.background, width=stroke_w, opacity=0.88, background=True)
        return txt

    @staticmethod
    def _split_words_by_lines(words: list[str], lines_hint: list[str] | None) -> list[list[str]]:
        if not words:
            return []
        if not lines_hint or len(lines_hint) <= 1:
            return [list(words)]
        counts = [max(0, len(str(ln).split())) for ln in lines_hint]
        if sum(counts) <= 0:
            return [list(words)]
        rows: list[list[str]] = []
        cursor = 0
        for i, cnt in enumerate(counts):
            if cursor >= len(words):
                break
            take = (len(words) - cursor) if i == len(counts) - 1 else min(cnt, len(words) - cursor)
            if take > 0:
                rows.append(list(words[cursor : cursor + take]))
                cursor += take
        if cursor < len(words):
            if rows:
                rows[-1].extend(words[cursor:])
            else:
                rows.append(list(words[cursor:]))
        return rows or [list(words)]

    def _make_line(self, words: list[str], lines_hint: list[str] | None = None) -> tuple[m.Mobject, list[m.VGroup]]:
        rows = self._split_words_by_lines(words, lines_hint)
        row_mobs: list[m.Text] = []
        parts: list[m.VGroup] = []
        word_to_row: list[int] = []
        row_word_ids: list[list[int]] = []

        for r_idx, r_words in enumerate(rows):
            row_txt = self._text(" ".join(r_words))
            glyphs = list(row_txt.submobjects)
            cursor = 0
            r_ids: list[int] = []
            for w in r_words:
                n = len(w.replace(" ", "").replace("\n", ""))
                parts.append(m.VGroup(*glyphs[cursor : cursor + n]))
                r_ids.append(len(parts) - 1)
                word_to_row.append(r_idx)
                cursor += n
            row_word_ids.append(r_ids)
            row_mobs.append(row_txt)

        if len(row_mobs) <= 1:
            line: m.Mobject = row_mobs[0] if row_mobs else self._text(" ".join(words))
        else:
            line_gap = max(0.16, self.line_h * 0.54)
            grp = m.VGroup(*row_mobs)
            grp.arrange(m.DOWN, buff=line_gap)
            line = grp

        fit_mobject(line, max_width=self.inner_w, max_height=self.card_h * (0.86 if self.preset.show_speaker_badge else 0.94))
        y_off = -self.card_h * 0.10 if (self.preset.show_speaker_badge and len(self._speaker_badge.submobjects) > 0) else 0.0
        line.move_to(self.bg.get_center() + np.array([0.0, y_off, 0.0]))
        self._row_meta[id(line)] = (word_to_row, row_word_ids)
        return line, parts

    PHRASE_END = (".", ",", "!", "?", ";", ":", "—")
    SOFT_BREAK_WORDS = {"and", "or", "but", "so", "because", "while", "where", "when", "then", "which", "that", "plus", "equals"}

    def set_words(self, words: list[tuple], beat_start: float) -> None:
        """Pack timed words into balanced phrase chunks via CaptionIR segmentation."""
        from studio.captions.layout import chunk_caption_cues
        from studio.audio.tts import SyncedBeat

        self._chunks = []
        self._built = {}
        if self.caption_style == "off" or not words:
            self.group.submobjects = []
            return
        start_s = float(words[0][1]) if len(words[0]) > 1 else beat_start
        end_s = max(start_s + 0.3, float(words[-1][2]) if len(words[-1]) > 2 else start_s + 2.0)
        b = SyncedBeat(
            id="_w",
            narration=" ".join(str(w[0]) for w in words),
            caption="",
            audio_path="",
            start_time=start_s,
            end_time=end_s,
            speech_duration=max(0.2, end_s - start_s),
            total_duration=max(0.2, end_s - start_s),
            words=[list(w) for w in words],
            marks={},
            speaker=self._speaker,
        )
        is_port = self.layout.frame_h > self.layout.frame_w
        max_chars = self.preset.portrait_max_chars if is_port else self.preset.landscape_max_chars
        cues = chunk_caption_cues([b], max_chars_per_line=max_chars, max_lines=self.preset.max_lines)
        for c in cues:
            self._chunks.append((float(c.start_s), [(w.text, w.start_s, w.end_s, w.emphasis) for w in c.words], list(c.lines)))
        if self._chunks:
            self._chunks[0] = (beat_start, self._chunks[0][1], self._chunks[0][2])
        self._shown = (-1, -1)

    def set_segments(self, segments: list, beat_start: float) -> None:
        """Bind precomputed semantic CaptionSegment / CaptionCue list cleanly."""
        self._chunks = []
        self._built = {}
        if self.caption_style == "off":
            self.group.submobjects = []
            return
        for seg in segments:
            lns = list(getattr(seg, "lines", []) or [])
            if getattr(seg, "words", None):
                wds = []
                for w in seg.words:
                    if isinstance(w, tuple) or isinstance(w, list):
                        wds.append((str(w[0]), float(w[1]), float(w[2]), bool(w[3]) if len(w) > 3 else False))
                    else:
                        wds.append((str(w.text), float(w.start_s), float(w.end_s), bool(w.emphasis)))
                self._chunks.append((float(seg.start_s), wds, lns))
            else:
                toks = seg.text.replace("\n", " ").split()
                if not toks:
                    continue
                span = max(0.1, float(seg.end_s) - float(seg.start_s))
                step = span / len(toks)
                wds = [(t, float(seg.start_s) + i * step, float(seg.start_s) + (i + 1) * step, False) for i, t in enumerate(toks)]
                self._chunks.append((float(seg.start_s), wds, lns or [" ".join(toks)]))
        if self._chunks:
            self._chunks[0] = (beat_start, self._chunks[0][1], self._chunks[0][2])
        self._shown = (-1, -1)

    def set_beat(
        self,
        words: list,
        narration: str,
        start: float,
        speech_end: float,
        speaker: str = "narrator",
    ) -> None:
        """Bind a beat using CaptionIR single source of truth across captions & subtitles."""
        from studio.audio.tts import SyncedBeat
        from studio.captions.layout import chunk_caption_cues

        self._speaker = speaker or "narrator"
        if self.caption_style == "off":
            self._chunks = []
            self.group.submobjects = []
            return

        # Optional quiet speaker badge for dialogue/accessible tracks
        if self.preset.show_speaker_badge and speaker:
            sp_lbl = speaker.replace("_", " ").title()
            sp_col = self.theme.accent_secondary if speaker.lower() not in ("narrator", "host", "explainer") else self.theme.accent_primary
            badge_t = m.Text(sp_lbl.upper(), font_size=20, font=HEADER_FONT, weight="BOLD", color=sp_col, **_text_font_kwargs())
            badge_t.scale(min(1.0, (self.card_h * 0.22) / max(badge_t.height, 1e-6)))
            badge_bg = m.RoundedRectangle(
                corner_radius=0.08,
                width=badge_t.width + 0.22,
                height=badge_t.height + 0.10,
                fill_color=self.theme.surface,
                fill_opacity=0.92,
                stroke_color=sp_col,
                stroke_width=1.2,
            )
            badge = m.VGroup(badge_bg, badge_t).move_to(
                self.bg.get_top() + np.array([0.0, badge_t.height * 0.45 + 0.04, 0.0])
            )
            self._speaker_badge = badge
        else:
            self._speaker_badge = m.VGroup()

        timed = [tuple(w) for w in words] if words else []
        if not timed:
            toks = narration.split()
            total = sum(len(t) + 1 for t in toks) or 1
            span = max(0.1, speech_end - start)
            cur = start
            for tok in toks:
                d = span * (len(tok) + 1) / total
                timed.append((tok, cur, cur + d, False))
                cur += d
        beat = SyncedBeat(
            id="_live",
            narration=narration,
            caption=narration,
            audio_path="",
            start_time=start,
            end_time=speech_end,
            speech_duration=max(0.1, speech_end - start),
            total_duration=max(0.1, speech_end - start),
            words=[list(w) for w in timed],
            marks={},
            speaker=self._speaker,
        )
        is_port = self.layout.frame_h > self.layout.frame_w
        max_chars = self.preset.portrait_max_chars if is_port else self.preset.landscape_max_chars
        cues = chunk_caption_cues([beat], max_chars_per_line=max_chars, max_lines=self.preset.max_lines)
        if cues:
            self.set_segments(cues, start)
        else:
            self.set_words(timed, start)
        self._tick(scene_time())

    # -- per-frame drive ------------------------------------------------------
    def _tick(self, t: float) -> None:
        if self.caption_style == "off" or not self._chunks:
            return
        ci = 0
        for idx, (c_start, _, _lns) in enumerate(self._chunks):
            if t + 1e-6 >= c_start:
                ci = idx
        words = self._chunks[ci][1]
        lines_hint = self._chunks[ci][2] if len(self._chunks[ci]) > 2 else None
        wi = 0
        for j, w in enumerate(words):
            if t + 1e-6 >= w[1]:
                wi = j
        if (ci, wi) != self._shown:
            self._show(ci, wi, words, lines_hint, t)
        self._animate(t)

    def _show(self, ci: int, wi: int, words: list[tuple], lines_hint: list[str] | None, t: float) -> None:
        if ci not in self._built:
            self._built[ci] = self._make_line([w[0] for w in words], lines_hint=lines_hint)
        line, parts = self._built[ci]
        self._reset_bounce()
        emph = words[wi][3] if len(words[wi]) > 3 else False
        mode = self.preset.word_highlight_mode
        sp_col = (
            self.theme.accent_secondary
            if self._speaker.lower() not in ("narrator", "host", "explainer", "")
            else self.theme.accent_primary
        )
        for j, part in enumerate(parts):
            if mode == "none":
                # Clean uniform broadcast subtitle text with zero distracting color flash
                part.set_fill(color=self.theme.text_primary)
            elif mode == "subtle_ink":
                if j == wi:
                    part.set_fill(color=sp_col if (self.preset.show_speaker_badge or emph) else self.theme.text_primary)
                elif j < wi:
                    part.set_fill(color=self.theme.text_primary)
                else:
                    part.set_fill(color=self.theme.text_muted)
            else:
                if self.style == "neon":
                    part.set_stroke(color=self.theme.background, width=self.OUTLINE_WIDTH, opacity=0.90, background=True)
                if j == wi:
                    if self.minimal or self.style == "neon":
                        part.set_fill(color=self.theme.accent_primary if emph else self.accent)
                    else:
                        part.set_color(self.theme.background)
                elif j < wi:
                    part.set_fill(color=self.theme.accent_primary if (len(words[j]) > 3 and words[j][3]) else self.theme.text_primary)
                else:
                    part.set_fill(color=self.theme.text_muted)

        if mode in ("none", "subtle_ink"):
            self.group.submobjects = [self._neon_bar, self.bg, self._speaker_badge, line]
            self.pill, self.line = m.VGroup(), line
            self._bounce_part = None
            self._pop_amp = 0.0
            self._bounce_amp = 0.0
            self._shown = (ci, wi)
            return

        if self.minimal or self.style == "neon" or not parts or wi >= len(parts) or len(parts[wi].submobjects) == 0:
            self._show_minimal(ci, wi, line, parts, emph, t)
            return

        word_to_row, row_word_ids = self._row_meta.get(id(line), ([0] * len(parts), [list(range(len(parts)))]))
        r_idx = word_to_row[wi] if wi < len(word_to_row) else 0
        is_multi_row = isinstance(line, m.VGroup) and len(row_word_ids) > 1 and len(line.submobjects) == len(row_word_ids)
        if is_multi_row and r_idx < len(line.submobjects):
            row_mob = line.submobjects[r_idx]
            row_ys = [float(rm.get_center()[1]) for rm in line.submobjects]
            row_pitch = min(abs(a - b) for a, b in zip(row_ys, row_ys[1:]))
            max_pill_h = max(float(parts[wi].height) + 0.04, row_pitch * 0.78)
        else:
            row_mob = line
            max_pill_h = max(float(row_mob.height) * 1.28, self.line_h * 1.22)

        r_ids = row_word_ids[r_idx] if r_idx < len(row_word_ids) else [wi]
        pos_in_row = r_ids.index(wi) if wi in r_ids else 0
        w_left = float(parts[wi].get_left()[0])
        w_right = float(parts[wi].get_right()[0])
        avail_l = (w_left - float(parts[r_ids[pos_in_row - 1]].get_right()[0])) if (pos_in_row > 0 and len(parts[r_ids[pos_in_row - 1]].submobjects) > 0) else self.line_h * 0.25
        avail_r = (float(parts[r_ids[pos_in_row + 1]].get_left()[0]) - w_right) if (pos_in_row + 1 < len(r_ids) and len(parts[r_ids[pos_in_row + 1]].submobjects) > 0) else self.line_h * 0.25
        min_gap = max(0.0, min(avail_l, avail_r))
        pad_x = max(0.024, min(self.line_h * 0.085, min_gap * 0.36))
        pill_w = max(0.08, float(parts[wi].width) + 2.0 * pad_x)

        row_cy = float(row_mob.get_center()[1])
        base_h = max(float(parts[wi].height) + 0.08, float(row_mob.height) * 1.14, self.line_h * 1.06)
        pill_h = max(min(base_h, max_pill_h), float(parts[wi].height) + 0.04)
        half_h = 0.5 * pill_h
        cy_lo = float(parts[wi].get_top()[1]) + 0.018 - half_h
        cy_hi = float(parts[wi].get_bottom()[1]) - 0.018 + half_h
        if cy_lo <= cy_hi:
            pill_cy = min(max(row_cy, cy_lo), cy_hi)
        else:
            pill_cy = float(parts[wi].get_center()[1])

        pill_col = self.theme.accent_primary if emph else self.accent
        pad_y = max(0.02, 0.5 * (pill_h - float(parts[wi].height)))
        pill = m.BackgroundRectangle(
            parts[wi],
            buff=(pad_x, pad_y),
            corner_radius=min(self.line_h * 0.16, pill_h * 0.32),
            color=pill_col,
            fill_opacity=0.95,
            stroke_width=0.0,
        ).move_to(np.array([float(parts[wi].get_center()[0]), pill_cy, 0.0]))
        self.group.submobjects = [self.bg, self._speaker_badge, pill, line]
        self.pill, self.line = pill, line
        self._has_pill = True
        self._pill_center = pill.get_center().copy()
        self._pill_born = t
        self._pill_scale = 1.0
        self._bounce_part = parts[wi] if emph else None
        self._bounce_center = parts[wi].get_center().copy()
        self._bounce_scale = 1.0
        max_expand_half = max(0.0, min_gap * 0.22)
        pop_cap = (2.0 * max_expand_half) / max(pill_w, 1e-6)
        self._pop_amp = min(0.10 if emph else 0.055, pop_cap)
        self._bounce_amp = min(0.08, max(0.0, (pill_w / max(float(parts[wi].width), 1e-6) - 1.0) * 0.65))
        self._shown = (ci, wi)

    def _show_minimal(self, ci: int, wi: int, line: m.Mobject, parts: list[m.VGroup], emph: bool, t: float) -> None:
        """Box-free or kinetic minimal: active word pops lightly without neighbor overlap."""
        self.group.submobjects = [self._neon_bar, self.bg, self._speaker_badge, line]
        self.pill, self.line = m.VGroup(), line
        self._has_pill = False
        self._pill_born = t
        self._pill_scale = 1.0
        self._bounce_part = parts[wi] if (parts and wi < len(parts)) else None
        self._bounce_center = self._bounce_part.get_center().copy() if self._bounce_part is not None else np.zeros(3)
        self._bounce_scale = 1.0
        space = self.line_h * 0.28
        w_part = float(self._bounce_part.width) if self._bounce_part is not None else 1.0
        cap = 0.75 * space / max(w_part, 1e-6)
        self._pop_amp = 0.0
        self._bounce_amp = min(0.14 if emph else 0.07, cap)
        self._spring = emph
        if self.style == "neon" and self._bounce_part is not None:
            glow_color = self.theme.accent_primary if emph else self.accent
            self._bounce_part.set_stroke(color=glow_color, width=self.OUTLINE_WIDTH * 1.35, opacity=0.42, background=True)
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
        if getattr(self, "_has_pill", self.style == "card") and self._pop_amp > 0.0:
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
