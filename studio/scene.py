"""Beat-synced Scene base classes (2D and 3D) integrating AudioTimeline, SFX, Theme, and SafeLayout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence
import manimgx as m
from studio.audio.sfx import SfxLibrary
from studio.audio.tts import AudioTimeline, BeatSpec, parse_script
from studio.config import FORMATS, FormatProfile, THEMES, Theme
from studio.layout import CaptionCard, SafeLayout, scene_time

# Lead time of the transition riser so its swell peaks exactly on the next beat's downbeat
RISER_LEAD_S = 0.55
# Fallback beat length used only when rendering without a TTS timeline (draft previews)
DRAFT_BEAT_S = 2.5


class _StoryMixin:
    """Shared beat-synchronization, theme, layout, and SFX orchestration for 2D & 3D scenes."""

    BEATS: list[BeatSpec] = []
    DEFAULT_FORMAT: str = "shorts_hd"
    DEFAULT_THEME: str = "midnight_studio"

    def setup_studio(self) -> None:
        """Load format profile, theme, layout zones, SFX library, and audio timeline."""
        fmt_name = os.environ.get("STUDIO_FORMAT", self.DEFAULT_FORMAT)
        theme_name = os.environ.get("STUDIO_THEME", self.DEFAULT_THEME)
        build_dir = Path(os.environ.get("STUDIO_BUILD_DIR", "build"))

        self.profile: FormatProfile = FORMATS[fmt_name]
        self.theme: Theme = THEMES[theme_name]

        m.config.background_color = self.theme.background
        self.camera.background_color = self.theme.background  # type: ignore[attr-defined]

        self.layout = SafeLayout.from_config(self.profile)
        self.sfx_lib = SfxLibrary(build_dir / "sfx")

        timeline_path = build_dir / "timeline.json"
        self.timeline: AudioTimeline | None = (
            AudioTimeline.load(timeline_path) if timeline_path.exists() else None
        )
        self.caption_card: CaptionCard | None = None
        self._draft_cursor = 0.0

    def now(self) -> float:
        """Exact scene time in seconds."""
        return scene_time()

    def sfx(self, name: str, time_offset: float = 0.0, gain_db: float = -4.0) -> None:
        """Trigger a cached procedural studio sound effect."""
        path = self.sfx_lib.get(name)
        self.add_sound(str(path), time_offset=max(0.0, time_offset), gain=gain_db)  # type: ignore[attr-defined]

    def _beat_window(self, beat_id: str) -> tuple[float, float, float, str, str, list, dict]:
        """(start, speech_end, end, caption, narration, words, marks) for a beat, in absolute seconds."""
        if self.timeline is not None:
            b = self.timeline.by_id(beat_id)
            return (b.start_time, b.start_time + b.speech_duration, b.end_time, b.caption, b.narration,
                    b.words, b.marks)
        spec = next(b for b in self.BEATS if b.id == beat_id)
        clean, tokens, mark_idx = parse_script(spec.narration)
        start = self._draft_cursor
        self._draft_cursor += DRAFT_BEAT_S
        speech = DRAFT_BEAT_S - spec.pause_after
        n = max(1, len(tokens))
        marks = {k: start + speech * i / n for k, i in mark_idx.items()}
        return start, start + speech, start + DRAFT_BEAT_S, spec.caption, clean, [], marks

    def play_beat(
        self,
        beat_id: str,
        *animations: m.Animation,
        hold: Sequence[m.Animation] | Callable[[], Sequence[m.Animation]] = (),
        cues: dict[str, Callable[[], Sequence[m.Animation]]] | None = None,
        caption_color: str | None = None,
        border_color: str | None = None,
        sfx: str | None = None,
        sfx_gain: float = -4.0,
        riser: bool = True,
        anim_ratio: float = 0.62,
        rate_func: Callable[[float], float] | None = None,
    ) -> None:
        """Execute a narrative beat locked to the exact TTS window (zero drift).

        The beat ends exactly at its timeline end: the duration is measured from the live
        scene clock, so the first beat absorbs the voice lead-in (no blank opening frame)
        and any rounding never accumulates. The tail of the beat plays `hold` animations
        (micro-emphasis) while scene updaters keep the stage in continuous motion.

        `cues` maps a narration bookmark (<bookmark mark='name'/>) to a factory of animations
        that start exactly when that word is spoken (manim-voiceover style), each running until
        the next cue or the beat's end.
        """
        start, speech_end, end, caption_text, narration, words, marks = self._beat_window(beat_id)
        target_dur = max(0.2, end - self.now())

        if self.caption_card is not None:
            self.caption_card.set_beat(words, narration, start, speech_end)
        self.add_subcaption(caption_text, duration=max(0.1, end - start), offset=max(0.0, start - self.now()))  # type: ignore[attr-defined]

        if sfx is not None:
            self.sfx(sfx, time_offset=max(0.0, start - self.now()), gain_db=sfx_gain)

        beat_anims = list(animations)
        if self.caption_card is not None:
            beat_anims.extend(
                self.caption_card.transition(caption_text, color=caption_color, border_color=border_color)
            )

        cue_times = sorted(
            (max(self.now(), float(marks[name])), factory)
            for name, factory in (cues or {}).items()
            if name in marks and float(marks[name]) < end - 0.05
        )

        if beat_anims:
            budget = target_dur * anim_ratio
            if cue_times:
                budget = min(budget, cue_times[0][0] - self.now())
            run_t = max(0.35, min(budget, target_dur - 0.10))
            kwargs = {"run_time": run_t}
            if rate_func is not None:
                kwargs["rate_func"] = rate_func
            self.play(*beat_anims, **kwargs)  # type: ignore[attr-defined]

        for idx, (cue_t, factory) in enumerate(cue_times):
            if cue_t - self.now() > 0.02:
                self.wait(cue_t - self.now())  # type: ignore[attr-defined]
            seg_end = cue_times[idx + 1][0] if idx + 1 < len(cue_times) else end
            cue_anims = list(factory())
            if cue_anims and seg_end - self.now() > 0.05:
                self.play(*cue_anims, run_time=seg_end - self.now())  # type: ignore[attr-defined]

        remainder = end - self.now()
        is_last = bool(self.BEATS) and beat_id == self.BEATS[-1].id
        if riser and not is_last and remainder > 0.2:
            self.sfx("riser_soft", time_offset=remainder - RISER_LEAD_S, gain_db=-13.0)

        if remainder > 0.02:
            hold_anims = list(hold() if callable(hold) else hold)
            if hold_anims:
                self.play(*hold_anims, run_time=remainder)  # type: ignore[attr-defined]
            else:
                self.wait(remainder)  # type: ignore[attr-defined]


class StoryScene(_StoryMixin, m.Scene):
    """Beat-synchronized 2D educational scene."""


class Story3DScene(_StoryMixin, m.ThreeDScene):
    """Beat-synchronized 3D multivariable calculus / spatial geometry scene."""
