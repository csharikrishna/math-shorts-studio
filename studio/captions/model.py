"""Typed CaptionIR data models and cohesive subtitle style presets."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

CaptionStyle = Literal["editorial", "kinetic", "minimal", "dialogue", "accessible", "off"]
TimingProvenance = Literal["forced_aligned", "tts_word_boundary", "interpolated", "estimated"]


class CaptionWord(BaseModel):
    text: str
    start_s: float
    end_s: float
    emphasis: bool = False
    spoken_token: str = ""
    timing_provenance: TimingProvenance = "tts_word_boundary"


class CaptionCue(BaseModel):
    cue_id: str
    beat_id: str
    speaker: str = "narrator"
    speaker_label: str = ""
    start_s: float
    end_s: float
    lines: list[str] = Field(default_factory=list)
    words: list[CaptionWord] = Field(default_factory=list)
    original_text: str = ""
    spoken_text: str = ""
    timing_provenance: TimingProvenance = "tts_word_boundary"

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class CaptionPresetConfig(BaseModel):
    style: CaptionStyle
    description: str
    landscape_max_chars: int
    portrait_max_chars: int
    max_lines: int
    use_box: bool
    use_pill_highlight: bool
    show_speaker_badge: bool
    word_highlight_mode: Literal["none", "subtle_ink", "pop_accent", "invert_pill"]
    reserve_band_ratio_landscape: float
    reserve_band_ratio_portrait: float
    min_contrast_ratio: float


CAPTION_PRESETS: dict[CaptionStyle, CaptionPresetConfig] = {
    "editorial": CaptionPresetConfig(
        style="editorial",
        description="Quiet box-free editorial subtitles with soft contrast halo and subtle accent weight (no yellow karaoke pill).",
        landscape_max_chars=46,
        portrait_max_chars=28,
        max_lines=2,
        use_box=False,
        use_pill_highlight=False,
        show_speaker_badge=False,
        word_highlight_mode="subtle_ink",
        reserve_band_ratio_landscape=0.105,
        reserve_band_ratio_portrait=0.145,
        min_contrast_ratio=4.5,
    ),
    "kinetic": CaptionPresetConfig(
        style="kinetic",
        description="Crisp active-word pop tuned for Shorts/TikTok pacing without full-width permanent dock bars.",
        landscape_max_chars=34,
        portrait_max_chars=24,
        max_lines=2,
        use_box=False,
        use_pill_highlight=True,
        show_speaker_badge=False,
        word_highlight_mode="pop_accent",
        reserve_band_ratio_landscape=0.115,
        reserve_band_ratio_portrait=0.155,
        min_contrast_ratio=4.5,
    ),
    "minimal": CaptionPresetConfig(
        style="minimal",
        description="Clean single- or two-line broadcast subtitles with balanced line breaks and stroke outline.",
        landscape_max_chars=48,
        portrait_max_chars=30,
        max_lines=2,
        use_box=False,
        use_pill_highlight=False,
        show_speaker_badge=False,
        word_highlight_mode="none",
        reserve_band_ratio_landscape=0.10,
        reserve_band_ratio_portrait=0.14,
        min_contrast_ratio=4.5,
    ),
    "dialogue": CaptionPresetConfig(
        style="dialogue",
        description="Two-speaker editorial captions with restrained speaker badge and speaker-coded ink accent.",
        landscape_max_chars=44,
        portrait_max_chars=28,
        max_lines=2,
        use_box=True,
        use_pill_highlight=False,
        show_speaker_badge=True,
        word_highlight_mode="subtle_ink",
        reserve_band_ratio_landscape=0.125,
        reserve_band_ratio_portrait=0.16,
        min_contrast_ratio=4.5,
    ),
    "accessible": CaptionPresetConfig(
        style="accessible",
        description="High-contrast AAA backing strip (>= 7:1) optimized for dense formulas or mobile outdoor viewing.",
        landscape_max_chars=42,
        portrait_max_chars=26,
        max_lines=2,
        use_box=True,
        use_pill_highlight=False,
        show_speaker_badge=True,
        word_highlight_mode="none",
        reserve_band_ratio_landscape=0.125,
        reserve_band_ratio_portrait=0.165,
        min_contrast_ratio=7.0,
    ),
    "off": CaptionPresetConfig(
        style="off",
        description="No burned-in captions; bottom band collapses cleanly while SRT/VTT/ASS sidecars remain exported.",
        landscape_max_chars=48,
        portrait_max_chars=30,
        max_lines=2,
        use_box=False,
        use_pill_highlight=False,
        show_speaker_badge=False,
        word_highlight_mode="none",
        reserve_band_ratio_landscape=0.055,
        reserve_band_ratio_portrait=0.08,
        min_contrast_ratio=4.5,
    ),
}


def normalize_caption_style(raw: str | None, *, default: CaptionStyle = "editorial") -> CaptionStyle:
    """Map legacy UI choices cleanly into canonical CaptionStyle presets."""
    if not raw:
        return default
    cleaned = raw.strip().lower()
    legacy_map: dict[str, CaptionStyle] = {
        "card": "kinetic",
        "neon": "kinetic",
    }
    if cleaned in legacy_map:
        return legacy_map[cleaned]
    if cleaned in CAPTION_PRESETS:
        return cleaned  # type: ignore[return-value]
    return default


class CaptionIR(BaseModel):
    """Canonical caption track shared by SRT, VTT, ASS, ManimGX, and HyperFrames."""

    style: CaptionStyle = "editorial"
    aspect_label: Literal["16:9", "9:16"] = "16:9"
    width_px: int = 1920
    height_px: int = 1080
    font_px: int = 44
    max_width_px: float = 1480.0
    reserve_band_px: int = 114
    speaker_labels: dict[str, str] = Field(default_factory=dict)
    cues: list[CaptionCue] = Field(default_factory=list)
    preset: Optional[CaptionPresetConfig] = None
    timing_provenance_counts: dict[str, int] = Field(default_factory=dict)
    raw_timing_evidence_count: int = 0
    timing_repairs: list[dict] = Field(default_factory=list)
    source_timeline_hash: str = ""
