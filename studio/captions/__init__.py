"""Unified caption & subtitle intermediate representation (CaptionIR)."""

from __future__ import annotations

from studio.captions.ass import export_ass
from studio.captions.layout import (
    build_caption_ir,
    chunk_caption_cues,
    export_caption_ir_files,
    merge_chapter_caption_irs,
)
from studio.captions.model import (
    CAPTION_PRESETS,
    CaptionCue,
    CaptionIR,
    CaptionPresetConfig,
    CaptionStyle,
    CaptionWord,
    normalize_caption_style,
)

__all__ = [
    "CAPTION_PRESETS",
    "CaptionCue",
    "CaptionIR",
    "CaptionPresetConfig",
    "CaptionStyle",
    "CaptionWord",
    "build_caption_ir",
    "chunk_caption_cues",
    "export_ass",
    "export_caption_ir_files",
    "merge_chapter_caption_irs",
    "normalize_caption_style",
]
