"""CaptionIR serialization and browser subtitle CSS builders for HyperFrames."""

from __future__ import annotations

from typing import Any

from studio.captions import CaptionIR, build_caption_ir, normalize_caption_style
from studio.design.spec import DesignSpec


def build_hyperframes_captions(
    synced_beats: list[Any],
    design: DesignSpec,
    caption_style: str | None = None,
) -> tuple[CaptionIR, str, list[dict[str, Any]]]:
    """Build CaptionIR and JSON payload synced with exact speech timings."""
    sz = design.safe_zone
    default_cap = "kinetic" if sz.aspect_label == "9:16" else "editorial"
    cap_style = normalize_caption_style(caption_style, default=default_cap)
    cap_ir = build_caption_ir(
        synced_beats,
        style=cap_style,
        width_px=sz.width,
        height_px=sz.height,
    )
    cues_payload = [
        {
            "cue_id": c.cue_id,
            "beat_id": c.beat_id,
            "speaker": c.speaker,
            "speaker_label": c.speaker_label,
            "start": round(c.start_s, 4),
            "end": round(c.end_s, 4),
            "lines": list(c.lines),
            "words": [
                {"w": w.text, "s": round(w.start_s, 4), "e": round(w.end_s, 4), "emph": bool(w.emphasis)}
                for w in c.words
            ],
        }
        for c in cap_ir.cues
    ]
    return cap_ir, cap_style, cues_payload
