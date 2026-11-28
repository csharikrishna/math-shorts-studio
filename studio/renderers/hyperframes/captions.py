"""CaptionIR serialization and browser subtitle CSS builders for HyperFrames."""

from __future__ import annotations

from typing import Any

from studio.captions import CaptionIR, build_caption_ir, normalize_caption_style
from studio.design.spec import DesignSpec


def build_hyperframes_captions(
    synced_beats: list[Any],
    design: DesignSpec,
    caption_style: str | None = None,
    *,
    caption_ir: CaptionIR | None = None,
) -> tuple[CaptionIR, str, list[dict[str, Any]]]:
    """Build or consume precompiled CaptionIR and serialize identical cue payload cleanly."""
    sz = design.safe_zone
    cap_style = normalize_caption_style(caption_style, default="editorial")
    cap_ir = caption_ir if caption_ir is not None else build_caption_ir(
        synced_beats,
        style=cap_style,
        width_px=sz.width,
        height_px=sz.height,
    )
    cap_style = cap_ir.style
    cues_payload = [
        {
            "cue_id": c.cue_id,
            "beat_id": c.beat_id,
            "speaker": c.speaker,
            "speaker_label": c.speaker_label,
            "start": round(c.start_s, 4),
            "end": round(c.end_s, 4),
            "lines": list(c.lines),
            "timing_provenance": c.timing_provenance,
            "words": [
                {
                    "w": w.text,
                    "s": round(w.start_s, 4),
                    "e": round(w.end_s, 4),
                    "emph": bool(w.emphasis),
                    "prov": w.timing_provenance,
                }
                for w in c.words
            ],
        }
        for c in cap_ir.cues
    ]
    return cap_ir, cap_style, cues_payload
