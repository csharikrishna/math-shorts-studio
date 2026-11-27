"""ASS (Advanced SubStation Alpha) subtitle exporter styled by LookTokens & CaptionIR."""

from __future__ import annotations

from pathlib import Path

from studio.captions.model import CaptionIR
from studio.design.spec import LookTokens, resolve_look


def _hex_to_ass_bgr(hex_color: str, alpha_hex: str = "00") -> str:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        return f"&H{alpha_hex}FFFFFF"
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha_hex}{bb}{gg}{rr}".upper()


def _fmt_ass_ts(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def export_ass(
    ir: CaptionIR,
    out_path: Path,
    *,
    look_name: str = "midnight_studio",
    look_tokens: LookTokens | None = None,
) -> Path:
    """Export ASS subtitles styled cleanly to light/dark theme tokens and CaptionIR."""
    lk = look_tokens or resolve_look(look_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    primary = _hex_to_ass_bgr(lk.ink_primary, "00")
    secondary = _hex_to_ass_bgr(lk.accent_primary, "00")
    outline = _hex_to_ass_bgr(lk.ground if lk.is_light else "#05070B", "00")
    box_alpha = "20" if ir.style in ("accessible", "dialogue") else "D8"
    back = _hex_to_ass_bgr(lk.surface, box_alpha)
    border_style = 3 if (ir.preset and ir.preset.use_box) else 1
    outline_px = 0 if border_style == 3 else (2 if lk.is_light else 3)
    margin_v = max(28, int(ir.reserve_band_px * 0.30))
    margin_lr = max(48, int(ir.width_px * 0.06))

    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {ir.width_px}",
            f"PlayResY: {ir.height_px}",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,Source Sans 3,{ir.font_px},{primary},{secondary},{outline},{back},"
            f"-1,0,0,0,100,100,0,0,{border_style},{outline_px},0,2,{margin_lr},{margin_lr},{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    events: list[str] = []
    if ir.style != "off":
        show_sp = bool(ir.preset and ir.preset.show_speaker_badge and len(ir.speaker_labels) > 1)
        for cue in ir.cues:
            txt = r"\N".join(cue.lines)
            if show_sp and cue.speaker_label:
                txt = f"{{\\b1\\c{_hex_to_ass_bgr(lk.accent_secondary)}}}{cue.speaker_label}: {{\\b0\\c{primary}}}{txt}"
            events.append(
                f"Dialogue: 0,{_fmt_ass_ts(cue.start_s)},{_fmt_ass_ts(cue.end_s)},Default,{cue.speaker},0,0,0,,{txt}"
            )

    out_path.write_text(header + "\n" + "\n".join(events) + "\n", encoding="utf-8")
    return out_path
