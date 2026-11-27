"""Pinned offline SIL Open Font License (OFL-1.1) typography & scaling utilities."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from studio.ai.env import PROJECT_ROOT

FONT_DIR: Path = PROJECT_ROOT / "assets" / "fonts"
MANIFEST_PATH: Path = FONT_DIR / "FONT_MANIFEST.json"

MANIM_DISPLAY_FONT = "Inter"
MANIM_BODY_FONT = "Source Sans 3"
MANIM_SERIF_FONT = "STIX Two Text"

FONT_SPECS: dict[str, dict[str, str]] = {
    "display": {
        "family": "Inter",
        "file": "Inter-Variable.ttf",
        "format": "truetype",
        "license": "OFL-1.1",
        "css_stack": "'Inter', 'Source Sans 3', system-ui, sans-serif",
    },
    "body": {
        "family": "Source Sans 3",
        "file": "SourceSans3-Variable.ttf",
        "format": "truetype",
        "license": "OFL-1.1",
        "css_stack": "'Source Sans 3', 'Inter', system-ui, sans-serif",
    },
    "serif": {
        "family": "STIX Two Text",
        "file": "STIXTwoText-Variable.ttf",
        "format": "truetype",
        "license": "OFL-1.1",
        "css_stack": "'STIX Two Text', Georgia, serif",
    },
    "math": {
        "family": "STIX Two Math",
        "file": "STIX2Math.otf",
        "format": "opentype",
        "license": "OFL-1.1",
        "css_stack": "'STIX Two Math', 'STIX Two Text', serif",
    },
}


def manimgx_font_paths() -> list[str]:
    """Return local font directory list for offline deterministic ManimGX / Typst rendering."""
    return [str(FONT_DIR)] if FONT_DIR.exists() else []


def verify_pinned_fonts() -> list[str]:
    """Verify all pinned font files exist locally and match SHA-256 sums in FONT_MANIFEST.json."""
    issues: list[str] = []
    if not MANIFEST_PATH.exists():
        return [f"Missing font manifest: {MANIFEST_PATH}"]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for role, meta in FONT_SPECS.items():
        fname = meta["file"]
        fpath = FONT_DIR / fname
        if not fpath.exists():
            issues.append(f"Missing pinned font for role={role}: {fpath}")
            continue
        expected = manifest.get(fname, {}).get("sha256")
        if expected:
            actual = hashlib.sha256(fpath.read_bytes()).hexdigest().lower()
            if actual != str(expected).lower():
                issues.append(f"SHA-256 mismatch for {fname}: expected {expected}, got {actual}")
    return issues


def font_face_css(url_prefix: str = "/fonts/", embed_base64: bool = False) -> str:
    """Generate offline @font-face declarations for local Chromium/HyperFrames rendering."""
    rules: list[str] = []
    for role, meta in FONT_SPECS.items():
        fpath = FONT_DIR / meta["file"]
        if not fpath.exists():
            continue
        fmt = meta["format"]
        if embed_base64:
            mime = "font/ttf" if fmt == "truetype" else "font/otf"
            b64 = base64.b64encode(fpath.read_bytes()).decode("ascii")
            src = f"url('data:{mime};base64,{b64}') format('{fmt}')"
        else:
            prefix = url_prefix.rstrip("/") + "/"
            src = f"url('{prefix}{meta['file']}') format('{fmt}')"
        weight_range = "300 800" if "Variable" in meta["file"] else "400"
        rules.append(
            f"@font-face {{\n"
            f"  font-family: '{meta['family']}';\n"
            f"  src: {src};\n"
            f"  font-weight: {weight_range};\n"
            f"  font-style: normal;\n"
            f"  font-display: block;\n"
            f"}}"
        )
    return "\n".join(rules)


@dataclass(frozen=True)
class PixelTypographyScale:
    """Target rendered pixel sizes derived from frame dimensions (1080p baseline)."""

    title_px: int
    subtitle_px: int
    body_px: int
    math_label_px: int
    equation_px: int
    caption_px: int
    speaker_px: int
    min_readable_px: int


def effective_px_scale(width_px: int, height_px: int) -> PixelTypographyScale:
    """Compute aspect-aware legible pixel typography floors calibrated to 1080p."""
    is_vert = height_px > width_px
    short_side = min(width_px, height_px)
    scale = max(0.65, short_side / 1080.0)
    if is_vert:
        return PixelTypographyScale(
            title_px=max(44, int(round(58 * scale))),
            subtitle_px=max(28, int(round(34 * scale))),
            body_px=max(26, int(round(32 * scale))),
            math_label_px=max(30, int(round(38 * scale))),
            equation_px=max(48, int(round(62 * scale))),
            caption_px=max(44, int(round(52 * scale))),
            speaker_px=max(22, int(round(24 * scale))),
            min_readable_px=max(26, int(round(30 * scale))),
        )
    return PixelTypographyScale(
        title_px=max(40, int(round(54 * scale))),
        subtitle_px=max(26, int(round(32 * scale))),
        body_px=max(24, int(round(30 * scale))),
        math_label_px=max(26, int(round(36 * scale))),
        equation_px=max(44, int(round(62 * scale))),
        caption_px=max(36, int(round(46 * scale))),
        speaker_px=max(20, int(round(24 * scale))),
        min_readable_px=max(24, int(round(28 * scale))),
    )


def px_to_scene_units(target_px: float, canvas_height_px: int, frame_height_units: float = 8.0) -> float:
    """Convert pixel height directly to ManimGX scene units."""
    return float(target_px) * (float(frame_height_units) / max(1.0, float(canvas_height_px)))


def px_to_manim_font_size(target_px: float, canvas_height_px: int = 1080) -> int:
    """Map target 1080p-calibrated pixel size cleanly into ManimGX font_size points."""
    ratio = 1080.0 / max(360.0, float(canvas_height_px))
    return max(22, int(round(float(target_px) * ratio * 1.15)))


def estimate_text_width_px(text: str, font_px: float, bold: bool = False) -> float:
    """Conservative typographic width estimator for multi-line phrase chunking & layout."""
    if not text:
        return 0.0
    factor = 0.56 if bold else 0.52
    wide_chars = sum(1 for ch in text if ch in "MWQG@#%&=+—")
    narrow_chars = sum(1 for ch in text if ch in "ilIjtfr.,:;!'| ")
    eff_chars = len(text) + 0.35 * wide_chars - 0.35 * narrow_chars
    return max(0.0, eff_chars * float(font_px) * factor)


FontRole = Literal["display", "body", "serif", "math"]
