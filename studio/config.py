"""Format profiles, safe zones, and cohesive design themes."""

from dataclasses import dataclass
from typing import Dict

# ---------------------------------------------------------------------------
# Pinned offline OFL-1.1 typography defaults (assets/fonts).
# Override via STUDIO_CAPTION_FONT / STUDIO_HEADER_FONT if needed.
import os as _os
CAPTION_FONT: str = _os.environ.get("STUDIO_CAPTION_FONT", "Source Sans 3")
HEADER_FONT: str = _os.environ.get("STUDIO_HEADER_FONT", "Inter")
MATH_SERIF_FONT: str = _os.environ.get("STUDIO_MATH_SERIF_FONT", "STIX Two Text")

CAPTION_STYLE_CHOICES = ["editorial", "kinetic", "minimal", "dialogue", "accessible", "off"]

# ---------------------------------------------------------------------------
# UI style names and their descriptions (registered here so pipeline.py can
# expose them as CLI choices without importing components.py at import time).
UI_STYLE_CHOICES = ["minimal", "card", "neon"]
# minimal – bare outlined words, no box (default, Shorts/TikTok)
# card    – frosted-glass pill behind each caption line
# neon    – fullwidth gradient bottom bar, word pops with a neon glow


@dataclass(frozen=True)
class FormatProfile:
    """Video output specification and UI safe-zone margins (in normalized frame ratios)."""

    name: str
    pixel_width: int
    pixel_height: int
    fps: int
    top_safe_ratio: float
    bottom_safe_ratio: float
    side_safe_ratio: float

    @property
    def is_vertical(self) -> bool:
        return self.pixel_height > self.pixel_width

    @property
    def resolution_str(self) -> str:
        return f"{self.pixel_width}x{self.pixel_height}"


FORMATS: Dict[str, FormatProfile] = {
    "shorts_hd": FormatProfile(
        name="shorts_hd",
        pixel_width=1080,
        pixel_height=1920,
        fps=60,
        top_safe_ratio=0.10,
        bottom_safe_ratio=0.16,
        side_safe_ratio=0.06,
    ),
    "shorts_fast": FormatProfile(
        name="shorts_fast",
        pixel_width=720,
        pixel_height=1280,
        fps=30,
        top_safe_ratio=0.10,
        bottom_safe_ratio=0.16,
        side_safe_ratio=0.06,
    ),
    "widescreen_hd": FormatProfile(
        name="widescreen_hd",
        pixel_width=1920,
        pixel_height=1080,
        fps=60,
        top_safe_ratio=0.06,
        bottom_safe_ratio=0.08,
        side_safe_ratio=0.05,
    ),
    "widescreen_fast": FormatProfile(
        name="widescreen_fast",
        pixel_width=1280,
        pixel_height=720,
        fps=30,
        top_safe_ratio=0.06,
        bottom_safe_ratio=0.08,
        side_safe_ratio=0.05,
    ),
}


@dataclass(frozen=True)
class Theme:
    """Semantic color tokens for coherent light and dark mathematical storytelling."""

    name: str
    background: str
    surface: str
    surface_border: str
    text_primary: str
    text_muted: str
    accent_primary: str
    accent_secondary: str
    accent_tertiary: str
    accent_highlight: str
    success: str
    danger: str
    grid_line: str
    is_light: bool = False
    medium: str = "Vector stage with crisp high-contrast ink strokes"


THEMES: Dict[str, Theme] = {
    "light": Theme(
        name="light",
        background="#F7F6F2",        # Warm off-white editorial paper
        surface="#ECE8DF",           # Subtle warm stone card
        surface_border="#CEC7BA",    # Fine hairline divider
        text_primary="#172033",      # Deep ink navy
        text_muted="#475569",        # Accessible cool slate (>= 6.5:1 on light ground)
        accent_primary="#1D4ED8",    # Deep cobalt accent (>= 6.1:1)
        accent_secondary="#0D6A63",  # Rich deep teal (>= 5.2:1 on surface)
        accent_tertiary="#9A3412",   # Warm rust/coral (>= 6.7:1)
        accent_highlight="#8E3B06",  # Editorial deep amber ink (>= 6.1:1 on surface)
        success="#166534",           # Crisp emerald green (>= 6.0:1 on surface)
        danger="#B91C1C",            # Deep vermilion
        grid_line="#E3DED4",         # Subtle drafting grid
        is_light=True,
        medium="Warm off-white editorial drafting paper with cobalt, teal, and rust ink",
    ),
    "dark": Theme(
        name="dark",
        background="#0D1421",        # Deep midnight navy slate
        surface="#151F32",           # Elevated obsidian panel
        surface_border="#2B3B59",    # Refined slate hairline
        text_primary="#F1F5F9",      # Crisp off-white ink
        text_muted="#94A3B8",        # Muted slate grey
        accent_primary="#60A5FA",    # Cool sky blue
        accent_secondary="#5EEAD4",  # Soft cyan-teal
        accent_tertiary="#F59E0B",   # Warm amber accent
        accent_highlight="#FACC15",  # High-legibility gold emphasis
        success="#4ADE80",           # Clean emerald mint
        danger="#FB7185",            # Soft coral rose
        grid_line="#1B273C",         # Restrained midnight grid
        is_light=False,
        medium="Deep midnight slate canvas with cool blue, teal, and amber highlights",
    ),
    "midnight_studio": Theme(
        name="midnight_studio",
        background="#0B1020",
        surface="#131B2E",
        surface_border="#2B3B5E",
        text_primary="#F4F7FB",
        text_muted="#9AA8C7",
        accent_primary="#38BDF8",
        accent_secondary="#F59E0B",
        accent_tertiary="#A78BFA",
        accent_highlight="#FDE047",
        success="#34D399",
        danger="#FB7185",
        grid_line="#1E293B",
        is_light=False,
        medium="Deep slate vector stage with crisp high-contrast ink strokes",
    ),
    "obsidian_chalk": Theme(
        name="obsidian_chalk",
        background="#090B0E",
        surface="#141820",
        surface_border="#2E3646",
        text_primary="#F8FAFC",
        text_muted="#94A3B8",
        accent_primary="#2DD4BF",
        accent_secondary="#FB923C",
        accent_tertiary="#60A5FA",
        accent_highlight="#FACC15",
        success="#4ADE80",
        danger="#F87171",
        grid_line="#1F2937",
        is_light=False,
        medium="Charcoal chalkboard with warm amber and teal drafting lines",
    ),
    "cartesian_stone": Theme(
        name="cartesian_stone",
        background="#EDE8E0",
        surface="#FAF7F2",
        surface_border="#B8B0A4",
        text_primary="#1A1A1A",
        text_muted="#4B5563",
        accent_primary="#1D4ED8",
        accent_secondary="#8E3B06",
        accent_tertiary="#6D28D9",
        accent_highlight="#0D6A63",
        success="#166534",
        danger="#B91C1C",
        grid_line="#D9D2C7",
        is_light=True,
        medium="Warm stone drafting paper with 1px hairline structure and restrained cobalt ink",
    ),
    "editorial_paper": Theme(
        name="editorial_paper",
        background="#FAF9F5",
        surface="#EFE9DE",
        surface_border="#CFC6B8",
        text_primary="#141413",
        text_muted="#57534E",
        accent_primary="#1D4ED8",
        accent_secondary="#9A3412",
        accent_tertiary="#0D6A63",
        accent_highlight="#8E3B06",
        success="#166534",
        danger="#B91C1C",
        grid_line="#E6DFD3",
        is_light=True,
        medium="Cream research-note paper with coral/cobalt emphasis and serif/sans editorial pairing",
    ),
}
