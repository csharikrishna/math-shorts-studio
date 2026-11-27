"""Format profiles, safe zones, and cohesive design themes."""

from dataclasses import dataclass
from typing import Dict

# ---------------------------------------------------------------------------
# Caption font – previously ManimGX defaulted to "New Computer Modern" (serif).
# This one global switches all Text() calls to a clean, punchy sans-serif that
# looks great on Shorts/TikTok.  Segoe UI ships on every Windows 10/11 machine;
# set STUDIO_CAPTION_FONT in the environment to override.
import os as _os
CAPTION_FONT: str = _os.environ.get("STUDIO_CAPTION_FONT", "Segoe UI")
HEADER_FONT: str  = _os.environ.get("STUDIO_HEADER_FONT", "Segoe UI")

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
    """Semantic color tokens for coherent mathematical storytelling."""

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


THEMES: Dict[str, Theme] = {
    "midnight_studio": Theme(
        name="midnight_studio",
        background="#0B1020",
        surface="#131B2E",
        surface_border="#2B3B5E",
        text_primary="#F4F7FB",
        text_muted="#9AA8C7",
        accent_primary="#38BDF8",    # Electric Sky Cyan
        accent_secondary="#F59E0B",  # Warm Amber Gold
        accent_tertiary="#A78BFA",   # Soft Violet
        accent_highlight="#FDE047",  # Radiant Lemon
        success="#34D399",           # Emerald Mint
        danger="#FB7185",            # Coral Rose
        grid_line="#1E293B",
    ),
    "obsidian_chalk": Theme(
        name="obsidian_chalk",
        background="#090B0E",
        surface="#141820",
        surface_border="#2E3646",
        text_primary="#F8FAFC",
        text_muted="#94A3B8",
        accent_primary="#2DD4BF",    # Teal
        accent_secondary="#FB923C",  # Tangerine
        accent_tertiary="#60A5FA",   # Cobalt
        accent_highlight="#FACC15",  # Gold
        success="#4ADE80",
        danger="#F87171",
        grid_line="#1F2937",
    ),
}
