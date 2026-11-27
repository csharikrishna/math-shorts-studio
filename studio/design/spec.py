"""Quantitative Design Specification (DESIGN.md) adapted from HyperFrames spec methodology."""

from __future__ import annotations

from pydantic import BaseModel, Field

from studio.config import FORMATS, THEMES


BANNED_DEFAULTS: list[str] = [
    "No unmotivated black/navy empty voids with tiny centered symbols.",
    "No permanent boxed equation cards stamped identically across every scene.",
    "No decorative ambient particle swarms or bloom/glow disconnected from math state.",
    "No repetitive identical slide-up fade-ins for every beat.",
    "No three-card KPI clichés or bullet-point textbook slides.",
    "No squashing a 16:9 composition uniformly into 9:16 portrait frames.",
]


class LookTokens(BaseModel):
    name: str
    ground: str
    surface: str
    surface_border: str
    ink_primary: str
    ink_muted: str
    accent_primary: str
    accent_secondary: str
    accent_tertiary: str
    accent_highlight: str
    success: str
    danger: str
    grid_line: str
    medium: str = "Vector chalkboard & crisp editorial ink overlay"
    grain_pct: float = 0.0
    vignette_pct: float = 0.0
    is_light: bool = False


class TypographySpec(BaseModel):
    display_font: str = "'Inter', 'Segoe UI', system-ui, sans-serif"
    body_font: str = "'Inter', 'Segoe UI', system-ui, sans-serif"
    mono_font: str = "'Consolas', 'Cascadia Mono', monospace"
    title_px: int = 52
    subtitle_px: int = 28
    body_px: int = 26
    math_label_px: int = 34
    caption_px: int = 38
    min_readable_px: int = 26
    line_height: float = 1.25


class MotionTimingSpec(BaseModel):
    enter_ease: str = "easeOutExpo"
    move_ease: str = "easeInOutCubic"
    exit_ease: str = "easeInCubic"
    spring_damping: float = 0.78
    spring_stiffness: float = 180.0
    enter_s: float = 0.45
    exit_s: float = 0.25
    stagger_ms: int = 75
    hold_min_s: float = 0.40


class SafeZoneSpec(BaseModel):
    width: int
    height: int
    fps: int
    aspect_label: str
    action_safe_margin_pct: float = 5.0
    title_safe_margin_pct: float = 10.0
    top_band_px: int
    bottom_caption_band_px: int
    side_margin_px: int
    prefer_stacked_comparison: bool = False


class DesignSpec(BaseModel):
    format_name: str
    look: LookTokens
    typography: TypographySpec
    motion: MotionTimingSpec = Field(default_factory=MotionTimingSpec)
    safe_zone: SafeZoneSpec
    banned_defaults: list[str] = Field(default_factory=lambda: list(BANNED_DEFAULTS))


NAMED_LOOKS: dict[str, LookTokens] = {
    "midnight_studio": LookTokens(
        name="midnight_studio",
        ground=THEMES["midnight_studio"].background,
        surface=THEMES["midnight_studio"].surface,
        surface_border=THEMES["midnight_studio"].surface_border,
        ink_primary=THEMES["midnight_studio"].text_primary,
        ink_muted=THEMES["midnight_studio"].text_muted,
        accent_primary=THEMES["midnight_studio"].accent_primary,
        accent_secondary=THEMES["midnight_studio"].accent_secondary,
        accent_tertiary=THEMES["midnight_studio"].accent_tertiary,
        accent_highlight=THEMES["midnight_studio"].accent_highlight,
        success=THEMES["midnight_studio"].success,
        danger=THEMES["midnight_studio"].danger,
        grid_line=THEMES["midnight_studio"].grid_line,
        medium="Deep slate vector stage with crisp high-contrast ink strokes",
        is_light=False,
    ),
    "obsidian_chalk": LookTokens(
        name="obsidian_chalk",
        ground=THEMES["obsidian_chalk"].background,
        surface=THEMES["obsidian_chalk"].surface,
        surface_border=THEMES["obsidian_chalk"].surface_border,
        ink_primary=THEMES["obsidian_chalk"].text_primary,
        ink_muted=THEMES["obsidian_chalk"].text_muted,
        accent_primary=THEMES["obsidian_chalk"].accent_primary,
        accent_secondary=THEMES["obsidian_chalk"].accent_secondary,
        accent_tertiary=THEMES["obsidian_chalk"].accent_tertiary,
        accent_highlight=THEMES["obsidian_chalk"].accent_highlight,
        success=THEMES["obsidian_chalk"].success,
        danger=THEMES["obsidian_chalk"].danger,
        grid_line=THEMES["obsidian_chalk"].grid_line,
        medium="Charcoal chalkboard with warm amber/teal drafting lines",
        is_light=False,
    ),
    "cartesian_stone": LookTokens(
        name="cartesian_stone",
        ground="#EDE8E0",
        surface="#FAF7F2",
        surface_border="#B8B0A4",
        ink_primary="#1A1A1A",
        ink_muted="#5A5A5A",
        accent_primary="#2563EB",
        accent_secondary="#B45309",
        accent_tertiary="#7C3AED",
        accent_highlight="#0F766E",
        success="#15803D",
        danger="#B91C1C",
        grid_line="#D9D2C7",
        medium="Warm stone drafting paper with 1px hairline structure and restrained cobalt ink",
        grain_pct=0.0,
        is_light=True,
    ),
    "editorial_paper": LookTokens(
        name="editorial_paper",
        ground="#FAF9F5",
        surface="#EFE9DE",
        surface_border="#D5CCBE",
        ink_primary="#141413",
        ink_muted="#57534E",
        accent_primary="#1D4ED8",
        accent_secondary="#CC785C",
        accent_tertiary="#0F766E",
        accent_highlight="#B45309",
        success="#15803D",
        danger="#DC2626",
        grid_line="#E6DFD3",
        medium="Cream research-note paper with coral emphasis and tabular monospace figures",
        is_light=True,
    ),
}


def default_design_spec(format_name: str = "widescreen_hd", look_name: str = "midnight_studio") -> DesignSpec:
    """Build a responsive, aspect-aware DesignSpec for portrait (9:16) or landscape (16:9)."""
    profile = FORMATS.get(format_name, FORMATS["widescreen_hd"])
    look = NAMED_LOOKS.get(look_name, NAMED_LOOKS["midnight_studio"])
    is_vert = profile.is_vertical
    typo = TypographySpec(
        title_px=56 if is_vert else 48,
        subtitle_px=30 if is_vert else 26,
        body_px=28 if is_vert else 24,
        math_label_px=36 if is_vert else 30,
        caption_px=42 if is_vert else 34,
        min_readable_px=28 if is_vert else 24,
    )
    safe = SafeZoneSpec(
        width=profile.pixel_width,
        height=profile.pixel_height,
        fps=profile.fps,
        aspect_label="9:16" if is_vert else "16:9",
        top_band_px=int(profile.pixel_height * profile.top_safe_ratio),
        bottom_caption_band_px=int(profile.pixel_height * profile.bottom_safe_ratio),
        side_margin_px=int(profile.pixel_width * profile.side_safe_ratio),
        prefer_stacked_comparison=is_vert,
    )
    return DesignSpec(format_name=profile.name, look=look, typography=typo, safe_zone=safe)


def export_design_md(spec: DesignSpec) -> str:
    """Render human-readable DESIGN.md adapted from HyperFrames SPEC.md methodology."""
    lk = spec.look
    tp = spec.typography
    sz = spec.safe_zone
    mt = spec.motion
    lines = [
        f"# DESIGN SPECIFICATION ({spec.format_name})",
        "",
        "## Output & Safe Zones",
        f"- **Canvas:** `{sz.width}x{sz.height}` (`{sz.aspect_label}`) @ `{sz.fps} fps`",
        f"- **Safe Zones:** Top band `{sz.top_band_px}px`, Bottom caption reserve `{sz.bottom_caption_band_px}px`, Side margin `{sz.side_margin_px}px`",
        f"- **Comparison Layout:** `{'stacked_comparison' if sz.prefer_stacked_comparison else 'side_by_side_comparison'}`",
        "",
        "## Look Tokens",
        f"- **Look Name:** `{lk.name}` ({lk.medium})",
        f"- **Ground / Surface / Border:** `{lk.ground}` / `{lk.surface}` / `{lk.surface_border}`",
        f"- **Ink Primary / Muted:** `{lk.ink_primary}` / `{lk.ink_muted}`",
        f"- **Accents (Primary / Secondary / Tertiary / Highlight):** `{lk.accent_primary}` / `{lk.accent_secondary}` / `{lk.accent_tertiary}` / `{lk.accent_highlight}`",
        f"- **Semantic (Success / Danger / Grid):** `{lk.success}` / `{lk.danger}` / `{lk.grid_line}`",
        "",
        "## Typography Scale",
        f"- **Display / Body:** `{tp.display_font}` / `{tp.body_font}`",
        f"- **Mono:** `{tp.mono_font}`",
        f"- **Sizes:** Title `{tp.title_px}px`, Subtitle `{tp.subtitle_px}px`, Body `{tp.body_px}px`, Math `{tp.math_label_px}px`, Caption `{tp.caption_px}px` (Floor `{tp.min_readable_px}px`)",
        "",
        "## Motion & Easing",
        f"- **Curves (Enter / Move / Exit):** `{mt.enter_ease}` (`{mt.enter_s}s`) / `{mt.move_ease}` / `{mt.exit_ease}` (`{mt.exit_s}s`)",
        f"- **Spring:** `damping={mt.spring_damping}`, `stiffness={mt.spring_stiffness}`, `stagger={mt.stagger_ms}ms`, `hold_min={mt.hold_min_s}s`",
        "",
        "## Banned Defaults",
        *[f"- {b}" for b in spec.banned_defaults],
        "",
    ]
    return "\n".join(lines)
