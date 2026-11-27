"""Quantitative Design Specification (DESIGN.md) adapted from HyperFrames spec methodology."""

from __future__ import annotations

from pydantic import BaseModel, Field

from studio.config import FORMATS, THEMES, Theme
from studio.design.fonts import FONT_SPECS, effective_px_scale


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
    display_font: str = FONT_SPECS["display"]["css_stack"]
    body_font: str = FONT_SPECS["body"]["css_stack"]
    serif_font: str = FONT_SPECS["serif"]["css_stack"]
    math_font: str = FONT_SPECS["math"]["css_stack"]
    mono_font: str = "'Consolas', 'Cascadia Mono', monospace"
    title_px: int = 54
    subtitle_px: int = 32
    body_px: int = 30
    math_label_px: int = 36
    equation_px: int = 60
    caption_px: int = 44
    min_readable_px: int = 28
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


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (128, 128, 128)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rel_luminance(rgb: tuple[int, int, int]) -> float:
    def _chan(v: int) -> float:
        s = v / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _chan(r) + 0.7152 * _chan(g) + 0.0722 * _chan(b)


def wcag_contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """Compute WCAG 2.1 contrast ratio between two hex colors."""
    l1 = _rel_luminance(_hex_to_rgb(fg_hex))
    l2 = _rel_luminance(_hex_to_rgb(bg_hex))
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def validate_look_contrast(look: LookTokens) -> list[str]:
    """Audit primary/muted text & semantic accent contrast ratios on ground & surface."""
    issues: list[str] = []
    for role, col, req in (
        ("ink_primary", look.ink_primary, 7.0),
        ("ink_muted", look.ink_muted, 4.5),
        ("accent_primary", look.accent_primary, 4.5),
        ("accent_secondary", look.accent_secondary, 4.5),
        ("accent_tertiary", look.accent_tertiary, 4.5),
        ("accent_highlight", look.accent_highlight, 4.5),
        ("success", look.success, 4.5),
        ("danger", look.danger, 4.5),
    ):
        for bg_name, bg_hex in (("ground", look.ground), ("surface", look.surface)):
            ratio = wcag_contrast_ratio(col, bg_hex)
            if ratio < req:
                issues.append(f"Look '{look.name}' {role} ({col}) on {bg_name} ({bg_hex}) contrast {ratio:.2f} < {req:.1f}")
    return issues


def look_from_theme(theme: Theme) -> LookTokens:
    """Build a synchronized LookTokens model directly from a Theme dataclass."""
    return LookTokens(
        name=theme.name,
        ground=theme.background,
        surface=theme.surface,
        surface_border=theme.surface_border,
        ink_primary=theme.text_primary,
        ink_muted=theme.text_muted,
        accent_primary=theme.accent_primary,
        accent_secondary=theme.accent_secondary,
        accent_tertiary=theme.accent_tertiary,
        accent_highlight=theme.accent_highlight,
        success=theme.success,
        danger=theme.danger,
        grid_line=theme.grid_line,
        medium=theme.medium,
        is_light=theme.is_light,
    )


NAMED_LOOKS: dict[str, LookTokens] = {name: look_from_theme(th) for name, th in THEMES.items()}


def resolve_look(look_name: str = "midnight_studio") -> LookTokens:
    """Resolve light/dark look aliases cleanly without coercing light themes to dark."""
    alias = {
        "editorial_light": "light",
        "cinematic_dark": "dark",
    }.get(look_name, look_name)
    return NAMED_LOOKS.get(alias, NAMED_LOOKS["midnight_studio"])


def default_design_spec(format_name: str = "widescreen_hd", look_name: str = "midnight_studio") -> DesignSpec:
    """Build a responsive, aspect-aware DesignSpec for portrait (9:16) or landscape (16:9)."""
    profile = FORMATS.get(format_name, FORMATS["widescreen_hd"])
    look = resolve_look(look_name)
    is_vert = profile.is_vertical
    px_scale = effective_px_scale(profile.pixel_width, profile.pixel_height)
    typo = TypographySpec(
        title_px=px_scale.title_px,
        subtitle_px=px_scale.subtitle_px,
        body_px=px_scale.body_px,
        math_label_px=px_scale.math_label_px,
        equation_px=px_scale.equation_px,
        caption_px=px_scale.caption_px,
        min_readable_px=px_scale.min_readable_px,
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
