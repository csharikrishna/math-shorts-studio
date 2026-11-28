"""Shared adaptive composition, dynamic region collapse, and collision-free layout solver.

Used by both ManimGX (`scenes/v3_scene.py`) and HyperFrames (`studio/renderers/hyperframes/`)
so diagram framing, aspect adaptation, caption reserve, and non-overlapping label bounds
stay identical across renderers and never starve diagrams to <25% frame width.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from studio.captions.model import CAPTION_PRESETS, normalize_caption_style
from studio.config import FORMATS, FormatProfile
from studio.registry.compositions import adapt_composition_for_aspect


@dataclass(frozen=True)
class PixelBox:
    """Axis-aligned pixel box with x, y top-left origin."""

    x: float
    y: float
    w: float
    h: float
    role: str = "generic"

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + 0.5 * self.w

    @property
    def cy(self) -> float:
        return self.y + 0.5 * self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def overlaps(self, other: "PixelBox", pad: float = 8.0) -> bool:
        return not (
            self.x2 + pad <= other.x
            or other.x2 + pad <= self.x
            or self.y2 + pad <= other.y
            or other.y2 + pad <= self.y
        )


@dataclass(frozen=True)
class AdaptiveFrameZones:
    """Dynamic regions computed after collapsing unused chrome (headers/captions)."""

    width_px: int
    height_px: int
    aspect_label: Literal["16:9", "9:16"]
    side_margin_px: float
    top_band_px: float
    equation_band_px: float
    caption_band_px: float
    stage_box: PixelBox
    equation_box: PixelBox
    caption_box: PixelBox

    @property
    def stage_width_fraction(self) -> float:
        return self.stage_box.w / max(1.0, float(self.width_px))

    @property
    def stage_height_fraction(self) -> float:
        return self.stage_box.h / max(1.0, float(self.height_px))


def compute_adaptive_zones(
    format_name: str = "widescreen_hd",
    *,
    has_header_banner: bool = False,
    has_equation_strip: bool = False,
    caption_style: str = "editorial",
) -> AdaptiveFrameZones:
    """Collapse unused top/bottom bands so visual diagrams fill 55-82% of stage height cleanly."""
    prof: FormatProfile = FORMATS.get(format_name, FORMATS["widescreen_hd"])
    w = float(prof.pixel_width)
    h = float(prof.pixel_height)
    is_vert = prof.is_vertical
    aspect: Literal["16:9", "9:16"] = "9:16" if is_vert else "16:9"

    side_margin = w * prof.side_safe_ratio
    safe_w = w - 2.0 * side_margin

    norm_cap = normalize_caption_style(caption_style)
    preset = CAPTION_PRESETS[norm_cap]
    cap_ratio = preset.reserve_band_ratio_portrait if is_vert else preset.reserve_band_ratio_landscape
    cap_band = h * cap_ratio
    bot_pad = h * (prof.bottom_safe_ratio * 0.25 if norm_cap == "off" else 0.0)
    caption_h = max(0.0 if norm_cap == "off" else 76.0, cap_band)

    if has_header_banner:
        top_band = h * (0.21 if is_vert else 0.16)
    elif has_equation_strip:
        top_band = max(h * prof.top_safe_ratio, h * (0.11 if is_vert else 0.082))
    else:
        top_band = h * prof.top_safe_ratio

    eq_band = h * (0.105 if is_vert else 0.105) if has_equation_strip else 0.0
    gap = h * (0.022 if is_vert else 0.018)

    stage_top = top_band + eq_band + (gap if (has_equation_strip or has_header_banner) else 0.0)
    stage_bot = h - cap_band - bot_pad - gap
    stage_h = max(h * 0.42, stage_bot - stage_top)

    eq_box = PixelBox(
        x=side_margin,
        y=top_band,
        w=safe_w,
        h=max(1.0, eq_band),
        role="equation",
    )
    cap_box = PixelBox(
        x=side_margin,
        y=h - caption_h - bot_pad - 6.0,
        w=safe_w,
        h=max(1.0, caption_h),
        role="caption",
    )
    stage_box = PixelBox(
        x=side_margin,
        y=stage_top,
        w=safe_w,
        h=stage_h,
        role="stage",
    )
    return AdaptiveFrameZones(
        width_px=prof.pixel_width,
        height_px=prof.pixel_height,
        aspect_label=aspect,
        side_margin_px=side_margin,
        top_band_px=top_band,
        equation_band_px=eq_band,
        caption_band_px=caption_h,
        stage_box=stage_box,
        equation_box=eq_box,
        caption_box=cap_box,
    )


def resolve_composition_layout(
    composition: str,
    object_kinds: Sequence[str],
    aspect_label: str = "16:9",
) -> str:
    """Resolve content-aware layout archetype respecting aspect ratio and object mix."""
    comp = (composition or "auto").strip()
    if comp in ("diagram_with_text", "annotated_chart", "annotated_overlay", "math_simulation", "model_with_readout", "progressive_derivation", "vector_field_with_trajectory"):
        comp = "diagram_with_text"
    elif comp in ("geometric_proof", "matrix_transform", "chapter_intro", "concept_summary", "dual_time_series"):
        comp = "centered_focus"
    elif comp in (
        "data_visualization",
        "time_frequency_split",
        "before_after_simulation",
        "multi_view_dashboard",
        "physical_system_with_displacement",
        "physical_system_with_phase_space",
        "parameter_sweep_with_response",
        "state_with_conserved_chart",
    ):
        comp = "stacked_comparison" if aspect_label == "9:16" else "side_by_side_comparison"
    comp = adapt_composition_for_aspect(comp, aspect_label)
    if comp != "auto":
        return comp

    has_axes = "axes" in object_kinds
    has_mat_or_tbl = any(k in ("matrix", "table") for k in object_kinds)
    has_eq = "equation" in object_kinds
    n_ann = sum(1 for k in object_kinds if k == "annotation")
    n_major = sum(1 for k in object_kinds if k in ("axes", "matrix", "table", "bar_chart", "graph_network"))
    if n_major >= 2:
        return "stacked_comparison" if aspect_label == "9:16" else "side_by_side_comparison"
    if has_axes and has_mat_or_tbl:
        return "stacked_comparison" if aspect_label == "9:16" else "side_by_side_comparison"
    if has_axes and n_ann >= 2 and aspect_label == "16:9":
        return "diagram_with_text"
    if has_eq and not has_axes and n_major == 0:
        return "centered_focus"
    if has_eq and has_axes:
        return "main_with_equation_panel"
    return "centered_focus"


def fit_axes_pixel_box(
    zones: AdaptiveFrameZones,
    *,
    xr: Sequence[float],
    yr: Sequence[float],
    equal_aspect: bool,
    composition: str,
    panel_slot: str = "auto",
    has_side_annotations: bool = False,
) -> PixelBox:
    """Compute non-overlapping, well-proportioned axes panel preserving gutter for ticks & axis labels."""
    st = zones.stage_box
    is_vert = zones.aspect_label == "9:16"
    dx_data = max(abs(float(xr[1]) - float(xr[0])), 1e-6)
    dy_data = max(abs(float(yr[1]) - float(yr[0])), 1e-6)

    slot = panel_slot
    if slot == "auto" and composition in ("side_by_side_comparison", "stacked_comparison"):
        slot = "left" if not is_vert else "top"

    # Strictly disjoint panel rectangles with explicit center gutter (never overlap left/right!)
    if slot == "left" and not is_vert:
        region = PixelBox(st.x, st.y, st.w * 0.465, st.h, role="axes")
    elif slot == "right" and not is_vert:
        region = PixelBox(st.x + st.w * 0.535, st.y, st.w * 0.465, st.h, role="axes")
    elif slot in ("left", "top") and (is_vert or composition == "stacked_comparison"):
        region = PixelBox(st.x, st.y, st.w, st.h * 0.465, role="axes")
    elif slot in ("right", "bottom") and (is_vert or composition == "stacked_comparison"):
        region = PixelBox(st.x, st.y + st.h * 0.535, st.w, st.h * 0.465, role="axes")
    elif slot == "top" and not is_vert:
        region = PixelBox(st.x, st.y, st.w, st.h * 0.465, role="axes")
    elif slot == "bottom" and not is_vert:
        region = PixelBox(st.x, st.y + st.h * 0.535, st.w, st.h * 0.465, role="axes")
    elif composition == "diagram_with_text" and not is_vert and has_side_annotations:
        region = PixelBox(st.x, st.y, st.w * 0.62, st.h * 0.94, role="axes")
    else:
        region = PixelBox(st.x, st.y, st.w * (0.94 if is_vert else 0.88), st.h * 0.92, role="axes")

    # Reserve inner breathing room so tick labels & axis names stay cleanly inside `region`
    pad_w_ratio = 0.86 if slot in ("left", "right") else 0.90
    pad_h_ratio = 0.80 if zones.equation_band_px > 0 else 0.84

    if equal_aspect:
        scale = min(region.w * pad_w_ratio / dx_data, region.h * pad_h_ratio / dy_data)
        w_out = dx_data * scale
        h_out = dy_data * scale
    else:
        w_out = region.w * pad_w_ratio
        h_out = region.h * pad_h_ratio

    offset_x = 0.0
    if composition == "diagram_with_text" and not is_vert and has_side_annotations and slot == "auto":
        offset_x = -(zones.width_px * 0.08)

    cx = min(max(region.cx + offset_x, region.x + w_out * 0.5 + 8.0), region.x2 - w_out * 0.5 - 8.0)
    cy = min(max(region.cy, region.y + h_out * 0.5 + 8.0), region.y2 - h_out * 0.5 - 8.0)
    return PixelBox(x=cx - 0.5 * w_out, y=cy - 0.5 * h_out, w=w_out, h=h_out, role="axes")


def clamp_box_to_safe_zone(box: PixelBox, zones: AdaptiveFrameZones, *, pad_px: float = 12.0) -> PixelBox:
    """Keep any overlay/annotation/equation strictly within safe visual frame boundaries."""
    min_x = zones.side_margin_px + pad_px
    max_x = float(zones.width_px) - zones.side_margin_px - pad_px
    min_y = max(14.0, zones.top_band_px * 0.55)
    bottom_clear = 28.0 if zones.aspect_label == "9:16" else 18.0
    max_y = zones.caption_box.y - bottom_clear

    w = min(box.w, max(40.0, max_x - min_x))
    h = min(box.h, max(24.0, max_y - min_y))
    x = min(max(box.x, min_x), max_x - w)
    y = min(max(box.y, min_y), max_y - h)
    return PixelBox(x=x, y=y, w=w, h=h, role=box.role)


def resolve_non_overlapping_boxes(
    desired: list[PixelBox],
    obstacles: Sequence[PixelBox],
    zones: AdaptiveFrameZones,
    *,
    gap_px: float = 20.0,
) -> list[PixelBox]:
    """Iteratively adjust auxiliary boxes (annotations, matrices, equations) to prevent overlaps and clipping."""
    placed: list[PixelBox] = []
    blockers = list(obstacles)

    for raw_box in desired:
        cand = clamp_box_to_safe_zone(raw_box, zones)
        if not any(cand.overlaps(b, pad=gap_px) for b in blockers):
            placed.append(cand)
            blockers.append(cand)
            continue

        # Probe alternate clean slots: right margin, left margin, below stage, above stage
        st = zones.stage_box
        probes: list[tuple[float, float]] = []
        # Step vertically below/above desired position
        for step in (1, -1, 2, -2, 3, -3):
            probes.append((cand.x, cand.y + step * (cand.h + gap_px)))
        # Right editorial rail (for landscape)
        if zones.aspect_label == "16:9":
            rail_x = st.x2 - cand.w - 8.0
            left_x = st.x + 8.0
            for k in range(5):
                py = st.y + k * (cand.h + gap_px * 1.1)
                probes.append((rail_x, py))
                probes.append((left_x, py))
        else:
            for k in range(4):
                probes.append((st.cx - 0.5 * cand.w, st.y + k * (cand.h + gap_px)))
                probes.append((st.cx - 0.5 * cand.w, st.y2 - (k + 1) * (cand.h + gap_px)))

        best = cand
        min_pen = float("inf")
        for px, py in probes:
            test = clamp_box_to_safe_zone(PixelBox(px, py, cand.w, cand.h, role=cand.role), zones)
            overlap_area = 0.0
            for b in blockers:
                ix = max(0.0, min(test.x2 + gap_px, b.x2) - max(test.x, b.x))
                iy = max(0.0, min(test.y2 + gap_px, b.y2) - max(test.y, b.y))
                overlap_area += ix * iy
            dist_pen = 0.02 * (abs(test.cx - raw_box.cx) + abs(test.cy - raw_box.cy))
            score = overlap_area * 100.0 + dist_pen
            if overlap_area <= 1.0:
                best = test
                break
            if score < min_pen:
                min_pen = score
                best = test
        placed.append(best)
        blockers.append(best)

    return placed


def px_box_to_manim(
    box: PixelBox,
    zones: AdaptiveFrameZones,
    frame_w: float,
    frame_h: float,
) -> tuple[float, float, float, float]:
    """Convert pixel box (top-left origin) to Manim world `(center_x, center_y, width, height)`."""
    sx = frame_w / float(zones.width_px)
    sy = frame_h / float(zones.height_px)
    cx = (box.cx - 0.5 * zones.width_px) * sx
    cy = (0.5 * zones.height_px - box.cy) * sy
    return cx, cy, box.w * sx, box.h * sy
