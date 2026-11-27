"""HTML/SVG component layout helpers and scene-object compilation for HyperFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from studio.design.spec import DesignSpec
from studio.layout.solver import AdaptiveFrameZones, PixelBox, fit_axes_pixel_box, resolve_non_overlapping_boxes
from studio.renderers.hyperframes.svg_math import _sample_function_points, _sample_parametric_points
from studio.renderers.hyperframes.typography import _pretty_math_label

if TYPE_CHECKING:
    from studio.ai.v3_schema import SceneObject


def _role_color(design: DesignSpec, role: str) -> str:
    lk = design.look
    return {
        "primary": lk.accent_primary,
        "secondary": lk.accent_secondary,
        "tertiary": lk.accent_tertiary,
        "success": lk.success,
        "highlight": lk.accent_highlight,
        "danger": lk.danger,
    }.get(role, lk.accent_primary)


def _slot_box(
    slot: str,
    eff_comp: str,
    stage_x: float,
    stage_y: float,
    stage_w: float,
    stage_h: float,
    aspect_label: str,
) -> tuple[float, float, float, float]:
    """Compute responsive bounding box inside safe stage respecting aspect ratio."""
    pad = 16.0
    if slot == "left" or (slot == "auto" and eff_comp == "side_by_side_comparison"):
        if aspect_label == "9:16" or eff_comp == "stacked_comparison":
            return (stage_x, stage_y, stage_w, stage_h * 0.46)
        if slot == "left":
            return (stage_x, stage_y, stage_w * 0.48, stage_h)
    if slot == "right":
        if aspect_label == "9:16" or eff_comp == "stacked_comparison":
            return (stage_x, stage_y + stage_h * 0.52, stage_w, stage_h * 0.46)
        return (stage_x + stage_w * 0.52, stage_y, stage_w * 0.48, stage_h)
    if slot == "top":
        return (stage_x, stage_y, stage_w, stage_h * 0.46)
    if slot == "bottom":
        return (stage_x, stage_y + stage_h * 0.52, stage_w, stage_h * 0.46)
    if eff_comp == "main_with_equation_panel":
        return (stage_x + pad, stage_y + 18.0, stage_w - 2 * pad, stage_h - 32.0)
    return (stage_x + pad, stage_y + pad, stage_w - 2 * pad, stage_h - 2 * pad)


def compile_scene_object_element(
    obj: "SceneObject",
    *,
    design: DesignSpec,
    sc_zones: AdaptiveFrameZones,
    comp: str,
    has_side_ann: bool,
    t_start: float,
    hide_at: float,
    axes_specs: dict[str, tuple[list[float], list[float], tuple[float, float, float, float]]],
    default_geom_xr: list[float],
    default_geom_yr: list[float],
    sc_obstacles: list[PixelBox],
    ann_idx_local: int,
) -> tuple[dict[str, Any], int]:
    """Compile a single SceneObject into its deterministic pixel-space HTML/SVG spec."""
    sz = design.safe_zone
    tp = design.typography
    col = _role_color(design, obj.color_role)

    if obj.kind == "axes":
        xr = (obj.x_range or [-4.0, 4.0])[:2]
        yr = (obj.y_range or [-2.5, 2.5])[:2]
        ax_pbox = fit_axes_pixel_box(
            sc_zones,
            xr=xr,
            yr=yr,
            equal_aspect=bool(obj.equal_aspect),
            composition=comp,
            panel_slot=obj.panel_slot,
            has_side_annotations=has_side_ann,
        )
        box = (ax_pbox.x, ax_pbox.y, ax_pbox.w, ax_pbox.h)
        sc_obstacles.append(ax_pbox)
    elif obj.kind == "equation":
        eb = sc_zones.equation_box
        box = (eb.x + 12.0, eb.y + 4.0, max(200.0, eb.w - 24.0), max(48.0, eb.h - 8.0))
        sc_obstacles.append(PixelBox(box[0], box[1], box[2], box[3], role="equation"))
    else:
        box = _slot_box(
            obj.panel_slot,
            comp,
            sc_zones.stage_box.x,
            sc_zones.stage_box.y,
            sc_zones.stage_box.w,
            sc_zones.stage_box.h,
            sz.aspect_label,
        )

    def _to_px(xy: list[float], b: tuple[float, float, float, float], xr_l: list[float], yr_l: list[float]) -> tuple[float, float]:
        bx, by, bw, bh = b
        px = bx + ((float(xy[0]) - xr_l[0]) / max(xr_l[1] - xr_l[0], 1e-6)) * bw
        py = by + bh - ((float(xy[1]) - yr_l[0]) / max(yr_l[1] - yr_l[0], 1e-6)) * bh
        return round(px, 2), round(py, 2)

    el: dict[str, Any] = {
        "id": obj.id,
        "kind": obj.kind,
        "axes_id": obj.axes_id,
        "panel_slot": getattr(obj, "panel_slot", "auto") or "auto",
        "model_output": obj.model_output or "",
        "color": col,
        "show_at": t_start,
        "hide_at": hide_at,
        "box": [round(v, 1) for v in box],
        "label": obj.label,
        "x_label": getattr(obj, "x_label", "") or "",
        "y_label": getattr(obj, "y_label", "") or "",
        "legend_items": [
            {"label": item[0], "color": _role_color(design, item[1] if len(item) > 1 else "primary")}
            for item in (getattr(obj, "legend_items", []) or [])
            if item
        ],
    }

    if getattr(obj, "sweep_keyframes", None) and obj.model_output in ("winding_curve", "centroid_point", "centroid_vector", "readout", "trajectory", "end_point"):
        xr_sw, yr_sw, ax_sw = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
        sw_px: list[dict[str, Any]] = []
        for kf in obj.sweep_keyframes:
            pt_sel = kf.spec_point if (obj.model_output == "spectrum_peak" and kf.spec_point) else kf.point
            sw_px.append(
                {
                    "param": round(float(kf.param), 4),
                    "point_px": list(_to_px(pt_sel, ax_sw, xr_sw, yr_sw)) if len(pt_sel) >= 2 else [],
                    "vec_start_px": list(_to_px([0.0, 0.0], ax_sw, xr_sw, yr_sw)),
                    "vec_end_px": list(_to_px(kf.vector_end, ax_sw, xr_sw, yr_sw)) if len(kf.vector_end) >= 2 else [],
                    "polyline_px": [list(_to_px(q[:2], ax_sw, xr_sw, yr_sw)) for q in kf.curve_pts] if (kf.curve_pts and obj.model_output == "winding_curve") else [],
                    "label": kf.label,
                }
            )
        el["sweep_frames_px"] = sw_px

    if obj.kind == "axes":
        xr = (obj.x_range or [-4.0, 4.0])[:2]
        yr = (obj.y_range or [-2.5, 2.5])[:2]
        axes_specs[obj.id] = (xr, yr, box)
        ox, oy = _to_px([0.0, 0.0], box, xr, yr)
        el["axes"] = {"xr": xr, "yr": yr, "ox": ox, "oy": oy}
    elif obj.kind in ("function_plot", "parametric_curve", "waveform", "data_series"):
        xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (obj.x_range or [-4.0, 4.0], obj.y_range or [-2.5, 2.5], box))
        if obj.kind == "function_plot":
            raw_pts = _sample_function_points(obj.expr or "0", xr, yr)
        elif obj.kind == "parametric_curve":
            raw_pts = _sample_parametric_points(obj.expr or "cos(t);sin(t)", obj.t_range or [0.0, 6.283185])
        elif obj.points_list:
            raw_pts = [p[:2] for p in obj.points_list]
        else:
            raw_pts = _sample_function_points(obj.expr or "sin(x)", xr, yr)
        def _clamp_pt(q: list[float]) -> tuple[float, float]:
            cx_q = max(float(xr[0]), min(float(xr[1]), float(q[0])))
            cy_q = max(float(yr[0]), min(float(yr[1]), float(q[1])))
            return _to_px([cx_q, cy_q], ax_box, xr, yr)
        el["polyline"] = [_clamp_pt(pt) for pt in raw_pts]
    elif obj.kind == "bar_chart" and obj.values:
        vals = [float(v) for v in obj.values]
        mx = max((abs(v) for v in vals), default=1.0) or 1.0
        el["bars"] = [
            {"label": obj.bar_labels[i] if i < len(obj.bar_labels) else str(i), "val": v, "frac": round(abs(v) / mx, 4)}
            for i, v in enumerate(vals)
        ]
    elif obj.kind in ("polygon", "probability_region") and obj.vertices:
        xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
        el["polygon"] = [_to_px(v[:2], ax_box, xr, yr) for v in obj.vertices]
    elif obj.kind == "circle":
        xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
        c_px = _to_px(obj.coords or [0.0, 0.0], ax_box, xr, yr)
        r_math = float(obj.radius or 1.0)
        r_px = abs(_to_px([r_math, 0.0], ax_box, xr, yr)[0] - _to_px([0.0, 0.0], ax_box, xr, yr)[0])
        el["circle"] = {"cx": c_px[0], "cy": c_px[1], "r": round(max(6.0, r_px), 2)}
    elif obj.kind == "point":
        xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
        p_xy = obj.coords or [0.0, 0.0]
        if obj.param_name and obj.expr and ";" in obj.expr:
            pts_t = _sample_parametric_points(obj.expr.replace(obj.param_name, "t"), [0.0, 6.283185], n=48)
            el["point"] = _to_px(pts_t[0] if pts_t else p_xy, ax_box, xr, yr)
            el["orbit"] = [_to_px(q, ax_box, xr, yr) for q in pts_t]
        else:
            el["point"] = _to_px(p_xy[:2], ax_box, xr, yr)
    elif obj.kind in ("vector", "line"):
        xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
        p0 = _to_px(obj.coords or [0.0, 0.0], ax_box, xr, yr)
        p1 = _to_px(obj.end_coords or [1.0, 1.0], ax_box, xr, yr)
        el["segment"] = {"x1": p0[0], "y1": p0[1], "x2": p1[0], "y2": p1[1], "arrow": obj.kind == "vector"}
    elif obj.kind == "equation":
        el["eq_text"] = _pretty_math_label(obj.tex or obj.label or "x")
        el["tex"] = obj.tex or obj.label or "x"
        el["card"] = bool(obj.card)
    elif obj.kind == "annotation":
        xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
        lbl_s = obj.text or obj.label or ""
        est_w = min(sc_zones.stage_box.w * 0.42, max(160.0, len(lbl_s) * tp.body_px * 0.54 + 28.0))
        est_h = float(tp.body_px) * 1.45
        if obj.coords:
            px_c = _to_px(obj.coords, ax_box, xr, yr)
            raw_b = PixelBox(px_c[0] - 0.5 * est_w, px_c[1] - 0.5 * est_h, est_w, est_h, role="annotation")
        elif obj.axes_id and obj.axes_id in axes_specs:
            pref_x = min(sc_zones.stage_box.x2 - est_w - 8.0, ax_box[0] + ax_box[2] + 20.0)
            if sz.aspect_label == "9:16":
                pref_x = sc_zones.stage_box.cx - 0.5 * est_w
                pref_y = max(sc_zones.stage_box.y + 8.0, ax_box[1] - est_h - 14.0)
            else:
                pref_y = ax_box[1] + 18.0 + ann_idx_local * (est_h + 14.0)
            raw_b = PixelBox(pref_x, pref_y, est_w, est_h, role="annotation")
        else:
            raw_b = PixelBox(
                sc_zones.stage_box.cx - 0.5 * est_w,
                sc_zones.stage_box.y + 24.0 + ann_idx_local * (est_h + 14.0),
                est_w,
                est_h,
                role="annotation",
            )
        resolved_b = resolve_non_overlapping_boxes([raw_b], sc_obstacles, sc_zones, gap_px=12.0)[0]
        sc_obstacles.append(resolved_b)
        el["text_pos"] = [round(resolved_b.cx, 1), round(resolved_b.cy + tp.body_px * 0.34, 1)]
        el["box"] = [round(resolved_b.x, 1), round(resolved_b.y, 1), round(resolved_b.w, 1), round(resolved_b.h, 1)]
        el["text"] = _pretty_math_label(lbl_s) if any(tok in lbl_s for tok in ("\\", "_", "^")) else lbl_s.replace("lambda", "λ")
        ann_idx_local += 1
    elif obj.kind == "matrix":
        el["matrix"] = obj.matrix_rows or [["a", "b"], ["c", "d"]]
    elif obj.kind == "table":
        el["headers"] = obj.table_headers
        el["rows"] = obj.table_rows
    elif obj.kind == "graph_network":
        xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
        el["network"] = {
            "nodes": [_to_px(p[:2], ax_box, xr, yr) for p in obj.points_list],
            "labels": obj.node_labels,
            "edges": obj.edge_pairs,
        }
    elif obj.kind == "particle_system":
        xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
        el["particles"] = [_to_px(p[:2], ax_box, xr, yr) for p in obj.points_list]
    elif obj.kind == "number_line":
        el["number_line"] = {"xr": (obj.x_range or [-4.0, 4.0])[:2]}

    return el, ann_idx_local
