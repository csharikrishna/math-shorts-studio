"""Visual intelligence stage: beat-level visual reasoning, reusable composition mapping, and data-driven layout fitting."""

from __future__ import annotations

import math
from typing import Any, Sequence
from pydantic import BaseModel, Field

from studio.ai.expr import compile_expr
from studio.ai.v3_schema import SceneObject, SceneV3, VideoPlanV3


class VisualBeatDecision(BaseModel):
    """Structured pedagogical & cinematic visual plan derived for one narration beat."""

    scene_id: str
    learner_understanding: str
    mathematical_relationship: str
    composition: str
    visual_components: list[str] = Field(default_factory=list)
    changing_parameters: list[str] = Field(default_factory=list)
    linked_views: list[str] = Field(default_factory=list)
    viewer_focus: str = ""
    beat_connection: str = ""


def _nice_range(vmin: float, vmax: float, *, pad_ratio: float = 0.12, anchor_zero: bool = True) -> list[float]:
    """Compute readable data-enclosing axis bounds [lo, hi, step] without clipping extremes."""
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        return [-2.0, 2.0, 1.0]
    if abs(vmax - vmin) < 1e-6:
        c = vmin
        return [round(c - 1.5, 4), round(c + 1.5, 4), 0.75]
    span = vmax - vmin
    lo = vmin - span * pad_ratio
    hi = vmax + span * pad_ratio
    if anchor_zero and vmin >= 0.0 and vmin <= span * 0.35:
        lo = 0.0
    elif anchor_zero and vmax <= 0.0 and abs(vmax) <= span * 0.35:
        hi = 0.0
    eff_span = max(1e-5, hi - lo)
    raw_step = eff_span / 4.5
    exp_p = math.pow(10.0, math.floor(math.log10(raw_step)))
    frac = raw_step / exp_p
    nice_mult = 1.0 if frac <= 1.2 else (2.0 if frac <= 2.4 else (2.5 if frac <= 3.2 else (5.0 if frac <= 6.0 else 10.0)))
    step = nice_mult * exp_p
    lo_n = math.floor(lo / step) * step if lo < 0 else (0.0 if lo == 0.0 else math.floor(lo / step) * step)
    hi_n = math.ceil(hi / step) * step
    return [round(lo_n, 5), round(hi_n, 5), round(step, 5)]


def infer_beat_visual_decision(beat: Any, idx: int = 0, total: int = 1) -> VisualBeatDecision:
    """Infer concrete learner takeaway, mathematical relationship, linked composition & components from an approved BeatIR."""
    sid = str(getattr(beat, "scene_id", f"scene_{idx + 1}"))
    purpose = str(getattr(beat, "educational_purpose", "") or "")
    vint = str(getattr(beat, "visual_intent", "") or "")
    spoken = str(getattr(beat, "spoken_text", "") or "")
    trans = str(getattr(beat, "transition_from_prev", "") or "")
    dc = getattr(beat, "demo_contract", None)
    blob = f"{purpose} {vint} {spoken} {(getattr(dc, 'relationship_explained', '') if dc else '')}".lower()

    if any(k in blob for k in ("spring", "damper", "pendulum", "mass", "bob", "physical")):
        if any(k in blob for k in ("phase", "velocity", "portrait", "orbit", "spiral")):
            comp = "physical_system_with_phase_space"
            views = ["mechanical_body", "phase_orbit", "vector_field", "readout"]
        else:
            comp = "physical_system_with_displacement"
            views = ["mechanical_body", "time_series_x", "restoring_force", "energy_bars"]
    elif any(k in blob for k in ("predator", "prey", "population", "two", "both curves", "versus", "vs")) and any(k in blob for k in ("time", "cycle", "rise", "fall", "lag")):
        comp = "dual_time_series" if "phase" not in blob else "multi_view_dashboard"
        views = ["time_series_x", "time_series_y"] + (["phase_orbit"] if "phase" in blob else [])
    elif any(k in blob for k in ("spectrum", "frequency response", "resonance", "sweep", "damping ratio", "winding")):
        comp = "parameter_sweep_with_response"
        views = ["winding_curve_or_system", "spectrum_or_response_curve", "moving_marker", "readout"]
    elif any(k in blob for k in ("conserved", "invariant", "energy", "hamiltonian", "dissipat")):
        comp = "state_with_conserved_chart"
        views = ["phase_orbit", "energy_bars_or_invariant_curve", "readout"]
    elif any(k in blob for k in ("vector field", "phase portrait", "phase space", "orbit", "equilibrium", "saddle", "fixed point")):
        comp = "vector_field_with_trajectory"
        views = ["vector_field", "phase_orbit", "point", "annotation"]
    elif any(k in blob for k in ("filter", "before", "after", "compare", "contrast", "limit", "carrying capacity")):
        comp = "before_after_simulation"
        views = ["baseline_curve", "modified_curve", "annotation"]
    elif getattr(beat, "latex_refs", None):
        comp = "main_with_equation_panel"
        views = ["axes", "data_series", "equation"]
    else:
        comp = "centered_focus"
        views = ["axes", "data_series", "annotation"]

    req_chg = str(getattr(dc, "required_state_change", "") if dc else "") or (
        "animate parameter sweep & synchronized trajectory marker"
        if any(k in blob for k in ("oscillat", "cycle", "sweep", "spiral", "decay", "resonat", "orbit", "evolve"))
        else "stepwise construction synced to narration bookmark"
    )
    return VisualBeatDecision(
        scene_id=sid,
        learner_understanding=(getattr(dc, "learner_takeaway", "") if dc else "") or purpose or f"Visual understanding of {sid}",
        mathematical_relationship=(getattr(dc, "relationship_explained", "") if dc else "") or vint or purpose,
        composition=comp,
        visual_components=list(getattr(dc, "visual_components", []) if dc and getattr(dc, "visual_components", None) else views),
        changing_parameters=list(getattr(dc, "changing_parameters", []) if dc and getattr(dc, "changing_parameters", None) else ["t"]),
        linked_views=views,
        viewer_focus=req_chg,
        beat_connection=trans or ("Establish concrete visual context" if idx == 0 else "Carry persistent apparatus forward"),
    )


def build_visual_intelligence_brief(beats: Sequence[Any]) -> tuple[list[VisualBeatDecision], str]:
    """Produce structured beat visual decisions + concise prompt table guiding scene compilation."""
    decs = [infer_beat_visual_decision(b, idx=i, total=len(beats)) for i, b in enumerate(beats)]
    lines = [
        f"- {d.scene_id}: comp={d.composition} | views={d.linked_views} | understand={d.learner_understanding[:90]} | focus={d.viewer_focus[:80]}"
        for d in decs
    ]
    return decs, "\n".join(lines)


def _sample_obj_points(obj: SceneObject, xr_hint: Sequence[float]) -> list[list[float]]:
    pts: list[list[float]] = []
    if obj.points_list:
        pts.extend(list(p[:2]) for p in obj.points_list if len(p) >= 2)
    if obj.vertices:
        pts.extend(list(p[:2]) for p in obj.vertices if len(p) >= 2)
    if getattr(obj, "sweep_keyframes", None):
        for kf in obj.sweep_keyframes:
            if kf.point and len(kf.point) >= 2:
                pts.append(list(kf.point[:2]))
            if kf.spec_point and len(kf.spec_point) >= 2 and obj.model_output in ("spectrum_peak", "time_point"):
                pts.append(list(kf.spec_point[:2]))
            if kf.curve_pts and obj.model_output in ("winding_curve", "phase_orbit", "trajectory", "loss_curve"):
                pts.extend(list(q[:2]) for q in kf.curve_pts[:: max(1, len(kf.curve_pts) // 24)] if len(q) >= 2)
    if obj.coords and len(obj.coords) >= 2 and obj.kind in ("point", "vector", "line", "mechanical_body"):
        pts.append(list(obj.coords[:2]))
    if obj.end_coords and len(obj.end_coords) >= 2 and obj.kind in ("vector", "line"):
        pts.append(list(obj.end_coords[:2]))
    if obj.kind == "function_plot" and obj.expr and not pts:
        try:
            fn = compile_expr(obj.expr, ("x",))
            x0, x1 = float(xr_hint[0]), float(xr_hint[1])
            for i in range(41):
                xv = x0 + (x1 - x0) * i / 40.0
                yv = float(fn(x=xv))
                if math.isfinite(yv) and abs(yv) < 1e4:
                    pts.append([xv, yv])
        except Exception:
            pass
    elif obj.kind == "parametric_curve" and obj.expr and ";" in obj.expr and not pts:
        try:
            p0, p1 = obj.expr.split(";", 1)
            fx, fy = compile_expr(p0, ("t",)), compile_expr(p1, ("t",))
            tr = obj.t_range or [0.0, 2.0 * math.pi]
            for i in range(41):
                tv = float(tr[0]) + (float(tr[1]) - float(tr[0])) * i / 40.0
                xv, yv = float(fx(t=tv)), float(fy(t=tv))
                if math.isfinite(xv) and math.isfinite(yv):
                    pts.append([xv, yv])
        except Exception:
            pass
    return [[float(p[0]), float(p[1])] for p in pts if math.isfinite(float(p[0])) and math.isfinite(float(p[1]))]


def apply_visual_intelligence_and_fit(plan: VideoPlanV3, *, aspect_label: str = "16:9") -> VideoPlanV3:
    """Data-driven post-materialization layout refinement: zero clipped curves, disjoint panels, labels & legends."""
    out = plan.model_copy(deep=True)
    model_map = {m.model_id: m for m in (getattr(out, "math_models", []) or [])}

    # Collect all axes across scenes and all dependent data points that remain attached to each axes_id
    axes_objs: dict[str, SceneObject] = {}
    axes_children: dict[str, list[SceneObject]] = {}
    axes_is_zoom: set[str] = set()
    for sc in out.scenes:
        has_zoom_ev = any(e.op == "zoom" for e in sc.events) or sc.composition == "spotlight_zoom"
        for o in sc.objects:
            if o.kind == "axes":
                axes_objs[o.id] = o
            if o.axes_id:
                axes_children.setdefault(o.axes_id, []).append(o)
                if has_zoom_ev:
                    axes_is_zoom.add(o.axes_id)

    # 1. Fit each non-zoomed axes to strictly contain its bound mathematical data (prevents clipped peaks!)
    for ax_id, ax in axes_objs.items():
        if ax_id in axes_is_zoom:
            continue
        children = axes_children.get(ax_id, [])
        xr_cur = list(ax.x_range or [-4.0, 4.0])[:2]
        yr_cur = list(ax.y_range or [-2.5, 2.5])[:2]
        all_pts: list[list[float]] = []
        for ch in children:
            if ch.kind in ("vector_field", "annotation", "equation"):
                continue
            all_pts.extend(_sample_obj_points(ch, xr_cur))
        if len(all_pts) >= 2:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            d_xmin, d_xmax = min(xs), max(xs)
            d_ymin, d_ymax = min(ys), max(ys)
            x_clips = d_xmin < xr_cur[0] - 1e-3 or d_xmax > xr_cur[1] + 1e-3
            y_clips = d_ymin < yr_cur[0] - 1e-3 or d_ymax > yr_cur[1] + 1e-3
            x_tiny = (d_xmax - d_xmin) > 1e-4 and (d_xmax - d_xmin) < 0.22 * max(1e-6, xr_cur[1] - xr_cur[0])
            y_mid = 0.5 * (d_ymin + d_ymax)
            is_flat_invariant = (d_ymax - d_ymin) <= 0.015 * max(1.0, abs(y_mid))
            y_tiny = (not is_flat_invariant) and (d_ymax - d_ymin) > 1e-4 and (d_ymax - d_ymin) < 0.22 * max(1e-6, yr_cur[1] - yr_cur[0])
            if x_clips or x_tiny:
                ax.x_range = _nice_range(d_xmin, d_xmax, pad_ratio=0.08, anchor_zero=True)
            if y_clips:
                if is_flat_invariant:
                    half_span = max(0.8, abs(y_mid) * 0.45)
                    ax.y_range = _nice_range(y_mid - half_span, y_mid + half_span, pad_ratio=0.10, anchor_zero=False)
                else:
                    ax.y_range = _nice_range(d_ymin, d_ymax, pad_ratio=0.14, anchor_zero=(d_ymin >= 0.0 and d_ymin < 0.35 * max(1e-6, d_ymax - d_ymin)))
            elif y_tiny:
                ax.y_range = _nice_range(d_ymin, d_ymax, pad_ratio=0.14, anchor_zero=(d_ymin >= 0.0 and d_ymin < 0.35 * max(1e-6, d_ymax - d_ymin)))
            # Synchronize child vector_field domain to updated axes bounds
            for ch in children:
                if ch.kind == "vector_field":
                    ch.x_range = list(ax.x_range)
                    ch.y_range = list(ax.y_range)

        # Populate helpful default axis labels & semantic curve legends when omitted
        if not ax.x_label or not ax.y_label:
            m_ref = next((model_map[ch.model_id] for ch in children if ch.model_id and ch.model_id in model_map), None)
            outs = {ch.model_output or "" for ch in children}
            if m_ref is not None:
                units = dict(getattr(m_ref, "units", {}) or {})
                if "winding_curve" in outs:
                    ax.x_label = ax.x_label or "Re z"
                    ax.y_label = ax.y_label or "Im z"
                elif "response_curve" in outs:
                    ax.x_label = ax.x_label or "ω (drive)"
                    ax.y_label = ax.y_label or "|H(ω)|"
                elif outs & {"spectrum", "continuous_fourier_spectrum", "normalized_spectrum"}:
                    u_f = "rad/s" if m_ref.angular_units == "rad" else "Hz"
                    ax.x_label = ax.x_label or f"Freq ({u_f})"
                    ax.y_label = ax.y_label or "|C(f)|"
                elif outs & {"phase_orbit"} and len(m_ref.state_vars) >= 2:
                    v0, v1 = m_ref.state_vars[0], m_ref.state_vars[1]
                    u0 = f" ({units[v0]})" if units.get(v0) else ""
                    u1 = f" ({units[v1]})" if units.get(v1) else ""
                    ax.x_label = ax.x_label or f"{v0}{u0}"
                    ax.y_label = ax.y_label or f"{v1}{u1}"
                elif outs & {"time_series_x", "time_series_y", "time_signal", "filtered_signal"}:
                    ut = f" ({units['t']})" if units.get("t") else " (t)"
                    ax.x_label = ax.x_label or f"Time{ut}"
                    if not ax.y_label:
                        v_nm = m_ref.state_vars[0] if m_ref.state_vars else "Value"
                        ax.y_label = v_nm if len(outs & {"time_series_x", "time_series_y"}) <= 1 else "State"

        if not ax.legend_items:
            labeled_children = [ch for ch in children if ch.kind in ("data_series", "function_plot", "parametric_curve", "waveform") and ch.label]
            if len(labeled_children) >= 2:
                seen_l: set[str] = set()
                items: list[list[str]] = []
                for ch in labeled_children:
                    if ch.label not in seen_l:
                        seen_l.add(ch.label)
                        items.append([ch.label[:20], ch.color_role])
                ax.legend_items = items[:3]

    # 2. Enforce disjoint multi-panel slot assignments across both new and carried live panels
    live_panels_prev: dict[str, SceneObject] = {}
    removed_prev: set[str] = set()
    for sc in out.scenes:
        active_map: dict[str, SceneObject] = {k: v for k, v in live_panels_prev.items() if k not in removed_prev}
        for o in sc.objects:
            active_map[o.id] = o
        major_panels = [
            o for o in active_map.values()
            if o.kind in ("axes", "table", "matrix", "bar_chart", "graph_network")
            or (o.kind == "mechanical_body" and not o.axes_id)
        ]
        if len(major_panels) >= 2:
            slots = ("top", "bottom", "right", "left") if aspect_label == "9:16" else ("left", "right", "bottom", "top")
            seen_slots: set[str] = set()
            for o in major_panels:
                cur_s = getattr(o, "panel_slot", "auto") or "auto"
                if cur_s in ("left", "right", "top", "bottom") and cur_s not in seen_slots:
                    seen_slots.add(cur_s)
                else:
                    nxt = next((s for s in slots if s not in seen_slots), None)
                    if nxt is not None:
                        o.panel_slot = nxt  # type: ignore[assignment]
                        seen_slots.add(nxt)
            if sc.composition in ("auto", "centered_focus"):
                sc.composition = "stacked_comparison" if aspect_label == "9:16" else "side_by_side_comparison"  # type: ignore[assignment]
        for o in sc.objects:
            live_panels_prev[o.id] = o
            removed_prev.discard(o.id)
        for ev in sc.events:
            if ev.op == "remove" and ev.target_id:
                removed_prev.add(ev.target_id)

    return out
