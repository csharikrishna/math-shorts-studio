"""HTML5/SVG document compiler for HyperFramesBackend."""

from __future__ import annotations

import html
import json
import re
from typing import TYPE_CHECKING, Any

from studio.ai.world_model import materialize_constructions
from studio.audio.tts import AudioTimeline, SyncedBeat
from studio.design.spec import DesignSpec
from studio.layout.solver import PixelBox, compute_adaptive_zones, resolve_composition_layout
from studio.renderers.hyperframes.captions import build_hyperframes_captions
from studio.renderers.hyperframes.components import compile_scene_object_element
from studio.renderers.hyperframes.runtime import build_runtime_js
from studio.renderers.hyperframes.typography import KATEX_DIR, build_font_css

if TYPE_CHECKING:
    from studio.ai.v3_schema import VideoPlanV3


def compile_plan_to_html(
    plan: "VideoPlanV3",
    design: DesignSpec,
    audio_timeline: AudioTimeline | None = None,
    *,
    caption_style: str | None = None,
    caption_ir: Any | None = None,
    meta_out_path: Any | None = None,
) -> tuple[str, float]:
    """Compile VideoPlanV3 into standalone deterministic offline HTML5/SVG + Motion runtime."""
    plan = materialize_constructions(plan)
    sz = design.safe_zone
    lk = design.look
    tp = design.typography
    mt = design.motion

    default_cap = "kinetic" if sz.aspect_label == "9:16" else "editorial"
    any_eq = any(any(o.kind == "equation" for o in sc.objects) for sc in plan.scenes)
    zones = compute_adaptive_zones(
        design.format_name,
        has_header_banner=False,
        has_equation_strip=any_eq,
        caption_style=caption_style or default_cap,
    )
    stage_x = float(zones.stage_box.x)
    stage_y = float(zones.stage_box.y)
    stage_w = float(zones.stage_box.w)
    stage_h = float(zones.stage_box.h)

    beat_windows: list[dict[str, Any]] = []
    synced_beats: list[SyncedBeat] = []
    if audio_timeline is not None and audio_timeline.beats:
        t_total = float(audio_timeline.total_duration)
        by_id = {b.id: b for b in audio_timeline.beats}
        cursor = 0.0
        for sc in plan.scenes:
            sb = by_id.get(sc.scene_id)
            if sb is not None:
                synced_beats.append(sb)
                start_s, end_s = float(sb.start_time), float(sb.end_time)
                words = [{"w": w[0], "s": round(float(w[1]), 4), "e": round(float(w[2]), 4)} for w in sb.words]
                cursor = end_s
            else:
                est = max(2.5, len(sc.narration.split()) / 2.08 + 0.3)
                start_s, end_s = cursor, cursor + est
                words = []
                cursor = end_s
            marks_map = {k: round(float(v), 4) for k, v in getattr(sb, "marks", {}).items()} if sb is not None else {}
            beat_windows.append(
                {
                    "scene_id": sc.scene_id,
                    "start": round(start_s, 4),
                    "end": round(end_s, 4),
                    "speaker": getattr(sc, "speaker", "narrator"),
                    "purpose": sc.purpose,
                    "caption": re.sub(r"<bookmark\b[^>]*>|\*", "", sc.narration).strip(),
                    "words": words,
                    "marks": marks_map,
                }
            )
    else:
        cursor = 0.0
        for sc in plan.scenes:
            clean_txt = re.sub(r"<bookmark\b[^>]*>|\*", "", sc.narration).strip()
            toks = clean_txt.split()
            dur = max(2.5, len(toks) / 2.08 + 0.35)
            w_dur = (dur - 0.3) / max(1, len(toks))
            words_raw = [
                [tok, round(cursor + i * w_dur, 4), round(cursor + (i + 1) * w_dur, 4), False]
                for i, tok in enumerate(toks)
            ]
            synced_beats.append(
                SyncedBeat(
                    id=sc.scene_id,
                    narration=clean_txt,
                    caption=sc.purpose,
                    audio_path="",
                    start_time=round(cursor, 4),
                    end_time=round(cursor + dur, 4),
                    speech_duration=round(max(0.2, dur - 0.3), 4),
                    total_duration=round(dur, 4),
                    words=words_raw,
                    marks={},
                    speaker=getattr(sc, "speaker", "narrator"),
                )
            )
            beat_windows.append(
                {
                    "scene_id": sc.scene_id,
                    "start": round(cursor, 4),
                    "end": round(cursor + dur, 4),
                    "speaker": getattr(sc, "speaker", "narrator"),
                    "purpose": sc.purpose,
                    "caption": clean_txt,
                    "words": [{"w": w[0], "s": w[1], "e": w[2]} for w in words_raw],
                }
            )
            cursor += dur
        t_total = max(0.5, cursor)

    cap_ir, cap_style, cues_payload = build_hyperframes_captions(synced_beats, design, caption_style, caption_ir=caption_ir)

    axes_specs: dict[str, tuple[list[float], list[float], tuple[float, float, float, float]]] = {}
    elements_payload: list[dict[str, Any]] = []
    meta_map_hf: dict[str, list[dict[str, Any]]] = {}
    zoom_events: list[dict[str, Any]] = []
    obj_by_id: dict[str, Any] = {}

    for s_idx, sc in enumerate(plan.scenes):
        for _o in sc.objects:
            obj_by_id[_o.id] = _o
        bw_info = beat_windows[s_idx]
        t_start, t_end = float(bw_info["start"]), float(bw_info["end"])
        comp = resolve_composition_layout(str(getattr(sc, "composition", "auto")), [o.kind for o in sc.objects], sz.aspect_label)
        sc_zones = compute_adaptive_zones(
            design.format_name,
            has_header_banner=False,
            has_equation_strip=any(o.kind == "equation" for o in sc.objects),
            caption_style=cap_style,
        )
        sc_obstacles: list[PixelBox] = [
            PixelBox(el["box"][0], el["box"][1], el["box"][2], el["box"][3], role=el["kind"])
            for el in elements_payload
            if el["hide_at"] > t_start
        ]
        has_side_ann = any(o.kind == "annotation" for o in sc.objects)

        unaxed_pts = [
            pt
            for o in sc.objects
            if o.kind in ("polygon", "probability_region") and not o.axes_id and o.vertices
            for pt in o.vertices
        ]
        if unaxed_pts:
            ux = [float(p[0]) for p in unaxed_pts]
            uy = [float(p[1]) for p in unaxed_pts]
            pad_x = max(0.4, 0.15 * (max(ux) - min(ux) or 1.0))
            pad_y = max(0.4, 0.15 * (max(uy) - min(uy) or 1.0))
            default_geom_xr = [min(ux) - pad_x, max(ux) + pad_x]
            default_geom_yr = [min(uy) - pad_y, max(uy) + pad_y]
        else:
            default_geom_xr = [-3.5, 3.5]
            default_geom_yr = [-2.5, 2.5]

        active_eq = [o for o in sc.objects if o.kind == "equation"]
        referenced_ids = (
            {e.target_id for e in sc.events}
            | {e.source_id for e in sc.events if e.source_id}
            | {o.axes_id for o in sc.objects if o.axes_id}
        )
        if any(o.kind in ("axes", "table", "matrix", "bar_chart", "graph_network") for o in sc.objects):
            new_slots = {getattr(o, "panel_slot", "auto") for o in sc.objects if o.kind in ("axes", "table", "matrix", "bar_chart", "graph_network")}
            is_split_keep = bool(new_slots <= {"right", "bottom"}) and comp in ("side_by_side_comparison", "stacked_comparison", "time_frequency_split", "multi_view_dashboard")
            retired_axes: set[str] = set()
            for el in elements_payload:
                if el["hide_at"] > t_start and el["id"] not in referenced_ids and el.get("axes_id") not in referenced_ids:
                    if is_split_keep and el.get("panel_slot") in ("left", "top"):
                        continue
                    if is_split_keep and el.get("axes_id") and any(p.get("id") == el.get("axes_id") and p.get("panel_slot") in ("left", "top") for p in elements_payload):
                        continue
                    if el["kind"] in ("axes", "table", "matrix", "bar_chart", "graph_network", "annotation", "function_plot", "parametric_curve", "data_series"):
                        el["hide_at"] = round(t_start, 4)
                        if el["kind"] == "axes":
                            retired_axes.add(el["id"])
            if retired_axes:
                for el in elements_payload:
                    if el["hide_at"] > t_start and el.get("axes_id") in retired_axes and el["id"] not in referenced_ids:
                        el["hide_at"] = round(t_start, 4)

        event_show_at: dict[str, float] = {}
        marks_for_scene = bw_info.get("marks", {}) or {}
        for ev in sc.events:
            if ev.op in ("create", "draw", "reveal", "trace", "duplicate", "sequence", "parallel") and ev.anchor_mark:
                if ev.anchor_mark in marks_for_scene:
                    event_show_at[ev.target_id] = float(marks_for_scene[ev.anchor_mark])
            if ev.op == "remove":
                for el in elements_payload:
                    if el["id"] == ev.target_id and el["hide_at"] > t_start:
                        el["hide_at"] = round(t_start + min(0.4, mt.exit_s), 4)

        ann_idx_local = 0
        for obj in sc.objects:
            hide_at = round(t_total + 1.0, 4) if obj.persist else round(t_end, 4)
            obj_show = event_show_at.get(obj.id, t_start)
            el, ann_idx_local = compile_scene_object_element(
                obj,
                design=design,
                sc_zones=sc_zones,
                comp=comp,
                has_side_ann=has_side_ann,
                t_start=obj_show,
                hide_at=hide_at,
                axes_specs=axes_specs,
                default_geom_xr=default_geom_xr,
                default_geom_yr=default_geom_yr,
                sc_obstacles=sc_obstacles,
                ann_idx_local=ann_idx_local,
            )
            elements_payload.append(el)

        # Compile state-changing transform events (move, rotate, scale, zoom) + sweep windows into keyframed pixel transforms
        for ev in sc.events:
            if ev.op in ("animate_param", "iterate_states", "interpolate_values"):
                sw_t0 = float(marks_for_scene[ev.anchor_mark]) if (ev.anchor_mark and ev.anchor_mark in marks_for_scene) else (t_start + 0.15)
                sw_dur = max(0.8, float(ev.duration_s or max(1.4, (t_end - sw_t0) * 0.85)))
                sw_t1 = round(min(t_end - 0.05, sw_t0 + sw_dur), 4)
                for el in elements_payload:
                    if el["hide_at"] <= t_start:
                        continue
                    if el.get("sweep_frames_px") and (el["id"] == ev.target_id or getattr(obj_by_id.get(el["id"]), "model_id", None) == getattr(obj_by_id.get(ev.target_id), "model_id", "__none__")):
                        el["sweep_t0"] = round(sw_t0, 4)
                        el["sweep_t1"] = max(round(sw_t0 + 0.35, 4), sw_t1)
                continue
            if ev.op not in ("move", "rotate", "scale", "zoom"):
                continue
            ev_t0 = float(marks_for_scene[ev.anchor_mark]) if (ev.anchor_mark and ev.anchor_mark in marks_for_scene) else (t_start + 0.35)
            ev_t1 = round(ev_t0 + max(0.35, float(ev.duration_s or 0.9)), 4)
            xr_e, yr_e, ax_e = axes_specs.get(getattr(ev, "axes_id", "") or "", (default_geom_xr, default_geom_yr, [stage_x, stage_y, stage_w, stage_h]))
            dx_u = float(ev.shift[0]) if ev.shift and len(ev.shift) >= 1 else 0.0
            dy_u = float(ev.shift[1]) if ev.shift and len(ev.shift) >= 2 else 0.0
            sx_px = ax_e[2] / max(0.001, (xr_e[1] - xr_e[0]))
            sy_px = ax_e[3] / max(0.001, (yr_e[1] - yr_e[0]))
            if ev.op == "zoom":
                tgt_el = next((e for e in elements_payload if e["id"] == ev.target_id and e["hide_at"] > t_start), None)
                t_box = tgt_el["box"] if tgt_el else [stage_x, stage_y, stage_w, stage_h]
                zoom_events.append(
                    {
                        "t0": round(ev_t0, 4),
                        "t1": ev_t1,
                        "hold_end": round(t_end, 4),
                        "scale": min(1.85, max(1.06, float(ev.scale_factor or 1.28))),
                        "cx": round(t_box[0] + 0.5 * t_box[2], 1),
                        "cy": round(t_box[1] + 0.5 * t_box[3], 1),
                    }
                )
                continue
            tx_record = {
                "t0": round(ev_t0, 4),
                "t1": ev_t1,
                "op": ev.op,
                "dx_px": round(dx_u * sx_px, 2),
                "dy_px": round(-dy_u * sy_px, 2),
                "angle_deg": round(float(ev.angle_deg or 0.0), 3),
                "scale": round(float(ev.scale_factor or 1.0), 4),
            }
            for el in elements_payload:
                if el["hide_at"] <= t_start:
                    continue
                if el["id"] == ev.target_id:
                    el.setdefault("transforms", []).append(tx_record)

        if active_eq:
            latest_eq_id = active_eq[-1].id
            for el in elements_payload:
                if el["kind"] == "equation" and el["id"] != latest_eq_id and el["hide_at"] > t_start:
                    el["hide_at"] = round(t_start, 4)

        meta_map_hf[sc.scene_id] = [
            {"id": el["id"], "kind": el["kind"], "box": el["box"]}
            for el in elements_payload
            if el["hide_at"] > t_start
        ]

    if meta_out_path is not None:
        from pathlib import Path as _Path

        mp = _Path(str(meta_out_path))
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(meta_map_hf, indent=2), encoding="utf-8")

    font_rules_css = build_font_css("/fonts/")
    cap_preset_dict = cap_ir.preset.model_dump() if cap_ir.preset else {}
    payload_json = json.dumps(
        {
            "title": plan.title,
            "width": sz.width,
            "height": sz.height,
            "stage": [stage_x, stage_y, stage_w, stage_h],
            "top_band": zones.top_band_px,
            "eq_box": [zones.equation_box.x, zones.equation_box.y, zones.equation_box.w, zones.equation_box.h],
            "bottom_band": zones.caption_band_px,
            "side_margin": zones.side_margin_px,
            "duration": round(t_total, 4),
            "look": lk.model_dump(),
            "typo": tp.model_dump(),
            "motion": mt.model_dump(),
            "caption_style": cap_style,
            "caption_preset": cap_preset_dict,
            "beats": beat_windows,
            "cues": cues_payload,
            "elements": elements_payload,
            "zoom_events": zoom_events,
        },
        ensure_ascii=False,
    )

    cap_bg = lk.surface if (cap_ir.preset and cap_ir.preset.use_box) else "transparent"
    cap_pad = "10px 22px" if (cap_ir.preset and cap_ir.preset.use_box) else "2px 8px"
    cap_border = f"1px solid {lk.surface_border}" if (cap_ir.preset and cap_ir.preset.use_box) else "none"
    cap_shadow = "none" if lk.is_light else "0 2px 10px rgba(0, 0, 0, 0.48)"
    has_katex = (KATEX_DIR / "katex.min.js").exists() and (KATEX_DIR / "katex.min.css").exists()
    katex_head = (
        '<link rel="stylesheet" href="/vendor/katex/katex.min.css" />\n'
        '<script src="/vendor/katex/katex.min.js"></script>\n'
        if has_katex
        else ""
    )

    safe_title = html.escape(plan.title)
    runtime_js = build_runtime_js(payload_json)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{safe_title}</title>
{katex_head}<style>
{font_rules_css}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{
  width: {sz.width}px;
  height: {sz.height}px;
  overflow: hidden;
  background: {lk.ground};
  color: {lk.ink_primary};
  font-family: {tp.body_font};
}}
#canvas-root {{
  position: relative;
  width: {sz.width}px;
  height: {sz.height}px;
  background: {lk.ground};
}}
#svg-stage {{
  position: absolute;
  left: 0;
  top: 0;
  width: {sz.width}px;
  height: {sz.height}px;
}}
#html-overlay {{
  position: absolute;
  left: 0;
  top: 0;
  width: {sz.width}px;
  height: {sz.height}px;
  pointer-events: none;
}}
.hf-eq-card {{
  position: absolute;
  left: {sz.side_margin_px + 16}px;
  right: {sz.side_margin_px + 16}px;
  top: {int(zones.equation_box.y)}px;
  min-height: {int(zones.equation_box.h)}px;
  display: flex;
  align-items: center;
  justify-content: center;
}}
.hf-eq-bare {{
  font-family: {tp.math_font};
  font-size: {tp.equation_px}px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: {lk.ink_primary};
}}
.hf-eq-inner {{
  background: {lk.surface};
  border: 1.4px solid {lk.surface_border};
  border-radius: 12px;
  padding: 10px 26px;
  font-family: {tp.math_font};
  font-size: {tp.equation_px}px;
  font-weight: 600;
  color: {lk.ink_primary};
}}
.hf-table {{
  position: absolute;
  border-collapse: collapse;
  background: {lk.surface};
  border: 1px solid {lk.surface_border};
  border-radius: 10px;
  overflow: hidden;
  font-family: {tp.body_font};
  font-size: {tp.body_px}px;
}}
.hf-table th, .hf-table td {{
  padding: 8px 14px;
  border: 1px solid {lk.surface_border};
  text-align: center;
}}
.hf-table th {{ color: {lk.accent_primary}; font-weight: 700; }}
.hf-matrix {{
  position: absolute;
  display: inline-grid;
  gap: 12px 26px;
  padding: 16px 26px;
  border-left: 3px solid {lk.ink_primary};
  border-right: 3px solid {lk.ink_primary};
  background: transparent;
  font-family: {tp.serif_font};
  font-size: {tp.math_label_px}px;
  font-weight: 700;
}}
.hf-matrix::before, .hf-matrix::after {{
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  width: 12px;
  border-top: 3px solid {lk.ink_primary};
  border-bottom: 3px solid {lk.ink_primary};
  pointer-events: none;
}}
.hf-matrix::before {{ left: 0; }}
.hf-matrix::after {{ right: 0; }}
#caption-band {{
  position: absolute;
  left: {sz.side_margin_px}px;
  right: {sz.side_margin_px}px;
  bottom: 18px;
  min-height: {max(64, int(zones.caption_band_px) - 20)}px;
  display: {"none" if cap_style == "off" else "flex"};
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 6px;
}}
#speaker-badge {{
  font-family: {tp.display_font};
  font-size: {max(18, tp.body_px - 6)}px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  padding: 3px 12px;
  border-radius: 999px;
  background: {lk.surface};
  border: 1px solid {lk.accent_secondary};
  color: {lk.accent_secondary};
  display: none;
}}
#caption-text {{
  font-family: {tp.body_font};
  font-size: {tp.caption_px}px;
  line-height: 1.24;
  font-weight: {600 if cap_style in ("editorial", "minimal") else 700};
  color: {lk.ink_primary};
  background: {cap_bg};
  border: {cap_border};
  border-radius: 12px;
  padding: {cap_pad};
  text-shadow: {cap_shadow};
  max-width: {int(stage_w * 0.92)}px;
}}
.cap-line {{ display: block; }}
.word-active {{ color: {lk.accent_primary if cap_style in ("editorial", "dialogue") else (lk.ink_primary if cap_style in ("minimal", "accessible") else lk.ink_primary)}; font-weight: 700; }}
.word-active-emph {{ color: {lk.accent_primary}; font-weight: 800; border-bottom: 2.5px solid {lk.accent_primary}; padding-bottom: 1px; }}
.word-past {{ color: {lk.ink_primary}; }}
.word-past-emph {{ color: {lk.accent_primary}; font-weight: 700; }}
.word-future {{ color: {lk.ink_primary if cap_style in ("editorial", "minimal", "accessible") else lk.ink_muted}; opacity: {0.92 if cap_style in ("editorial", "minimal", "accessible") else 0.85}; }}
</style>
</head>
<body>
<div id="canvas-root">
  <svg id="svg-stage" viewBox="0 0 {sz.width} {sz.height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="hf-arrow" markerWidth="10" markerHeight="10" refX="7" refY="3.5" orient="auto">
        <polygon points="0 0, 8 3.5, 0 7" fill="{lk.accent_primary}" />
      </marker>
    </defs>
    <g id="svg-layers"></g>
  </svg>
  <div id="html-overlay"></div>
  <div id="caption-band">
    <div id="speaker-badge"></div>
    <div id="caption-text"></div>
  </div>
</div>
<script>
{runtime_js}
</script>
</body>
</html>
"""
    return html_doc, t_total
