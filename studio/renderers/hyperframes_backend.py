"""Local OSI-clean HTML5/SVG + headless Chromium CDP rendering backend (HyperFrames adapter)."""

from __future__ import annotations

import base64
import functools
import html
import http.server
import json
import math
import os
import re
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from studio.ai.env import PROJECT_ROOT
from studio.ai.expr import compile_expr
from studio.ai.world_model import materialize_constructions
from studio.audio.bgm import generate_ambient_bgm
from studio.audio.master import master_video
from studio.audio.sfx import SfxLibrary
from studio.audio.tts import AudioTimeline, BeatSpec, build_voice_timeline, find_ffmpeg
from studio.config import FORMATS
from studio.design.spec import DesignSpec, default_design_spec
from studio.registry.compositions import adapt_composition_for_aspect
from studio.renderers.base import BackendRenderResult, RenderBackend

if TYPE_CHECKING:
    from studio.ai.v3_schema import SceneObject, VideoPlanV3


def find_chromium_binary() -> str | None:
    """Locate local headless Chromium / Chrome / Edge binary (no remote downloads)."""
    env_p = os.environ.get("STUDIO_CHROME_BIN") or os.environ.get("CHROME_PATH")
    if env_p and Path(env_p).exists():
        return env_p
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("chrome"),
        shutil.which("msedge"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def _current_process_rss_mb() -> float:
    """Return current process peak/working set RSS in MB (Windows ctypes or POSIX fallback)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            hproc = k32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(hproc, ctypes.byref(counters), counters.cb):
                return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)
        else:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except Exception:
        pass
    return 0.0


def validate_safe_relpath(root_dir: Path, raw_url_path: str) -> Path | None:
    """Path-traversal-safe resolver for local loopback static server (adapted from HyperFrames fileServer.ts)."""
    clean = raw_url_path.split("?", 1)[0].split("#", 1)[0]
    try:
        decoded = urllib.parse.unquote(clean)
    except Exception:
        return None
    if "\x00" in decoded or "\\" in decoded:
        return None
    parts = [p for p in decoded.split("/") if p]
    if not parts:
        parts = ["index.html"]
    for part in parts:
        if part in (".", "..") or part.startswith("."):
            return None
    root_resolved = root_dir.resolve()
    candidate = (root_resolved.joinpath(*parts)).resolve()
    if not candidate.is_relative_to(root_resolved):
        return None
    return candidate


class _LoopbackHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, root_dir: Path, **kwargs: Any) -> None:
        self._root_dir = root_dir
        super().__init__(*args, directory=str(root_dir), **kwargs)

    def do_GET(self) -> None:
        safe = validate_safe_relpath(self._root_dir, self.path)
        if safe is None:
            self.send_error(403, "Forbidden path traversal")
            return
        if not safe.exists() or not safe.is_file():
            self.send_error(404, "Not found")
            return
        ctype = "text/html; charset=utf-8" if safe.suffix == ".html" else "application/json; charset=utf-8"
        if safe.suffix == ".svg":
            ctype = "image/svg+xml"
        elif safe.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        data = safe.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _start_loopback_server(root_dir: Path) -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(_LoopbackHandler, root_dir=root_dir)
    socketserver.TCPServer.allow_reuse_address = True
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = int(server.server_address[1])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _pretty_math_label(tex: str) -> str:
    """Convert basic LaTeX to crisp typographic Unicode strings offline."""
    out = re.sub(r"\\(?:mathbf|mathrm|text|vec|operatorname)\{([^{}]*)\}", r"\1", tex)
    out = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", out)
    repls = [
        (r"\det", "det"),
        (r"\Re", "Re"),
        (r"\Im", "Im"),
        (r"\pi", "π"),
        (r"\theta", "θ"),
        (r"\lambda", "λ"),
        (r"\mu", "μ"),
        (r"\sigma", "σ"),
        (r"\sum", "Σ"),
        (r"\int", "∫"),
        (r"\cdot", "·"),
        (r"\times", "×"),
        (r"\approx", "≈"),
        (r"\le", "≤"),
        (r"\ge", "≥"),
        (r"\neq", "≠"),
        (r"\to", "→"),
        (r"\rightarrow", "→"),
        (r"\infty", "∞"),
        (r"\sqrt", "√"),
    ]
    for k, v in repls:
        out = out.replace(k, v)
    out = re.sub(r"[{}]", "", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


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


def _sample_function_points(expr_str: str, x_range: list[float], y_range: list[float], n: int = 160) -> list[list[float]]:
    """Safely evaluate 1D function curves in Python (zero browser eval)."""
    fn = compile_expr(expr_str or "0", ("x",))
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    y_span = max(y1 - y0, 1e-6)
    pts: list[list[float]] = []
    for i in range(n):
        xv = x0 + (x1 - x0) * i / max(1, n - 1)
        try:
            yv = float(fn(x=xv))
        except Exception:
            continue
        if math.isfinite(yv) and (y0 - 0.5 * y_span) <= yv <= (y1 + 0.5 * y_span):
            pts.append([round(xv, 5), round(max(y0 - 0.1 * y_span, min(y1 + 0.1 * y_span, yv)), 5)])
    return pts


def _sample_parametric_points(expr_str: str, t_range: list[float], n: int = 180) -> list[list[float]]:
    if ";" not in expr_str:
        return []
    p0, p1 = expr_str.split(";", 1)
    fx, fy = compile_expr(p0, ("t",)), compile_expr(p1, ("t",))
    t0, t1 = float(t_range[0]), float(t_range[1])
    pts: list[list[float]] = []
    for i in range(n):
        tv = t0 + (t1 - t0) * i / max(1, n - 1)
        try:
            xv, yv = float(fx(t=tv)), float(fy(t=tv))
            if math.isfinite(xv) and math.isfinite(yv):
                pts.append([round(xv, 5), round(yv, 5)])
        except Exception:
            continue
    return pts


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
    pad = 18.0
    if slot == "left" or (slot == "auto" and eff_comp == "side_by_side_comparison"):
        if aspect_label == "9:16" or eff_comp == "stacked_comparison":
            return (stage_x, stage_y, stage_w, stage_h * 0.46)
        if slot == "left":
            return (stage_x, stage_y, stage_w * 0.47, stage_h)
    if slot == "right":
        if aspect_label == "9:16" or eff_comp == "stacked_comparison":
            return (stage_x, stage_y + stage_h * 0.52, stage_w, stage_h * 0.46)
        return (stage_x + stage_w * 0.53, stage_y, stage_w * 0.47, stage_h)
    if slot == "top":
        return (stage_x, stage_y, stage_w, stage_h * 0.46)
    if slot == "bottom":
        return (stage_x, stage_y + stage_h * 0.52, stage_w, stage_h * 0.46)
    if eff_comp == "main_with_equation_panel":
        return (stage_x + pad, stage_y + 72.0, stage_w - 2 * pad, stage_h - 80.0)
    return (stage_x + pad, stage_y + pad, stage_w - 2 * pad, stage_h - 2 * pad)


def compile_plan_to_html(
    plan: "VideoPlanV3",
    design: DesignSpec,
    audio_timeline: AudioTimeline | None = None,
) -> tuple[str, float]:
    """Compile VideoPlanV3 into standalone deterministic offline HTML5/SVG + Motion runtime."""
    plan = materialize_constructions(plan)
    sz = design.safe_zone
    lk = design.look
    tp = design.typography
    mt = design.motion

    stage_x = float(sz.side_margin_px)
    stage_y = float(sz.top_band_px + 28)
    stage_w = float(sz.width - 2 * sz.side_margin_px)
    stage_h = float(sz.height - stage_y - sz.bottom_caption_band_px - 68)

    # Precompute beat windows & synced captions
    beat_windows: list[dict[str, Any]] = []
    if audio_timeline is not None and audio_timeline.beats:
        t_total = float(audio_timeline.total_duration)
        by_id = {b.id: b for b in audio_timeline.beats}
        cursor = 0.0
        for sc in plan.scenes:
            sb = by_id.get(sc.scene_id)
            if sb is not None:
                start_s, end_s = float(sb.start_time), float(sb.end_time)
                words = [{"w": w[0], "s": round(float(w[1]), 4), "e": round(float(w[2]), 4)} for w in sb.words]
                cursor = end_s
            else:
                est = max(2.5, len(sc.narration.split()) / 2.5 + 0.3)
                start_s, end_s = cursor, cursor + est
                words = []
                cursor = end_s
            beat_windows.append(
                {
                    "scene_id": sc.scene_id,
                    "start": round(start_s, 4),
                    "end": round(end_s, 4),
                    "speaker": getattr(sc, "speaker", "narrator"),
                    "purpose": sc.purpose,
                    "caption": re.sub(r"<bookmark\b[^>]*>|\*", "", sc.narration).strip(),
                    "words": words,
                }
            )
    else:
        cursor = 0.0
        for sc in plan.scenes:
            clean_txt = re.sub(r"<bookmark\b[^>]*>|\*", "", sc.narration).strip()
            toks = clean_txt.split()
            dur = max(2.5, len(toks) / 2.5 + 0.35)
            w_dur = (dur - 0.3) / max(1, len(toks))
            words = [
                {"w": tok, "s": round(cursor + i * w_dur, 4), "e": round(cursor + (i + 1) * w_dur, 4)}
                for i, tok in enumerate(toks)
            ]
            beat_windows.append(
                {
                    "scene_id": sc.scene_id,
                    "start": round(cursor, 4),
                    "end": round(cursor + dur, 4),
                    "speaker": getattr(sc, "speaker", "narrator"),
                    "purpose": sc.purpose,
                    "caption": clean_txt,
                    "words": words,
                }
            )
            cursor += dur
        t_total = max(0.5, cursor)

    # Precompute persistent coordinate frames and pre-sampled scene objects
    axes_specs: dict[str, tuple[list[float], list[float], tuple[float, float, float, float]]] = {}
    live_specs: dict[str, dict[str, Any]] = {}
    elements_payload: list[dict[str, Any]] = []

    def _to_px(xy: list[float], box: tuple[float, float, float, float], xr: list[float], yr: list[float]) -> tuple[float, float]:
        bx, by, bw, bh = box
        px = bx + ((float(xy[0]) - xr[0]) / max(xr[1] - xr[0], 1e-6)) * bw
        py = by + bh - ((float(xy[1]) - yr[0]) / max(yr[1] - yr[0], 1e-6)) * bh
        return round(px, 2), round(py, 2)

    for s_idx, sc in enumerate(plan.scenes):
        bw_info = beat_windows[s_idx]
        t_start, t_end = float(bw_info["start"]), float(bw_info["end"])
        comp = adapt_composition_for_aspect(str(getattr(sc, "composition", "auto")), sz.aspect_label)

        # Compute unified un-axed geometry bounding box if scene introduces un-axed polygons
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

        # Track retirements & prevent superseded center-stage apparatus overlap
        active_eq = [o for o in sc.objects if o.kind == "equation"]
        referenced_ids = (
            {e.target_id for e in sc.events}
            | {e.source_id for e in sc.events if e.source_id}
            | {o.axes_id for o in sc.objects if o.axes_id}
        )
        if any(o.kind in ("axes", "table", "matrix", "bar_chart", "graph_network") for o in sc.objects):
            retired_axes: set[str] = set()
            for el in elements_payload:
                if el["hide_at"] > t_start and el["id"] not in referenced_ids and el.get("axes_id") not in referenced_ids:
                    if el["kind"] in ("axes", "table", "matrix", "bar_chart", "graph_network", "annotation", "function_plot", "parametric_curve", "data_series"):
                        el["hide_at"] = round(t_start, 4)
                        if el["kind"] == "axes":
                            retired_axes.add(el["id"])
            if retired_axes:
                for el in elements_payload:
                    if el["hide_at"] > t_start and el.get("axes_id") in retired_axes and el["id"] not in referenced_ids:
                        el["hide_at"] = round(t_start, 4)

        for ev in sc.events:
            if ev.op == "remove":
                for el in elements_payload:
                    if el["id"] == ev.target_id and el["hide_at"] > t_start:
                        el["hide_at"] = round(t_start + min(0.4, mt.exit_s), 4)

        for obj in sc.objects:
            col = _role_color(design, obj.color_role)
            box = _slot_box(obj.panel_slot, comp, stage_x, stage_y, stage_w, stage_h, sz.aspect_label)
            hide_at = round(t_total + 1.0, 4) if obj.persist else round(t_end, 4)
            el: dict[str, Any] = {
                "id": obj.id,
                "kind": obj.kind,
                "axes_id": obj.axes_id,
                "color": col,
                "show_at": t_start,
                "hide_at": hide_at,
                "box": [round(v, 1) for v in box],
                "label": obj.label,
            }

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
                px_pts = [_to_px(pt, ax_box, xr, yr) for pt in raw_pts]
                el["polyline"] = px_pts
            elif obj.kind == "bar_chart" and obj.values:
                vals = [float(v) for v in obj.values]
                mx = max((abs(v) for v in vals), default=1.0) or 1.0
                el["bars"] = [
                    {
                        "label": obj.bar_labels[i] if i < len(obj.bar_labels) else str(i),
                        "val": v,
                        "frac": round(abs(v) / mx, 4),
                    }
                    for i, v in enumerate(vals)
                ]
            elif obj.kind in ("polygon", "probability_region") and obj.vertices:
                xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
                px_pts = [_to_px(v[:2], ax_box, xr, yr) for v in obj.vertices]
                el["polygon"] = px_pts
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
                el["card"] = bool(obj.card)
            elif obj.kind == "annotation":
                xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
                pos = _to_px(obj.coords or [0.0, 0.0], ax_box, xr, yr) if obj.coords else (round(box[0] + box[2] * 0.5, 1), round(box[1] + 42.0, 1))
                el["text_pos"] = list(pos)
                el["text"] = obj.text or obj.label
            elif obj.kind == "matrix":
                el["matrix"] = obj.matrix_rows or [["a", "b"], ["c", "d"]]
            elif obj.kind == "table":
                el["headers"] = obj.table_headers
                el["rows"] = obj.table_rows
            elif obj.kind == "graph_network":
                xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
                nodes = [_to_px(p[:2], ax_box, xr, yr) for p in obj.points_list]
                el["network"] = {"nodes": nodes, "labels": obj.node_labels, "edges": obj.edge_pairs}
            elif obj.kind == "particle_system":
                xr, yr, ax_box = axes_specs.get(obj.axes_id or "", (default_geom_xr, default_geom_yr, box))
                el["particles"] = [_to_px(p[:2], ax_box, xr, yr) for p in obj.points_list]
            elif obj.kind == "number_line":
                el["number_line"] = {"xr": (obj.x_range or [-4.0, 4.0])[:2]}

            live_specs[obj.id] = el
            elements_payload.append(el)

        # Avoid stacking persistent equation cards forever
        if active_eq:
            latest_eq_id = active_eq[-1].id
            for el in elements_payload:
                if el["kind"] == "equation" and el["id"] != latest_eq_id and el["hide_at"] > t_start:
                    el["hide_at"] = round(t_start, 4)

    payload_json = json.dumps(
        {
            "title": plan.title,
            "width": sz.width,
            "height": sz.height,
            "stage": [stage_x, stage_y, stage_w, stage_h],
            "top_band": sz.top_band_px,
            "bottom_band": sz.bottom_caption_band_px,
            "side_margin": sz.side_margin_px,
            "duration": round(t_total, 4),
            "look": lk.model_dump(),
            "typo": tp.model_dump(),
            "motion": mt.model_dump(),
            "beats": beat_windows,
            "elements": elements_payload,
        },
        ensure_ascii=False,
    )

    safe_title = html.escape(plan.title)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{safe_title}</title>
<style>
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
#header-band {{
  position: absolute;
  left: {sz.side_margin_px}px;
  right: {sz.side_margin_px}px;
  top: 18px;
  height: {max(44, sz.top_band_px - 22)}px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid {lk.surface_border};
  padding-bottom: 8px;
}}
#doc-title {{
  font-family: {tp.display_font};
  font-size: {tp.subtitle_px}px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: {lk.ink_primary};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
#beat-pill {{
  font-family: {tp.mono_font};
  font-size: {max(16, tp.body_px - 6)}px;
  padding: 6px 14px;
  border-radius: 999px;
  background: {lk.surface};
  border: 1px solid {lk.surface_border};
  color: {lk.ink_muted};
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
  left: {sz.side_margin_px + 20}px;
  right: {sz.side_margin_px + 20}px;
  top: {sz.top_band_px + 4}px;
  display: flex;
  justify-content: center;
}}
.hf-eq-inner {{
  background: {lk.surface};
  border: 1.5px solid {lk.accent_highlight};
  border-radius: 14px;
  padding: 10px 26px;
  font-family: {tp.mono_font};
  font-size: {tp.math_label_px}px;
  font-weight: 700;
  color: {lk.ink_primary};
}}
.hf-table {{
  position: absolute;
  border-collapse: collapse;
  background: {lk.surface};
  border: 1px solid {lk.surface_border};
  border-radius: 10px;
  overflow: hidden;
  font-family: {tp.mono_font};
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
  gap: 8px 18px;
  padding: 14px 24px;
  border-left: 3px solid {lk.accent_primary};
  border-right: 3px solid {lk.accent_primary};
  background: {lk.surface};
  font-family: {tp.mono_font};
  font-size: {tp.math_label_px}px;
  font-weight: 700;
}}
#caption-band {{
  position: absolute;
  left: {sz.side_margin_px}px;
  right: {sz.side_margin_px}px;
  bottom: 24px;
  min-height: {max(80, sz.bottom_caption_band_px - 36)}px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 8px;
}}
#speaker-badge {{
  font-family: {tp.mono_font};
  font-size: {max(16, tp.body_px - 4)}px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 4px 12px;
  border-radius: 999px;
  background: {lk.surface};
  border: 1px solid {lk.accent_secondary};
  color: {lk.accent_secondary};
  display: none;
}}
#caption-text {{
  font-size: {tp.caption_px}px;
  line-height: 1.26;
  font-weight: 700;
  color: {lk.ink_primary};
  max-width: {int(stage_w * 0.96)}px;
}}
.word-active {{ color: {lk.accent_highlight}; }}
.word-past {{ color: {lk.ink_primary}; }}
.word-future {{ color: {lk.ink_muted}; opacity: 0.85; }}
</style>
</head>
<body>
<div id="canvas-root">
  <div id="header-band">
    <div id="doc-title">{safe_title}</div>
    <div id="beat-pill">beat 1</div>
  </div>
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
(() => {{
  const DATA = {payload_json};
  const SVG_NS = "http://www.w3.org/2000/svg";
  const layers = document.getElementById("svg-layers");
  const overlay = document.getElementById("html-overlay");
  const beatPill = document.getElementById("beat-pill");
  const spBadge = document.getElementById("speaker-badge");
  const capText = document.getElementById("caption-text");

  // OSI-clean closed-form Motion Canvas timing & damped spring evaluators (MIT pattern, zero external runtime)
  function clamp01(t) {{ return t < 0 ? 0 : (t > 1 ? 1 : t); }}
  function easeOutExpo(t) {{ t = clamp01(t); return t >= 1 ? 1 : 1 - Math.pow(2, -10 * t); }}
  function easeInOutCubic(t) {{ t = clamp01(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }}
  function springEval(t, enterS, zeta, stiffness) {{
    if (t <= 0) return 0;
    const wn = Math.sqrt(Math.max(10, stiffness));
    const tau = t * (3.5 / Math.max(0.2, enterS));
    if (zeta < 0.99) {{
      const wd = wn * Math.sqrt(1 - zeta * zeta);
      const env = Math.exp(-zeta * wn * tau);
      return 1 - env * (Math.cos(wd * tau) + (zeta * wn / wd) * Math.sin(wd * tau));
    }}
    return easeOutExpo(t / Math.max(0.15, enterS));
  }}

  // Install VIRTUAL_TIME_SHIM determinism for deterministic frame seek
  let __vtimeMs = 0;
  if (window.performance && window.performance.now) {{
    window.performance.now = () => __vtimeMs;
  }}

  function svgEl(tag, attrs) {{
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    return el;
  }}

  function renderAtTime(t) {{
    __vtimeMs = t * 1000.0;
    layers.innerHTML = "";
    overlay.innerHTML = "";

    // Active beat lookup
    let activeBeat = DATA.beats[DATA.beats.length - 1] || null;
    let activeIdx = 0;
    for (let i = 0; i < DATA.beats.length; i++) {{
      if (t >= DATA.beats[i].start && t < DATA.beats[i].end) {{
        activeBeat = DATA.beats[i];
        activeIdx = i;
        break;
      }}
    }}
    if (activeBeat) {{
      beatPill.textContent = `${{activeIdx + 1}} / ${{DATA.beats.length}} • ${{activeBeat.purpose.slice(0, 34)}}`;
      if (activeBeat.speaker && activeBeat.speaker.toLowerCase() !== "narrator") {{
        spBadge.style.display = "inline-block";
        spBadge.textContent = activeBeat.speaker.replace("_", " ");
      }} else {{
        spBadge.style.display = "none";
      }}
      if (activeBeat.words && activeBeat.words.length > 0) {{
        capText.innerHTML = "";
        let activeWIdx = activeBeat.words.findIndex(w => t >= w.s && t <= w.e);
        if (activeWIdx < 0) {{
          activeWIdx = Math.max(0, activeBeat.words.filter(w => t > w.e).length - 1);
        }}
        const winSize = 6;
        const startIdx = Math.floor(activeWIdx / winSize) * winSize;
        const chunk = activeBeat.words.slice(startIdx, startIdx + winSize);
        for (const w of chunk) {{
          const span = document.createElement("span");
          span.textContent = w.w + " ";
          if (t >= w.s && t <= w.e) span.className = "word-active";
          else if (t > w.e) span.className = "word-past";
          else span.className = "word-future";
          capText.appendChild(span);
        }}
      }} else {{
        capText.textContent = activeBeat.caption;
      }}
    }}

    const enterDur = DATA.motion.enter_s || 0.45;
    for (const el of DATA.elements) {{
      if (t < el.show_at || t >= el.hide_at) continue;
      const p = clamp01(easeOutExpo((t - el.show_at) / enterDur));
      const [bx, by, bw, bh] = el.box;

      if (el.kind === "axes" && el.axes) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id }});
        g.appendChild(svgEl("rect", {{ x: bx, y: by, width: bw, height: bh, fill: DATA.look.surface, "fill-opacity": 0.25, stroke: DATA.look.grid_line, "stroke-width": 1, rx: 10 }}));
        g.appendChild(svgEl("line", {{ x1: bx, y1: el.axes.oy, x2: bx + bw, y2: el.axes.oy, stroke: DATA.look.ink_muted, "stroke-width": 2 }}));
        g.appendChild(svgEl("line", {{ x1: el.axes.ox, y1: by + bh, x2: el.axes.ox, y2: by, stroke: DATA.look.ink_muted, "stroke-width": 2 }}));
        layers.appendChild(g);
      }} else if ((el.kind === "function_plot" || el.kind === "parametric_curve" || el.kind === "waveform" || el.kind === "data_series") && el.polyline && el.polyline.length > 1) {{
        const nVis = Math.max(2, Math.floor(el.polyline.length * p));
        const ptsStr = el.polyline.slice(0, nVis).map(pt => `${{pt[0]}},${{pt[1]}}`).join(" ");
        layers.appendChild(svgEl("polyline", {{
          points: ptsStr, fill: "none", stroke: el.color, "stroke-width": 4, "stroke-linecap": "round", "stroke-linejoin": "round",
          "data-hf-elem": el.id
        }}));
      }} else if ((el.kind === "polygon" || el.kind === "probability_region") && el.polygon && el.polygon.length >= 3) {{
        const ptsStr = el.polygon.map(pt => `${{pt[0]}},${{pt[1]}}`).join(" ");
        layers.appendChild(svgEl("polygon", {{
          points: ptsStr, fill: el.color, "fill-opacity": 0.30 * p, stroke: el.color, "stroke-width": 3, opacity: p, "data-hf-elem": el.id
        }}));
      }} else if (el.kind === "circle" && el.circle) {{
        layers.appendChild(svgEl("circle", {{
          cx: el.circle.cx, cy: el.circle.cy, r: el.circle.r * p, fill: "none", stroke: el.color, "stroke-width": 3.2, opacity: p, "data-hf-elem": el.id
        }}));
      }} else if (el.kind === "point" && el.point) {{
        const pt = el.orbit && el.orbit.length > 1
          ? el.orbit[Math.min(el.orbit.length - 1, Math.floor((el.orbit.length - 1) * easeInOutCubic((t - el.show_at) / Math.max(0.5, el.hide_at - el.show_at))))]
          : el.point;
        layers.appendChild(svgEl("circle", {{ cx: pt[0], cy: pt[1], r: 8 * springEval(t - el.show_at, enterDur, 0.72, 180), fill: el.color, "data-hf-elem": el.id }}));
      }} else if ((el.kind === "vector" || el.kind === "line") && el.segment) {{
        const x2 = el.segment.x1 + (el.segment.x2 - el.segment.x1) * p;
        const y2 = el.segment.y1 + (el.segment.y2 - el.segment.y1) * p;
        const attrs = {{ x1: el.segment.x1, y1: el.segment.y1, x2: x2, y2: y2, stroke: el.color, "stroke-width": 3.5, "data-hf-elem": el.id }};
        if (el.segment.arrow) attrs["marker-end"] = "url(#hf-arrow)";
        layers.appendChild(svgEl("line", attrs));
      }} else if (el.kind === "bar_chart" && el.bars) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id }});
        const n = el.bars.length;
        const colW = bw / Math.max(1, n);
        const baseY = by + bh * 0.85;
        const maxH = bh * 0.68;
        el.bars.forEach((b, i) => {{
          const barH = Math.max(2, b.frac * maxH * p);
          const x = bx + i * colW + colW * 0.18;
          g.appendChild(svgEl("rect", {{ x: x, y: baseY - barH, width: Math.max(6, colW * 0.64), height: barH, fill: el.color, rx: 4 }}));
          if (i % Math.max(1, Math.ceil(n / 8)) === 0) {{
            const txt = svgEl("text", {{ x: x + colW * 0.32, y: baseY + 24, fill: DATA.look.ink_muted, "font-size": 20, "text-anchor": "middle" }});
            txt.textContent = b.label;
            g.appendChild(txt);
          }}
        }});
        layers.appendChild(g);
      }} else if (el.kind === "graph_network" && el.network) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id }});
        const ns = el.network.nodes || [];
        for (const edge of (el.network.edges || [])) {{
          const si = parseInt(edge[0], 10), di = parseInt(edge[1], 10);
          if (ns[si] && ns[di]) {{
            g.appendChild(svgEl("line", {{ x1: ns[si][0], y1: ns[si][1], x2: ns[di][0], y2: ns[di][1], stroke: DATA.look.accent_secondary, "stroke-width": 2.5 }}));
          }}
        }}
        ns.forEach((pt, idx) => {{
          g.appendChild(svgEl("circle", {{ cx: pt[0], cy: pt[1], r: 24, fill: DATA.look.surface, stroke: el.color, "stroke-width": 3 }}));
          const lbl = svgEl("text", {{ x: pt[0], y: pt[1] + 6, fill: DATA.look.ink_primary, "font-size": 18, "font-weight": "bold", "text-anchor": "middle" }});
          lbl.textContent = (el.network.labels && el.network.labels[idx]) ? el.network.labels[idx] : `S${{idx}}`;
          g.appendChild(lbl);
        }});
        layers.appendChild(g);
      }} else if (el.kind === "particle_system" && el.particles) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id }});
        for (const pt of el.particles) {{
          g.appendChild(svgEl("circle", {{ cx: pt[0], cy: pt[1], r: 4.5, fill: el.color }}));
        }}
        layers.appendChild(g);
      }} else if (el.kind === "equation") {{
        const card = document.createElement("div");
        card.className = "hf-eq-card";
        card.setAttribute("data-hf-elem", el.id);
        card.style.opacity = String(p);
        const inner = document.createElement("div");
        inner.className = "hf-eq-inner";
        inner.style.borderColor = el.color;
        inner.textContent = el.eq_text || el.label;
        card.appendChild(inner);
        overlay.appendChild(card);
      }} else if (el.kind === "matrix" && el.matrix) {{
        const mat = document.createElement("div");
        mat.className = "hf-matrix";
        mat.setAttribute("data-hf-elem", el.id);
        mat.style.opacity = String(p);
        const cols = el.matrix[0] ? el.matrix[0].length : 2;
        mat.style.gridTemplateColumns = `repeat(${{cols}}, auto)`;
        mat.style.left = `${{Math.round(bx + bw * 0.28)}}px`;
        mat.style.top = `${{Math.round(by + bh * 0.30)}}px`;
        for (const r of el.matrix) {{
          for (const c of r) {{
            const cell = document.createElement("span");
            cell.textContent = String(c);
            mat.appendChild(cell);
          }}
        }}
        overlay.appendChild(mat);
      }} else if (el.kind === "table") {{
        const tbl = document.createElement("table");
        tbl.className = "hf-table";
        tbl.setAttribute("data-hf-elem", el.id);
        tbl.style.opacity = String(p);
        tbl.style.left = `${{Math.round(bx + 20)}}px`;
        tbl.style.top = `${{Math.round(by + 30)}}px`;
        if (el.headers && el.headers.length) {{
          const tr = document.createElement("tr");
          for (const h of el.headers) {{ const th = document.createElement("th"); th.textContent = h; tr.appendChild(th); }}
          tbl.appendChild(tr);
        }}
        for (const row of (el.rows || [])) {{
          const tr = document.createElement("tr");
          for (const v of row) {{ const td = document.createElement("td"); td.textContent = v; tr.appendChild(td); }}
          tbl.appendChild(tr);
        }}
        overlay.appendChild(tbl);
      }} else if (el.kind === "annotation" && el.text_pos) {{
        const txt = svgEl("text", {{ x: el.text_pos[0], y: el.text_pos[1], fill: el.color, "font-size": DATA.typo.body_px, "font-weight": "700", "text-anchor": "middle", opacity: p, "data-hf-elem": el.id }});
        txt.textContent = el.text || el.label;
        layers.appendChild(txt);
      }}
    }}
  }}

  window.__HF_SEEK = (t) => {{
    renderAtTime(Number(t) || 0);
    return true;
  }};

  window.__HF_AUDIT_LAYOUT = () => {{
    const issues = [];
    const capRect = document.getElementById("caption-band").getBoundingClientRect();
    const hdrRect = document.getElementById("header-band").getBoundingClientRect();
    const elems = Array.from(document.querySelectorAll("[data-hf-elem]"));
    for (const node of elems) {{
      const r = node.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const eid = node.getAttribute("data-hf-elem");
      if (r.left < 0 || r.right > DATA.width) issues.push(`${{eid}}: horizontal overflow`);
      if (r.bottom > capRect.top + 4 && !node.classList.contains("hf-eq-card")) issues.push(`${{eid}}: overlaps caption band`);
      if (r.top < hdrRect.bottom - 2 && !node.classList.contains("hf-eq-card")) issues.push(`${{eid}}: overlaps header band`);
    }}
    return issues;
  }};

  renderAtTime(0);
}})();
</script>
</body>
</html>
"""
    return html_doc, t_total


def _launch_chrome_cdp(url: str, width: int, height: int, user_data_dir: Path) -> tuple[subprocess.Popen[str], str]:
    chrome = find_chromium_binary()
    if not chrome:
        raise RuntimeError("No local Chrome/Chromium/Edge binary found on system for HyperFramesBackend.")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-extensions",
        "--mute-audio",
        f"--user-data-dir={user_data_dir}",
        "--remote-debugging-port=0",
        f"--window-size={width},{height}",
        "about:blank",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ws_url = ""
    deadline = time.time() + 15.0
    port_file = user_data_dir / "DevToolsActivePort"
    while time.time() < deadline:
        if port_file.exists():
            lines = [ln.strip() for ln in port_file.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            if len(lines) >= 2:
                ws_url = f"ws://127.0.0.1:{lines[0]}{lines[1]}"
                break
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Chromium exited early ({proc.returncode}): {err}")
        time.sleep(0.05)

    if not ws_url:
        proc.terminate()
        raise RuntimeError("Timed out waiting for Chromium DevToolsActivePort.")

    # Resolve target WebSocket via browser HTTP JSON list
    import urllib.request

    port = ws_url.split("://127.0.0.1:", 1)[1].split("/", 1)[0]
    pages_json = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5).read().decode("utf-8")
    pages = json.loads(pages_json)
    page_ws = next((p["webSocketDebuggerUrl"] for p in pages if p.get("type") == "page" and "webSocketDebuggerUrl" in p), "")
    if not page_ws:
        proc.terminate()
        raise RuntimeError("No page WebSocket debugger URL exposed by Chromium.")
    return proc, page_ws


class _CDPClient:
    """Minimal synchronous CDP WebSocket client using `websockets`."""

    def __init__(self, ws_url: str) -> None:
        from websockets.sync.client import connect

        self._ws = connect(ws_url, max_size=32 * 1024 * 1024)
        self._msg_id = 0

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._msg_id += 1
        cid = self._msg_id
        self._ws.send(json.dumps({"id": cid, "method": method, "params": params or {}}))
        while True:
            raw = self._ws.recv()
            msg = json.loads(raw)
            # Fail blocked external requests immediately if Fetch.requestPaused fires
            if msg.get("method") == "Fetch.requestPaused":
                rid = msg["params"]["requestId"]
                req_url = str(msg["params"].get("request", {}).get("url", ""))
                allow = req_url.startswith(("http://127.0.0.1", "http://localhost", "data:", "blob:", "about:"))
                self._msg_id += 1
                if allow:
                    self._ws.send(json.dumps({"id": self._msg_id, "method": "Fetch.continueRequest", "params": {"requestId": rid}}))
                else:
                    self._ws.send(json.dumps({"id": self._msg_id, "method": "Fetch.failRequest", "params": {"requestId": rid, "errorReason": "BlockedByClient"}}))
                continue
            if msg.get("id") == cid:
                if "error" in msg:
                    raise RuntimeError(f"CDP {method} failed: {msg['error']}")
                return msg.get("result", {})

    def close(self) -> None:
        self._ws.close()


class HyperFramesBackend(RenderBackend):
    """Local offline HTML5+SVG+Chromium CDP composition & deterministic capture backend."""

    name = "hyperframes"

    def render(
        self,
        plan: "VideoPlanV3",
        *,
        slug: str,
        format_name: str = "widescreen_hd",
        look_name: str = "midnight_studio",
        audio_timeline: AudioTimeline | None = None,
        out_path: Path | None = None,
        build_dir: Path | None = None,
        subtitle_ass: Path | None = None,
    ) -> BackendRenderResult:
        t0 = time.perf_counter()
        prof = FORMATS.get(format_name, FORMATS["widescreen_hd"])
        bdir = build_dir or (PROJECT_ROOT / "build" / f"backend_hyperframes_{slug}_{format_name}")
        bdir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_path or (PROJECT_ROOT / "output" / f"HF_{slug}_{format_name}.mp4")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        if audio_timeline is None:
            beats = [
                BeatSpec(id=s.scene_id, narration=s.narration, caption=s.purpose, pause_after=0.30, speaker=s.speaker)
                for s in plan.scenes
            ]
            audio_timeline = build_voice_timeline(beats, build_dir=bdir, rate="-4%", engine="kokoro")

        SfxLibrary(bdir / "sfx").ensure_all()
        bgm_wav = generate_ambient_bgm(bdir / "bgm.wav", duration=audio_timeline.total_duration + 2.0)

        design = default_design_spec(format_name=format_name, look_name=look_name)
        html_doc, total_dur = compile_plan_to_html(plan, design, audio_timeline=audio_timeline)
        site_dir = bdir / "site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text(html_doc, encoding="utf-8")

        raw_mp4 = bdir / f"{slug}_hf_raw.mp4"
        fps = prof.fps
        total_frames = max(1, int(math.ceil(audio_timeline.total_duration * fps)) + 2)
        ffmpeg_bin = find_ffmpeg()

        ff_cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-framerate", str(fps),
            "-i", "-",
            "-f", "lavfi",
            "-i", f"anullsrc=r=48000:cl=stereo:d={audio_timeline.total_duration + 0.5:.3f}",
            "-shortest",
            "-c:v", "libx264",
            "-preset", "faster",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ar", "48000",
            str(raw_mp4),
        ]

        server, port = _start_loopback_server(site_dir)
        tmp_profile = Path(tempfile.mkdtemp(prefix="hf_chrome_"))
        layout_issues: list[str] = []
        proc: subprocess.Popen[str] | None = None
        cdp: _CDPClient | None = None
        ff_proc: subprocess.Popen[bytes] | None = None

        try:
            url = f"http://127.0.0.1:{port}/index.html"
            proc, page_ws = _launch_chrome_cdp(url, prof.pixel_width, prof.pixel_height, tmp_profile)
            cdp = _CDPClient(page_ws)
            cdp.call("Fetch.enable", {"patterns": [{"urlPattern": "*"}]})
            cdp.call("Emulation.setDeviceMetricsOverride", {
                "width": prof.pixel_width,
                "height": prof.pixel_height,
                "deviceScaleFactor": 1,
                "mobile": False,
            })
            cdp.call("Page.enable")
            cdp.call("Page.navigate", {"url": url})
            time.sleep(0.35)
            cdp.call("Runtime.evaluate", {"expression": "typeof window.__HF_SEEK === 'function'"})

            ff_proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audit_frames = {int(round(0.5 * (sb.start_time + sb.end_time) * fps)) for sb in audio_timeline.beats}
            seen_issues: set[str] = set()

            for f_idx in range(total_frames):
                t_s = f_idx / fps
                cdp.call("Runtime.evaluate", {"expression": f"window.__HF_SEEK({t_s:.6f})"})
                if f_idx in audit_frames:
                    aud = cdp.call("Runtime.evaluate", {"expression": "window.__HF_AUDIT_LAYOUT()", "returnByValue": True})
                    vals = aud.get("result", {}).get("value") or []
                    for v in vals:
                        if v not in seen_issues:
                            seen_issues.add(v)
                            layout_issues.append(str(v))
                shot = cdp.call("Page.captureScreenshot", {"format": "jpeg", "quality": 92})
                img_bytes = base64.b64decode(shot["data"])
                assert ff_proc.stdin is not None
                ff_proc.stdin.write(img_bytes)

            if ff_proc.stdin:
                ff_proc.stdin.close()
            _, ff_err = ff_proc.communicate(timeout=180)
            if ff_proc.returncode != 0:
                raise RuntimeError(f"FFmpeg mux failed: {ff_err.decode('utf-8', errors='ignore')}")
        finally:
            if cdp is not None:
                try:
                    cdp.close()
                except Exception:
                    pass
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            server.shutdown()
            server.server_close()
            shutil.rmtree(tmp_profile, ignore_errors=True)

        master_video(
            raw_video=raw_mp4,
            voice_wav=Path(audio_timeline.master_voice_wav),
            bgm_wav=bgm_wav,
            output_mp4=out_mp4,
            look="none",
        )
        dt = time.perf_counter() - t0
        from studio.longform import _probe_duration

        dur = _probe_duration(out_mp4)
        return BackendRenderResult(
            backend=self.name,
            video_path=out_mp4,
            duration_s=dur,
            render_time_s=round(dt, 3),
            peak_rss_mb=_current_process_rss_mb(),
            file_size_bytes=out_mp4.stat().st_size if out_mp4.exists() else 0,
            width=prof.pixel_width,
            height=prof.pixel_height,
            fps=fps,
            audio_sample_rate=48000,
            layout_issues=layout_issues,
            notes=f"Rendered via HyperFrames offline Chromium CDP + safe loopback server ({total_frames} frames).",
        )
