"""Scene-aware rendered-frame quality inspector across scenes, bookmarks, switches, and zones."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from studio.audio.tts import find_ffmpeg
from studio.design.spec import resolve_look
from studio.layout.solver import compute_adaptive_zones


@dataclass
class FrameSampleTarget:
    t: float
    kind: str  # opening | scene_start | scene_mid | scene_end | bookmark | backend_switch | final
    scene_id: str = ""
    has_motion_op: bool = False
    is_intro_fade: bool = False


@dataclass
class FrameAuditResult:
    summary: str
    passed: bool
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scene_motion_ok: bool = True
    caption_zone_ok: bool = True
    median_stage_w: float = 0.0
    mean_fg_ratio: float = 0.0
    max_frame_delta: float = 0.0
    empty_midpoint_scenes: list[str] = field(default_factory=list)
    static_motion_scenes: list[str] = field(default_factory=list)
    edge_clip_frames: int = 0
    contact_sheet_path: str = ""
    report_path: str = ""
    samples: list[dict[str, Any]] = field(default_factory=list)


def plan_scene_aware_timestamps(
    duration_s: float,
    *,
    plan: object | None = None,
    timeline: object | None = None,
    backend_switches_s: list[float] | None = None,
    max_samples: int = 36,
) -> list[FrameSampleTarget]:
    """Select timestamps covering opening, every scene start/mid/end, bookmarks, switches, and final frame."""
    dur = max(0.4, float(duration_s))
    targets: list[FrameSampleTarget] = [
        FrameSampleTarget(t=round(min(0.45, dur * 0.15), 3), kind="opening", scene_id="opening")
    ]
    motion_ops = {"transform", "morph", "move", "rotate", "scale", "trace", "animate_param", "iterate_states", "interpolate_values"}
    scenes = list(getattr(plan, "scenes", []) or []) if plan is not None else []
    tl_beats = {getattr(b, "id", ""): b for b in getattr(timeline, "beats", [])} if timeline is not None else {}

    if scenes:
        cursor = 0.25
        est_total = sum(max(2.5, len(str(getattr(s, "narration", "")).split()) / 2.08 + 0.35) for s in scenes) or dur
        scale_t = dur / est_total
        for sc in scenes:
            sid = str(getattr(sc, "scene_id", ""))
            has_mot = any(str(getattr(e, "op", "")) in motion_ops for e in getattr(sc, "events", []))
            b = tl_beats.get(sid)
            if b is not None:
                s0 = float(getattr(b, "start_time", cursor))
                s1 = float(getattr(b, "end_time", s0 + 2.5))
                marks = dict(getattr(b, "marks", {}) or {})
            else:
                est_d = max(2.5, len(str(getattr(sc, "narration", "")).split()) / 2.08 + 0.35) * scale_t
                s0, s1 = cursor, min(dur, cursor + est_d)
                marks = {}
            cursor = s1
            span = max(0.3, s1 - s0)
            targets.append(FrameSampleTarget(t=round(min(dur - 0.15, s0 + min(0.65, span * 0.22)), 3), kind="scene_start", scene_id=sid, has_motion_op=has_mot, is_intro_fade=True))
            targets.append(FrameSampleTarget(t=round(min(dur - 0.15, s0 + span * 0.55), 3), kind="scene_mid", scene_id=sid, has_motion_op=has_mot))
            targets.append(FrameSampleTarget(t=round(max(0.2, min(dur - 0.18, s1 - min(0.30, span * 0.12))), 3), kind="scene_end", scene_id=sid, has_motion_op=has_mot))
            for _mname, mt in list(marks.items())[:2]:
                targets.append(FrameSampleTarget(t=round(max(0.2, min(dur - 0.18, float(mt) + 0.35)), 3), kind="bookmark", scene_id=sid, has_motion_op=has_mot))
    else:
        cnt = min(12, max(6, int(round(dur / 8.0))))
        for i in range(cnt):
            ts = round(min(max(0.35, dur * (i + 0.5) / cnt), max(0.35, dur - 0.25)), 3)
            targets.append(FrameSampleTarget(t=ts, kind="sample", scene_id=f"seg_{i}"))

    for sw in backend_switches_s or []:
        if 0.2 < sw < dur - 0.2:
            targets.append(FrameSampleTarget(t=round(sw + 0.15, 3), kind="backend_switch", scene_id="switch"))

    targets.append(FrameSampleTarget(t=round(max(0.25, dur - 0.30), 3), kind="final", scene_id="final"))

    # Deduplicate near-identical timestamps while prioritizing scene_mid, opening, final, switch, bookmark
    prio = {"scene_mid": 0, "opening": 1, "final": 1, "backend_switch": 1, "bookmark": 2, "scene_start": 3, "scene_end": 4, "sample": 5}
    dedup: list[FrameSampleTarget] = []
    for item in sorted(targets, key=lambda x: (x.t, prio.get(x.kind, 9))):
        if any(abs(item.t - prev.t) < 0.35 and item.scene_id == prev.scene_id and item.kind == prev.kind for prev in dedup):
            continue
        dedup.append(item)

    if len(dedup) > max_samples:
        # Keep opening, final, all scene_mid + switches first, then evenly stride remaining slots
        essential = [d for d in dedup if d.kind in ("opening", "final", "backend_switch", "scene_mid")]
        if len(essential) >= max_samples:
            step = max(1, len(essential) // max_samples)
            chosen = essential[::step][: max_samples - 1]
            if essential[-1] not in chosen:
                chosen.append(essential[-1])
            return sorted(chosen, key=lambda x: x.t)
        extras = [d for d in dedup if d not in essential]
        room = max_samples - len(essential)
        step = max(1, len(extras) // max(1, room))
        return sorted(essential + extras[::step][:room], key=lambda x: x.t)
    return sorted(dedup, key=lambda x: x.t)


def inspect_video_scene_aware(
    video_path: Path,
    *,
    look_name: str = "midnight_studio",
    format_name: str = "widescreen_hd",
    plan: object | None = None,
    timeline: object | None = None,
    renderer_meta_path: Path | None = None,
    backend_switches_s: list[float] | None = None,
    contact_sheet_out: Path | None = None,
    report_out: Path | None = None,
    max_samples: int = 36,
) -> FrameAuditResult:
    """Decode scene-aware video frames via FFmpeg and measure bounds, voids, contrast, captions, and motion deltas."""
    if not video_path.exists() or video_path.stat().st_size == 0:
        return FrameAuditResult(summary="Missing video file", passed=False, findings=["Missing rendered video for frame QA"])

    ff = find_ffmpeg()
    info = subprocess.run([ff, "-hide_banner", "-i", str(video_path)], capture_output=True, text=True, timeout=30)
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info.stderr)
    dur = (int(m_dur.group(1)) * 3600 + int(m_dur.group(2)) * 60 + float(m_dur.group(3))) if m_dur else 0.0
    if dur <= 0.2:
        return FrameAuditResult(summary=f"Invalid duration {dur:.2f}s", passed=False, findings=[f"Invalid duration {dur:.2f}s"])

    targets = plan_scene_aware_timestamps(
        dur, plan=plan, timeline=timeline, backend_switches_s=backend_switches_s, max_samples=max_samples
    )
    zones = compute_adaptive_zones(format_name, has_header_banner=False, has_equation_strip=False)
    lk = resolve_look(look_name)
    is_light = bool(lk.is_light)

    meta_boxes: dict[str, list[dict[str, Any]]] = {}
    if renderer_meta_path is not None and renderer_meta_path.exists():
        try:
            raw_m = json.loads(renderer_meta_path.read_text(encoding="utf-8"))
            if isinstance(raw_m, dict):
                meta_boxes = {str(k): list(v) for k, v in raw_m.items() if isinstance(v, list)}
        except Exception:
            pass

    tmp_dir = video_path.parent / f".qa_scene_frames_{video_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    decoded: list[tuple[FrameSampleTarget, Image.Image, np.ndarray]] = []
    findings: list[str] = []
    warnings: list[str] = []
    sample_stats: list[dict[str, Any]] = []

    try:
        for i, tgt in enumerate(targets):
            img_p = tmp_dir / f"f_{i:03d}.png"
            subprocess.run(
                [ff, "-y", "-ss", f"{tgt.t:.3f}", "-i", str(video_path), "-frames:v", "1", str(img_p)],
                capture_output=True,
                timeout=20,
            )
            if img_p.exists() and img_p.stat().st_size > 0:
                pil = Image.open(img_p).convert("RGB")
                decoded.append((tgt, pil, np.asarray(pil, dtype=np.float32)))

        if not decoded:
            return FrameAuditResult(summary="Zero frames decoded", passed=False, findings=["FFmpeg decoded zero frames"])

        prev_arr: np.ndarray | None = None
        diffs: list[float] = []
        stage_widths: list[float] = []
        fg_ratios: list[float] = []
        edge_clips = 0
        cap_band_hits = 0
        empty_midpoints: list[str] = []
        arr_by_scene: dict[str, list[tuple[str, np.ndarray, bool]]] = {}

        cap_y0 = max(0, int(zones.caption_box.y))
        for tgt, _pil, arr in decoded:
            h, w, _ = arr.shape
            corners = np.concatenate(
                [arr[:12, :12].reshape(-1, 3), arr[:12, -12:].reshape(-1, 3), arr[-12:, :12].reshape(-1, 3)],
                axis=0,
            )
            bg_rgb = np.median(corners, axis=0)
            dist = np.linalg.norm(arr - bg_rgb, axis=2)
            fg_mask = dist > 22.0
            fg_ratio = float(np.mean(fg_mask))
            fg_ratios.append(fg_ratio)

            y0, y1 = int(h * 0.06), min(int(h * 0.85), max(int(h * 0.50), cap_y0 - 8))
            x0, x1 = int(w * 0.04), int(w * 0.96)
            stage_fg = fg_mask[y0:y1, x0:x1]
            stage_fg_ratio = float(np.mean(stage_fg)) if stage_fg.size else 0.0
            ys, xs = np.where(stage_fg)
            if xs.size > 20:
                span_w = float((xs.max() - xs.min()) / w)
                span_h = float((ys.max() - ys.min()) / h)
            else:
                span_w, span_h = 0.0, 0.0
            stage_widths.append(span_w)

            # Caption band foreground check (detects burned-in caption visibility inside caption_box)
            cap_slice = fg_mask[cap_y0 : int(h * 0.98), x0:x1]
            cap_fg = float(np.mean(cap_slice)) if cap_slice.size else 0.0
            if cap_fg > 0.002:
                cap_band_hits += 1

            edge_l = float(np.mean(fg_mask[int(h * 0.08) : y1, : max(2, int(w * 0.012))]))
            edge_r = float(np.mean(fg_mask[int(h * 0.08) : y1, -max(2, int(w * 0.012)) :]))
            if edge_l > 0.04 or edge_r > 0.04:
                edge_clips += 1

            bg_luma = float(0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2])
            if is_light and bg_luma < 140.0:
                findings.append(f"t={tgt.t:.1f}s ({tgt.scene_id}): light theme low background luma={bg_luma:.0f}")
            elif (not is_light) and bg_luma > 100.0:
                findings.append(f"t={tgt.t:.1f}s ({tgt.scene_id}): dark theme high background luma={bg_luma:.0f}")

            # Scene-level void detection at midpoints/ends (distinguishes sparse dark gridlines from populated math)
            if tgt.kind in ("scene_mid", "scene_end") and stage_fg_ratio < 0.0028 and span_w < 0.12:
                empty_midpoints.append(f"{tgt.scene_id}@t={tgt.t:.1f}s")

            delta = float(np.mean(np.abs(arr - prev_arr))) if prev_arr is not None else 0.0
            if prev_arr is not None:
                diffs.append(delta)
            prev_arr = arr

            if tgt.scene_id not in ("opening", "final", "switch"):
                arr_by_scene.setdefault(tgt.scene_id, []).append((tgt.kind, stage_fg.astype(np.float32), tgt.has_motion_op))

            sample_stats.append(
                {
                    "t": tgt.t,
                    "kind": tgt.kind,
                    "scene_id": tgt.scene_id,
                    "fg_ratio": round(fg_ratio, 4),
                    "stage_fg": round(stage_fg_ratio, 4),
                    "span_w": round(span_w, 3),
                    "span_h": round(span_h, 3),
                    "cap_fg": round(cap_fg, 4),
                    "bg_luma": round(bg_luma, 1),
                    "delta_from_prev": round(delta, 2),
                    "meta_objects": len(meta_boxes.get(tgt.scene_id, [])),
                }
            )

        static_motion_scenes: list[str] = []
        for sid, frames_s in arr_by_scene.items():
            has_mot = any(m_flag for _k, _a, m_flag in frames_s)
            if has_mot and len(frames_s) >= 2:
                max_sc_delta = max(
                    float(np.mean(np.abs(frames_s[i][1] - frames_s[j][1])))
                    for i in range(len(frames_s))
                    for j in range(i + 1, len(frames_s))
                )
                if max_sc_delta < 0.0004:
                    static_motion_scenes.append(sid)

        if empty_midpoints:
            findings.append(f"Empty visual stage at {len(empty_midpoints)} scene midpoint/end frame(s): {empty_midpoints[:5]}")
        if edge_clips > max(1, len(decoded) // 4):
            findings.append(f"{edge_clips}/{len(decoded)} frames clip horizontal safe margins.")
        med_span = float(np.median(stage_widths)) if stage_widths else 0.0
        if med_span < 0.22:
            findings.append(f"Median primary object width {med_span*100:.1f}% < 22% minimum.")
        max_diff = max(diffs) if diffs else 0.0
        if len(decoded) >= 3 and max_diff < 0.7:
            findings.append(f"Entire video appears static (max frame delta={max_diff:.2f}).")
        if static_motion_scenes:
            warnings.append(f"Scenes with motion events showed negligible stage pixel delta: {static_motion_scenes}")

        sheet_path_str = ""
        if contact_sheet_out is not None and decoded:
            cols = 4 if len(decoded) > 12 else 3
            rows = (len(decoded) + cols - 1) // cols
            cw, ch = 420, int(round(420 * decoded[0][1].height / max(1, decoded[0][1].width)))
            sheet = Image.new("RGB", (cols * cw, rows * ch), color=(240, 240, 236) if is_light else (14, 18, 28))
            draw = ImageDraw.Draw(sheet)
            for idx_f, (tgt, pil_img, _) in enumerate(decoded):
                r_i, c_i = divmod(idx_f, cols)
                thumb = pil_img.resize((cw, ch), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (c_i * cw, r_i * ch))
                draw.rectangle([c_i * cw + 6, r_i * ch + 6, c_i * cw + 210, r_i * ch + 28], fill=(15, 20, 32))
                draw.text((c_i * cw + 12, r_i * ch + 10), f"{tgt.t:.1f}s [{tgt.kind[:7]}]", fill=(240, 245, 255))
            contact_sheet_out.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(contact_sheet_out)
            sheet_path_str = str(contact_sheet_out)

        rep_path_str = ""
        if report_out is not None:
            report_out.parent.mkdir(parents=True, exist_ok=True)
            report_out.write_text(
                json.dumps(
                    {
                        "video_path": str(video_path),
                        "duration_s": round(dur, 3),
                        "samples": sample_stats,
                        "empty_midpoint_scenes": empty_midpoints,
                        "static_motion_scenes": static_motion_scenes,
                        "findings": findings,
                        "warnings": warnings,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            rep_path_str = str(report_out)

        mean_fg = float(np.mean(fg_ratios)) if fg_ratios else 0.0
        summary = (
            f"frames={len(decoded)}, median_stage_w={med_span*100:.1f}%, "
            f"mean_fg={mean_fg*100:.1f}%, max_delta={max_diff:.1f}, edge_clips={edge_clips}, empty_mids={len(empty_midpoints)}"
        )
        return FrameAuditResult(
            summary=summary,
            passed=len(findings) == 0,
            findings=findings,
            warnings=warnings,
            scene_motion_ok=(len(static_motion_scenes) == 0),
            caption_zone_ok=(cap_band_hits > 0 or len(decoded) < 2),
            median_stage_w=med_span,
            mean_fg_ratio=mean_fg,
            max_frame_delta=max_diff,
            empty_midpoint_scenes=empty_midpoints,
            static_motion_scenes=static_motion_scenes,
            edge_clip_frames=edge_clips,
            contact_sheet_path=sheet_path_str,
            report_path=rep_path_str,
            samples=sample_stats,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
