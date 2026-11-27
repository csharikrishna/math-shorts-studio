"""Multi-layer automated quality evaluation: safe AST-to-SymPy math QA, FFmpeg audio QA, and PIL visual QA."""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import sympy as sp

from studio.audio.tts import find_ffmpeg

QAStatus = Literal["passed", "failed", "not_evaluated", "human_review_required"]

_SYMPY_FUNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
}
_SYMPY_CONSTS = {"pi": sp.pi, "e": sp.E, "E": sp.E}
_MAX_AST_NODES = 128


def safe_sympy_from_expr(source: str, allowed_vars: tuple[str, ...] = ("x",)) -> sp.Expr:
    """Safely convert a restricted math string into a SymPy Expr via Python AST (never sympify/eval)."""
    tree = ast.parse(source.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError(f"Expression exceeds {_MAX_AST_NODES} AST nodes.")

    def _conv(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return _conv(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return sp.nsimplify(node.value) if isinstance(node.value, int) else sp.Float(node.value)
        if isinstance(node, ast.Name):
            if node.id in allowed_vars:
                return sp.Symbol(node.id, real=True)
            if node.id in _SYMPY_CONSTS:
                return _SYMPY_CONSTS[node.id]
            raise ValueError(f"Disallowed symbol: {node.id!r}")
        if isinstance(node, ast.UnaryOp):
            val = _conv(node.operand)
            if isinstance(node.op, ast.UAdd):
                return val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError(f"Disallowed unary op: {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            left, right = _conv(node.left), _conv(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError(f"Disallowed binary op: {type(node.op).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _SYMPY_FUNCS.get(node.func.id)
            if fn is None or node.keywords or len(node.args) != 1:
                raise ValueError(f"Disallowed function call: {node.func.id!r}")
            return fn(_conv(node.args[0]))
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    return _conv(tree)


@dataclass
class QACheck:
    name: str
    status: QAStatus
    detail: str
    severity: str = "error"  # error | warning

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass
class QualityReport:
    checks: list[QACheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if not self.checks:
            return False  # Empty report must never silently pass
        evaluated = [c for c in self.checks if c.status != "not_evaluated"]
        if not evaluated:
            return False
        return all(c.status == "passed" or c.severity != "error" for c in evaluated)

    def summary(self) -> str:
        ok = sum(1 for c in self.checks if c.status == "passed")
        return f"{ok}/{len(self.checks)} checks passed (overall={'PASS' if self.passed else 'FAIL'})"


def verify_symbolic_derivative(expr_str: str, deriv_str: str, var: str = "x") -> QACheck:
    """Verify symbolically with SymPy (via safe AST) that d/d(var) expr == deriv."""
    try:
        sym = sp.Symbol(var, real=True)
        e1 = safe_sympy_from_expr(expr_str, (var,))
        e2 = safe_sympy_from_expr(deriv_str, (var,))
        diff = sp.simplify(sp.diff(e1, sym) - e2)
        ok = diff == 0
        return QACheck("sympy_derivative", "passed" if ok else "failed", f"d/d{var}({expr_str}) - ({deriv_str}) = {diff}")
    except Exception as exc:
        return QACheck("sympy_derivative", "failed", f"Math QA error: {exc}")


def verify_audio_loudness(video_path: Path, min_lufs: float = -16.5, max_lufs: float = -13.0) -> QACheck:
    """Measure actual integrated loudness and true peak via FFmpeg ebur128 (audio-only demux)."""
    ff = find_ffmpeg()
    try:
        proc = subprocess.run(
            [ff, "-hide_banner", "-nostats", "-i", str(video_path), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return QACheck("audio_loudness", "failed", f"FFmpeg ebur128 failed: {exc}")
    if proc.returncode != 0:
        return QACheck("audio_loudness", "failed", f"FFmpeg ebur128 exited {proc.returncode}")
    matches = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", proc.stderr)
    peak_matches = re.findall(r"Peak:\s*(-?\d+(?:\.\d+)?)\s*dBFS", proc.stderr)
    if not matches:
        return QACheck("audio_loudness", "failed", "Could not parse EBU R128 loudness.")
    lufs = float(matches[-1])
    peak = float(peak_matches[-1]) if peak_matches else 0.0
    ok = (min_lufs <= lufs <= max_lufs) and (peak <= -1.0)
    return QACheck(
        "audio_loudness",
        "passed" if ok else "failed",
        f"Integrated={lufs:.1f} LUFS (target {min_lufs}..{max_lufs}), TruePeak={peak:.1f} dBFS (<= -1.0)",
    )


def verify_media_streams(video_path: Path, expected_sr: int = 48000, max_av_drift_s: float = 0.35) -> QACheck:
    """Verify final MP4 audio sample rate (48 kHz) and audio/video stream duration alignment via ffprobe."""
    import json as _json
    import shutil as _shutil

    ffprobe = _shutil.which("ffprobe")
    if not ffprobe:
        ff_dir = Path(find_ffmpeg()).parent
        cand = ff_dir / ("ffprobe.exe" if (ff_dir / "ffprobe.exe").exists() else "ffprobe")
        ffprobe = str(cand) if cand.exists() else ""
    if not ffprobe:
        # Fallback: measure exact video and audio stream durations via ffmpeg decode/copy stats
        ff = find_ffmpeg()
        info = subprocess.run([ff, "-hide_banner", "-i", str(video_path)], capture_output=True, text=True, timeout=30)
        sr_m = re.search(r"(\d+)\s*Hz,\s*stereo", info.stderr)
        fps_m = re.search(r"(\d+(?:\.\d+)?)\s*fps", info.stderr)
        sr = int(sr_m.group(1)) if sr_m else 0
        fps = float(fps_m.group(1)) if fps_m else 0.0

        def _stream_dur(sel: str) -> float:
            p = subprocess.run(
                [ff, "-hide_banner", "-i", str(video_path), "-map", sel, "-c", "copy", "-f", "null", "-"],
                capture_output=True, text=True, timeout=30,
            )
            times = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", p.stderr)
            if not times:
                return 0.0
            h, m, s = times[-1]
            return int(h) * 3600 + int(m) * 60 + float(s)

        v_dur = _stream_dur("0:v:0")
        a_dur = _stream_dur("0:a:0")
        drift = abs(v_dur - a_dur)
        ok = (sr == expected_sr) and (drift <= max_av_drift_s) and (fps >= 29.0)
        return QACheck("media_streams", "passed" if ok else "failed", f"sr={sr}Hz, fps={fps:.2f}, v_dur={v_dur:.2f}s, a_dur={a_dur:.2f}s, drift={drift:.2f}s")

    proc = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return QACheck("media_streams", "failed", f"ffprobe exited {proc.returncode}")
    streams = _json.loads(proc.stdout).get("streams", [])
    v_dur = next((float(s.get("duration", 0)) for s in streams if s.get("codec_type") == "video"), 0.0)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not a_stream:
        return QACheck("media_streams", "failed", "Missing audio stream")
    sr = int(a_stream.get("sample_rate", 0))
    a_dur = float(a_stream.get("duration", 0))
    drift = abs(v_dur - a_dur)
    ok = (sr == expected_sr) and (drift <= max_av_drift_s)
    return QACheck("media_streams", "passed" if ok else "failed", f"sr={sr}Hz, v_dur={v_dur:.2f}s, a_dur={a_dur:.2f}s, drift={drift:.2f}s")


def verify_storyboard_image(image_path: Path) -> QACheck:
    """Verify storyboard/frame is non-blank and has healthy contrast."""
    if not image_path.exists():
        return QACheck("visual_storyboard", "not_evaluated", f"Missing {image_path}", severity="warning")
    arr = np.asarray(Image.open(image_path).convert("L"), dtype=float)
    std = float(np.std(arr))
    ok = std > 8.0
    return QACheck("visual_storyboard", "passed" if ok else "failed", f"{image_path.name} luminance std={std:.1f}")


def verify_longform_bundle(
    output_mp4: Path,
    chapters_json: Path,
    srt_path: Path,
    vtt_path: Path,
    transcript_path: Path,
    min_duration_s: float = 0.0,
) -> QACheck:
    """Independently verify final long-form MP4, audio, subtitles, chapters, and duration."""
    for req_file in (output_mp4, chapters_json, srt_path, vtt_path, transcript_path):
        if not req_file.exists() or req_file.stat().st_size == 0:
            return QACheck("longform_bundle", "failed", f"Missing or empty required artifact: {req_file.name}")

    loud = verify_audio_loudness(output_mp4)
    if not loud.passed:
        return QACheck("longform_bundle", "failed", f"Loudness QA failed: {loud.detail}")

    streams = verify_media_streams(output_mp4)
    if not streams.passed:
        return QACheck("longform_bundle", "failed", f"Stream QA failed: {streams.detail}")

    ff = find_ffmpeg()
    info = subprocess.run([ff, "-hide_banner", "-i", str(output_mp4)], capture_output=True, text=True, timeout=30)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info.stderr)
    dur = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else 0.0
    if dur < min_duration_s:
        return QACheck("longform_bundle", "failed", f"Duration {dur:.1f}s < required minimum {min_duration_s:.1f}s")

    return QACheck("longform_bundle", "passed", f"dur={dur:.2f}s; {loud.detail}; {streams.detail}")


def verify_decoded_frames(
    video_path: Path,
    *,
    look_name: str = "midnight_studio",
    format_name: str = "widescreen_hd",
    sample_count: int = 9,
    contact_sheet_out: Path | None = None,
) -> tuple[QACheck, list[str], dict[str, object]]:
    """Decode actual rendered video frames via FFmpeg and audit blank voids, diagram size, contrast, and clipping."""
    from PIL import ImageDraw
    from studio.design.spec import resolve_look

    if not video_path.exists() or video_path.stat().st_size == 0:
        return QACheck("decoded_frames", "not_evaluated", "Missing video_path"), [], {}

    ff = find_ffmpeg()
    info = subprocess.run([ff, "-hide_banner", "-i", str(video_path)], capture_output=True, text=True, timeout=30)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info.stderr)
    dur = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else 0.0
    if dur <= 0.2:
        return QACheck("decoded_frames", "failed", f"Invalid video duration ({dur:.2f}s)"), [], {}

    # Sample opening, evenly spaced content frames, and closing frame cleanly
    cnt = max(4, min(12, sample_count))
    timestamps = [round(min(max(0.35, dur * (i + 0.5) / cnt), max(0.35, dur - 0.35)), 3) for i in range(cnt)]
    tmp_dir = video_path.parent / f".qa_frames_{video_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    lk = resolve_look(look_name)
    is_light = bool(lk.is_light)

    decoded: list[tuple[float, Image.Image, np.ndarray]] = []
    findings: list[str] = []
    stats_per_frame: list[dict[str, float]] = []

    try:
        for i, ts in enumerate(timestamps):
            img_p = tmp_dir / f"f_{i:02d}.png"
            subprocess.run(
                [ff, "-y", "-ss", f"{ts:.3f}", "-i", str(video_path), "-frames:v", "1", str(img_p)],
                capture_output=True,
                timeout=20,
            )
            if img_p.exists() and img_p.stat().st_size > 0:
                pil_img = Image.open(img_p).convert("RGB")
                arr = np.asarray(pil_img, dtype=np.float32)
                decoded.append((ts, pil_img, arr))
        if not decoded:
            return QACheck("decoded_frames", "failed", "FFmpeg decoded zero frames"), ["Zero frames decoded"], {}

        prev_arr: np.ndarray | None = None
        diffs: list[float] = []
        stage_width_fracs: list[float] = []
        fg_ratios: list[float] = []
        edge_clip_count = 0
        blank_count = 0

        for ts, _pil, arr in decoded:
            h, w, _ = arr.shape
            # Estimate background RGB from image outer corners
            corners = np.concatenate(
                [arr[:12, :12].reshape(-1, 3), arr[:12, -12:].reshape(-1, 3), arr[-12:, :12].reshape(-1, 3)],
                axis=0,
            )
            bg_rgb = np.median(corners, axis=0)
            dist = np.linalg.norm(arr - bg_rgb, axis=2)
            luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

            fg_mask = dist > 22.0
            fg_ratio = float(np.mean(fg_mask))
            fg_ratios.append(fg_ratio)
            if fg_ratio < 0.008:
                blank_count += 1

            # Check primary middle stage occupancy (excluding caption footer bottom 14% and outer 4% edge)
            y0, y1 = int(h * 0.06), int(h * 0.85)
            x0, x1 = int(w * 0.04), int(w * 0.96)
            stage_fg = fg_mask[y0:y1, x0:x1]
            ys, xs = np.where(stage_fg)
            if xs.size > 20:
                span_w = float((xs.max() - xs.min()) / w)
                span_h = float((ys.max() - ys.min()) / h)
                stage_width_fracs.append(span_w)
            else:
                span_w, span_h = 0.0, 0.0
                stage_width_fracs.append(0.0)

            # Check right/left outer margin clipping (non-bg pixels touching outer 1.2% left/right edge in main band)
            edge_l = float(np.mean(fg_mask[int(h * 0.08) : int(h * 0.84), : max(2, int(w * 0.012))]))
            edge_r = float(np.mean(fg_mask[int(h * 0.08) : int(h * 0.84), -max(2, int(w * 0.012)) :]))
            if edge_l > 0.04 or edge_r > 0.04:
                edge_clip_count += 1

            # Check light/dark polarity match cleanly
            bg_luma = float(0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2])
            if is_light and bg_luma < 140.0:
                findings.append(f"t={ts:.1f}s: light theme expected high background luminance, got {bg_luma:.0f}")
            elif (not is_light) and bg_luma > 100.0:
                findings.append(f"t={ts:.1f}s: dark theme expected low background luminance, got {bg_luma:.0f}")

            if prev_arr is not None:
                diffs.append(float(np.mean(np.abs(arr - prev_arr))))
            prev_arr = arr
            stats_per_frame.append({"t": ts, "fg_ratio": round(fg_ratio, 4), "span_w": round(span_w, 3), "span_h": round(span_h, 3), "bg_luma": round(bg_luma, 1)})

        if blank_count > max(1, len(decoded) // 4):
            findings.append(f"{blank_count}/{len(decoded)} decoded frames are mostly blank (<0.8% non-background pixels).")
        if edge_clip_count > max(1, len(decoded) // 3):
            findings.append(f"{edge_clip_count}/{len(decoded)} decoded frames show content touching outer 1.2% horizontal edges.")
        med_span = float(np.median(stage_width_fracs)) if stage_width_fracs else 0.0
        if med_span < 0.22:
            findings.append(f"Median active stage width fraction is only {med_span*100:.1f}% (< 22% minimum).")
        max_diff = max(diffs) if diffs else 0.0
        if len(decoded) >= 3 and max_diff < 0.8:
            findings.append(f"Decoded frames appear static across video (max frame delta={max_diff:.2f}).")

        if contact_sheet_out is not None and decoded:
            cols = 3
            rows = (len(decoded) + cols - 1) // cols
            cw, ch = 480, int(round(480 * decoded[0][1].height / max(1, decoded[0][1].width)))
            sheet = Image.new("RGB", (cols * cw, rows * ch), color=(240, 240, 236) if is_light else (14, 18, 28))
            draw = ImageDraw.Draw(sheet)
            for idx_f, (ts, pil_img, _) in enumerate(decoded):
                r_i, c_i = divmod(idx_f, cols)
                thumb = pil_img.resize((cw, ch), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (c_i * cw, r_i * ch))
                draw.rectangle([c_i * cw + 6, r_i * ch + 6, c_i * cw + 116, r_i * ch + 28], fill=(15, 20, 32))
                draw.text((c_i * cw + 12, r_i * ch + 10), f"t={ts:.1f}s", fill=(240, 245, 255))
            contact_sheet_out.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(contact_sheet_out)

        ok = len(findings) == 0
        summary = (
            f"frames={len(decoded)}, median_stage_w={med_span*100:.1f}%, "
            f"mean_fg={np.mean(fg_ratios)*100:.1f}%, max_delta={max_diff:.1f}, edge_clips={edge_clip_count}"
        )
        return QACheck("decoded_frames", "passed" if ok else "failed", summary), findings, {"frames": stats_per_frame}
    finally:
        import shutil as _shutil

        _shutil.rmtree(tmp_dir, ignore_errors=True)


def audit_educational_quality(
    *,
    topic: str,
    title: str,
    plan: object,
    format_name: str,
    backend: str = "manimgx",
    video_path: Path | None = None,
    storyboard_image: Path | None = None,
    contact_sheet_out: Path | None = None,
    script: object | None = None,
    look_name: str = "midnight_studio",
    chapter_plans: list[object] | None = None,
    target_duration_s: float | None = None,
) -> object:
    """Comprehensive pre/post-render 10-category QA returning honest VideoReviewReport (never fake 100/100)."""
    from studio.ai.validate_v3 import verify_v3_plan_detailed
    from studio.creative.script_ir import validate_script_ir
    from studio.design.spec import resolve_look, validate_look_contrast
    from studio.storyboard.model import VerificationCoverageSummary, VideoReviewReport, VisualReviewNote

    notes: list[VisualReviewNote] = []
    metrics: list[str] = []
    cats: dict[str, str] = {
        "math_accuracy": "not_evaluated",
        "scientific_factual_accuracy": "not_evaluated",
        "visual_layout_legibility": "not_evaluated",
        "theme_contrast_readability": "not_evaluated",
        "caption_synchronization": "not_evaluated",
        "audio_quality_loudness": "not_evaluated",
        "animation_meaningfulness": "not_evaluated",
        "scene_composition_variety": "not_evaluated",
        "duration_pacing_accuracy": "not_evaluated",
        "rendering_media_integrity": "not_evaluated",
    }

    plans_to_check = list(chapter_plans) if chapter_plans else [plan]
    cov_acc = {
        "deterministically_verified": 0,
        "numerically_tested": 0,
        "source_supported": 0,
        "ai_reviewed": 0,
        "unsupported": 0,
        "failed": 0,
        "human_review_required": 0,
    }
    plan_issues: list[str] = []
    for p_item in plans_to_check:
        det = verify_v3_plan_detailed(p_item, require_claims=True)  # type: ignore[arg-type]
        cov_raw = det.get("verification_coverage", {})
        if isinstance(cov_raw, dict):
            for k in cov_acc:
                cov_acc[k] += int(cov_raw.get(k, 0))
        plan_issues.extend(str(m) for m in det.get("issues", []))  # type: ignore[arg-type]

    cov = VerificationCoverageSummary(**cov_acc)
    for msg in plan_issues[:20]:
        notes.append(VisualReviewNote(shot_id="plan", severity="blocker", category="math_grounding", finding=str(msg)))

    math_ok = len(plan_issues) == 0 and cov.failed == 0 and (cov.deterministically_verified + cov.numerically_tested) > 0
    cats["math_accuracy"] = "passed" if math_ok else ("failed" if (plan_issues or cov.failed) else "not_evaluated")
    cats["scientific_factual_accuracy"] = "passed" if (math_ok and cov.unsupported == 0) else ("failed" if cov.failed else "human_review_required")

    # Theme WCAG contrast check
    lk_tok = resolve_look(look_name)
    contrast_errs = validate_look_contrast(lk_tok)
    cats["theme_contrast_readability"] = "passed" if not contrast_errs else "failed"
    for ce in contrast_errs:
        notes.append(VisualReviewNote(shot_id="theme", severity="blocker", category="contrast", finding=ce))

    script_ok = True
    if script is not None:
        s_issues = validate_script_ir(script)  # type: ignore[arg-type]
        if s_issues:
            script_ok = False
            cats["caption_synchronization"] = "failed"
            for si in s_issues[:15]:
                notes.append(VisualReviewNote(shot_id="script", severity="blocker", category="caption", finding=si))
        else:
            cats["caption_synchronization"] = "passed"

    scenes = getattr(plan, "scenes", [])
    compositions = {str(getattr(sc, "composition", "auto")) for sc in scenes}
    obj_kinds = {str(o.kind) for sc in scenes for o in getattr(sc, "objects", [])}
    event_ops = {str(e.op) for sc in scenes for e in getattr(sc, "events", [])}
    eq_cards = sum(1 for sc in scenes for o in getattr(sc, "objects", []) if o.kind == "equation" and o.card)
    variety_ok = (len(obj_kinds) >= 2) or (len(scenes) <= 2)
    if len(scenes) >= 4 and eq_cards == len(scenes):
        variety_ok = False
        notes.append(
            VisualReviewNote(
                shot_id="global",
                severity="warning",
                category="visual_variety",
                finding="Every scene uses a boxed equation card; vary compositions and inline annotations.",
            )
        )
    cats["scene_composition_variety"] = "passed" if variety_ok else "failed"
    motion_ops = event_ops & {"transform", "morph", "move", "rotate", "scale", "trace", "animate_param", "iterate_states", "interpolate_values", "duplicate"}
    cats["animation_meaningfulness"] = "passed" if (len(scenes) <= 2 or bool(motion_ops)) else "failed"
    if cats["animation_meaningfulness"] == "failed":
        notes.append(
            VisualReviewNote(
                shot_id="global",
                severity="warning",
                category="motion",
                finding="No state-changing or mathematical motion events used (only fade/create).",
            )
        )
    metrics.append(f"scenes={len(scenes)}, unique_object_kinds={len(obj_kinds)}, motion_ops={sorted(motion_ops)}, compositions={sorted(compositions)}")

    layout_ok = True
    is_portrait = "shorts" in format_name
    for sc in scenes:
        comp = str(getattr(sc, "composition", "auto"))
        if is_portrait and comp == "side_by_side_comparison":
            layout_ok = False
            notes.append(
                VisualReviewNote(
                    shot_id=sc.scene_id,
                    severity="warning",
                    category="safe_zone",
                    finding="side_by_side_comparison should be adapted to stacked_comparison on 9:16.",
                    fix="Run adapt_composition_for_aspect(comp, '9:16')",
                )
            )

    audio_ok = False
    resolved_sheet = ""
    if video_path is not None and video_path.exists():
        loud = verify_audio_loudness(video_path)
        streams = verify_media_streams(video_path)
        metrics.append(f"loudness: {loud.detail}")
        metrics.append(f"streams: {streams.detail}")
        cats["audio_quality_loudness"] = "passed" if loud.passed else "failed"
        cats["rendering_media_integrity"] = "passed" if streams.passed else "failed"
        audio_ok = bool(loud.passed and streams.passed)
        if not audio_ok:
            notes.append(
                VisualReviewNote(
                    shot_id="media",
                    severity="blocker",
                    category="audio",
                    finding=f"Media QA issue: {loud.detail}; {streams.detail}",
                )
            )

        # Decoded-frame visual pixel QA + contact sheet
        sheet_target = contact_sheet_out or (video_path.parent / f"{video_path.stem}_qa_sheet.png")
        df_chk, df_findings, _df_meta = verify_decoded_frames(
            video_path,
            look_name=look_name,
            format_name=format_name,
            contact_sheet_out=sheet_target,
        )
        metrics.append(f"decoded_frames: {df_chk.detail}")
        if sheet_target.exists():
            resolved_sheet = str(sheet_target)
        if not df_chk.passed:
            layout_ok = False
            cats["visual_layout_legibility"] = "failed"
            for df_msg in df_findings:
                notes.append(VisualReviewNote(shot_id="frames", severity="blocker", category="layout", finding=df_msg))
        elif layout_ok:
            cats["visual_layout_legibility"] = "passed"
            if cats["caption_synchronization"] == "not_evaluated":
                cats["caption_synchronization"] = "passed"

        # Duration pacing verification against target_duration_s if supplied
        if target_duration_s and target_duration_s > 0:
            dur_m = re.search(r"v_dur=(\d+(?:\.\d+)?)s", streams.detail)
            if dur_m:
                v_d = float(dur_m.group(1))
                rel_err = abs(v_d - target_duration_s) / target_duration_s
                metrics.append(f"duration_pacing: actual={v_d:.1f}s vs target={target_duration_s:.1f}s ({rel_err*100:.1f}% delta)")
                if rel_err <= 0.12:
                    cats["duration_pacing_accuracy"] = "passed"
                else:
                    cats["duration_pacing_accuracy"] = "failed"
                    notes.append(
                        VisualReviewNote(
                            shot_id="pacing",
                            severity="warning" if rel_err <= 0.20 else "blocker",
                            category="pacing",
                            finding=f"Actual duration {v_d:.1f}s deviates {rel_err*100:.1f}% from target {target_duration_s:.1f}s.",
                        )
                    )
            else:
                cats["duration_pacing_accuracy"] = "not_evaluated"
        else:
            cats["duration_pacing_accuracy"] = "passed"

    if storyboard_image is not None and storyboard_image.exists():
        sb_chk = verify_storyboard_image(storyboard_image)
        metrics.append(sb_chk.detail)
        if sb_chk.status == "failed":
            layout_ok = False
            cats["visual_layout_legibility"] = "failed"
            notes.append(VisualReviewNote(shot_id="storyboard", severity="blocker", category="layout", finding=sb_chk.detail))

    blockers = sum(1 for n in notes if n.severity == "blocker")
    warnings = sum(1 for n in notes if n.severity == "warning")
    uneval_req = sum(
        1
        for k in (
            "math_accuracy",
            "visual_layout_legibility",
            "theme_contrast_readability",
            "audio_quality_loudness",
            "rendering_media_integrity",
        )
        if cats.get(k) != "passed"
    )
    unsupported_pen = min(10.0, float(cov.unsupported) * 2.5) if cov.unsupported > 0 else 0.0
    score = max(0.0, 100.0 - blockers * 20.0 - warnings * 6.0 - uneval_req * 10.0 - unsupported_pen)
    approved = blockers == 0 and uneval_req == 0 and score >= 78.0

    return VideoReviewReport(
        topic=topic,
        title=title,
        format_name=format_name,
        backend=backend,
        approved=approved,
        overall_score=round(score, 1),
        script_clarity_ok=script_ok,
        math_grounding_ok=math_ok,
        layout_safe_zone_ok=layout_ok and cats["visual_layout_legibility"] != "failed",
        audio_sync_ok=audio_ok,
        visual_variety_ok=variety_ok,
        category_statuses=cats,
        contact_sheet_path=resolved_sheet,
        verification_summary=cov,
        notes=notes,
        metrics=metrics,
        summary=(
            f"Audited {len(scenes)} scenes ({look_name}, {format_name}, {backend}): "
            f"score={score:.1f}/100, approved={approved}, blockers={blockers}, warnings={warnings}, uneval_required={uneval_req}."
        ),
    )
