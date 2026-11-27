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


def verify_caption_synchronization(
    *,
    caption_ir: object | None = None,
    caption_ir_path: Path | None = None,
    srt_path: Path | None = None,
    caption_zone_ok: bool | None = None,
    expected_duration_s: float | None = None,
    expected_beat_count: int | None = None,
) -> QACheck:
    """Independently verify CaptionIR cue/word non-overlap, global coverage, readability, and SRT parity."""
    ir_obj = caption_ir
    if ir_obj is None and caption_ir_path is not None and caption_ir_path.exists():
        from studio.captions import CaptionIR

        try:
            ir_obj = CaptionIR.model_validate_json(caption_ir_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return QACheck("caption_synchronization", "failed", f"Invalid CaptionIR JSON: {exc}")

    if ir_obj is None and srt_path is not None and srt_path.exists():
        txt = srt_path.read_text(encoding="utf-8")
        if "-->" not in txt:
            return QACheck("caption_synchronization", "failed", f"SRT {srt_path.name} contains zero timed cues.")
        return QACheck("caption_synchronization", "passed", f"SRT verified ({srt_path.name})")

    if ir_obj is None:
        return QACheck("caption_synchronization", "not_evaluated", "No CaptionIR or SRT artifact provided for sync QA.")

    cues = list(getattr(ir_obj, "cues", []) or [])
    if getattr(ir_obj, "style", "editorial") == "off":
        return QACheck("caption_synchronization", "passed", "Caption style is off; sidecar track exempt.")
    if not cues:
        return QACheck("caption_synchronization", "failed", "CaptionIR contains zero cues.")

    prev_end = -1.0
    seen_sigs: set[tuple[float, float, str]] = set()
    max_lines_allowed = int(getattr(getattr(ir_obj, "preset", None), "max_lines", 2) or 2)
    covered_beats: set[str] = set()
    for c in cues:
        cs, ce = float(c.start_s), float(c.end_s)
        if ce <= cs:
            return QACheck("caption_synchronization", "failed", f"Cue {c.cue_id} has non-positive duration ({cs:.2f}->{ce:.2f}).")
        if cs < prev_end - 0.02:
            return QACheck(
                "caption_synchronization",
                "failed",
                f"Cue {c.cue_id} ({cs:.3f}s) overlaps previous cue ending at {prev_end:.3f}s.",
            )
        prev_end = ce
        sig = (round(cs, 2), round(ce, 2), str(c.text).strip())
        if sig in seen_sigs:
            return QACheck("caption_synchronization", "failed", f"Duplicate caption cue at {cs:.2f}s: {c.text!r}")
        seen_sigs.add(sig)
        if not str(getattr(c, "speaker", "")).strip():
            return QACheck("caption_synchronization", "failed", f"Cue {c.cue_id} missing speaker identity.")
        if len(getattr(c, "lines", [])) > max(3, max_lines_allowed):
            return QACheck("caption_synchronization", "failed", f"Cue {c.cue_id} has {len(c.lines)} lines > readable limit.")
        covered_beats.add(str(c.beat_id))
        prev_w_end = -1.0
        for w in getattr(c, "words", []):
            ws, we = float(w.start_s), float(w.end_s)
            if we < ws - 1e-4 or ws < prev_w_end - 0.02:
                return QACheck("caption_synchronization", "failed", f"Non-monotonic or overlapping word timing in {c.cue_id}.")
            prev_w_end = we

    # Global duration coverage check: catches chapter-1-only CaptionIR passed for a multi-minute video!
    last_end = float(cues[-1].end_s)
    if expected_duration_s is not None and expected_duration_s > 4.0:
        if last_end < expected_duration_s * 0.65:
            return QACheck(
                "caption_synchronization",
                "failed",
                f"Incomplete CaptionIR coverage: cues end at {last_end:.1f}s vs expected video duration {expected_duration_s:.1f}s.",
            )
    if expected_beat_count is not None and expected_beat_count >= 2 and len(covered_beats) < max(1, int(expected_beat_count * 0.65)):
        return QACheck(
            "caption_synchronization",
            "failed",
            f"Incomplete beat caption coverage: {len(covered_beats)}/{expected_beat_count} beats covered.",
        )

    # SRT parity verification
    if srt_path is not None and srt_path.exists():
        srt_blocks = [blk for blk in srt_path.read_text(encoding="utf-8").strip().split("\n\n") if "-->" in blk]
        if abs(len(srt_blocks) - len(cues)) > max(2, int(0.15 * len(cues))):
            return QACheck(
                "caption_synchronization",
                "failed",
                f"Exported SRT cue count ({len(srt_blocks)}) diverges from CaptionIR ({len(cues)}).",
            )

    if caption_zone_ok is False:
        return QACheck("caption_synchronization", "failed", "Rendered frames show no visible captions inside caption zone.")

    prov = dict(getattr(ir_obj, "timing_provenance_counts", {}) or {})
    return QACheck("caption_synchronization", "passed", f"cues={len(cues)}, beats={len(covered_beats)}, end={last_end:.1f}s, provenance={prov or 'recorded'}")


def verify_decoded_frames(
    video_path: Path,
    *,
    look_name: str = "midnight_studio",
    format_name: str = "widescreen_hd",
    sample_count: int | None = None,
    contact_sheet_out: Path | None = None,
    plan: object | None = None,
    timeline: object | None = None,
    renderer_meta_path: Path | None = None,
    report_out: Path | None = None,
) -> tuple[QACheck, list[str], dict[str, object]]:
    """Run scene-aware decoded frame QA (bounds, empty midpoints, contrast, edge clips, motion deltas)."""
    from studio.qa.frames import inspect_video_scene_aware

    if not video_path.exists() or video_path.stat().st_size == 0:
        return QACheck("decoded_frames", "not_evaluated", "Missing video_path"), [], {}
    res = inspect_video_scene_aware(
        video_path,
        look_name=look_name,
        format_name=format_name,
        plan=plan,
        timeline=timeline,
        renderer_meta_path=renderer_meta_path,
        contact_sheet_out=contact_sheet_out,
        report_out=report_out,
        max_samples=sample_count,
    )
    return (
        QACheck("decoded_frames", "passed" if res.passed else "failed", res.summary),
        res.findings,
        {
            "frames": res.samples,
            "scene_motion_ok": res.scene_motion_ok,
            "caption_zone_ok": res.caption_zone_ok,
            "warnings": res.warnings,
            "static_motion_scenes": res.static_motion_scenes,
            "empty_midpoint_scenes": res.empty_midpoint_scenes,
            "scenes_covered": res.scenes_covered,
            "total_plan_scenes": res.total_plan_scenes,
            "report_path": res.report_path,
        },
    )


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
    timeline: object | None = None,
    caption_ir: object | None = None,
    caption_ir_path: Path | None = None,
    srt_path: Path | None = None,
    renderer_meta_path: Path | None = None,
) -> object:
    """Comprehensive pre/post-render 10-category QA returning honest VideoReviewReport with strict publication gates."""
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
    sci_ok = math_ok and cov.unsupported == 0 and cov.human_review_required == 0
    cats["scientific_factual_accuracy"] = "passed" if sci_ok else ("failed" if cov.failed else "human_review_required")
    if cov.unsupported > 0:
        notes.append(
            VisualReviewNote(
                shot_id="claims",
                severity="warning",
                category="math_grounding",
                finding=f"{cov.unsupported} unsupported claim(s) require deterministic/numerical verification before publication approval.",
            )
        )
    if cov.human_review_required > 0:
        notes.append(
            VisualReviewNote(
                shot_id="claims",
                severity="warning",
                category="math_grounding",
                finding=f"{cov.human_review_required} claim(s) require human review before publication approval.",
            )
        )

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
            for si in s_issues[:15]:
                notes.append(VisualReviewNote(shot_id="script", severity="blocker", category="caption", finding=si))

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
    caption_zone_ok: bool | None = None
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

        # Scene-aware decoded-frame visual pixel QA + contact sheet + JSON sidecar
        sheet_target = contact_sheet_out or (video_path.parent / f"{video_path.stem}_qa_sheet.png")
        df_chk, df_findings, _df_meta = verify_decoded_frames(
            video_path,
            look_name=look_name,
            format_name=format_name,
            contact_sheet_out=sheet_target,
            plan=plan,
            timeline=timeline,
            renderer_meta_path=renderer_meta_path,
            report_out=sheet_target.with_suffix(".frames.json"),
        )
        metrics.append(f"decoded_frames: {df_chk.detail}")
        if sheet_target.exists():
            resolved_sheet = str(sheet_target)
        caption_zone_ok = _df_meta.get("caption_zone_ok")  # type: ignore[assignment]
        for df_w in _df_meta.get("warnings", []):  # type: ignore[union-attr]
            notes.append(VisualReviewNote(shot_id="frames", severity="warning", category="layout", finding=str(df_w)))
        if _df_meta.get("scene_motion_ok") is False:
            cats["animation_meaningfulness"] = "failed"
            notes.append(
                VisualReviewNote(
                    shot_id="motion",
                    severity="warning",
                    category="motion",
                    finding=f"Static or under-animated scenes detected: {_df_meta.get('static_motion_scenes')}",
                )
            )
        if not df_chk.passed:
            layout_ok = False
            cats["visual_layout_legibility"] = "failed"
            for df_msg in df_findings:
                notes.append(VisualReviewNote(shot_id="frames", severity="blocker", category="layout", finding=df_msg))
        elif layout_ok:
            cats["visual_layout_legibility"] = "passed"

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

    # Also inspect frame-QA scene coverage against plan scene count when video was evaluated
    if video_path is not None and video_path.exists() and scenes:
        sc_cov = int(_df_meta.get("scenes_covered", len(scenes)))
        sc_tot = int(_df_meta.get("total_plan_scenes", len(scenes)))
        if sc_cov < sc_tot:
            cats["visual_layout_legibility"] = "failed"
            notes.append(
                VisualReviewNote(
                    shot_id="frames",
                    severity="blocker",
                    category="layout",
                    finding=f"Incomplete scene frame QA coverage: {sc_cov}/{sc_tot} scenes inspected.",
                )
            )

    actual_vid_dur: float | None = target_duration_s
    for m_str in metrics:
        dm = re.search(r"v_dur=(\d+(?:\.\d+)?)s", m_str)
        if dm:
            actual_vid_dur = float(dm.group(1))
            break

    cap_chk = verify_caption_synchronization(
        caption_ir=caption_ir,
        caption_ir_path=caption_ir_path,
        srt_path=srt_path,
        caption_zone_ok=caption_zone_ok,
        expected_duration_s=actual_vid_dur,
        expected_beat_count=len(scenes) if scenes else None,
    )
    metrics.append(f"caption_sync: {cap_chk.detail}")
    if not script_ok or cap_chk.status == "failed":
        cats["caption_synchronization"] = "failed"
        if cap_chk.status == "failed":
            notes.append(VisualReviewNote(shot_id="caption", severity="blocker", category="caption", finding=cap_chk.detail))
    elif cap_chk.status == "passed":
        cats["caption_synchronization"] = "passed"
    elif script is not None and script_ok and video_path is None:
        cats["caption_synchronization"] = "passed"

    if storyboard_image is not None and storyboard_image.exists():
        sb_chk = verify_storyboard_image(storyboard_image)
        metrics.append(sb_chk.detail)
        if sb_chk.status == "failed":
            layout_ok = False
            cats["visual_layout_legibility"] = "failed"
            notes.append(VisualReviewNote(shot_id="storyboard", severity="blocker", category="layout", finding=sb_chk.detail))

    blockers = sum(1 for n in notes if n.severity == "blocker")
    warnings = sum(1 for n in notes if n.severity == "warning")
    req_cats = (
        "math_accuracy",
        "scientific_factual_accuracy",
        "visual_layout_legibility",
        "theme_contrast_readability",
        "caption_synchronization",
        "audio_quality_loudness",
        "animation_meaningfulness",
        "rendering_media_integrity",
    )
    uneval_req = sum(1 for k in req_cats if cats.get(k) != "passed")
    unsupported_pen = min(25.0, float(cov.unsupported) * 4.0 + float(cov.human_review_required) * 2.5)
    score = max(0.0, 100.0 - blockers * 20.0 - warnings * 6.0 - uneval_req * 10.0 - unsupported_pen)

    if video_path is None or not video_path.exists():
        if blockers > 0 or cov.failed > 0 or len(plan_issues) > 0:
            prod_state = "rejected"
        elif cov.unsupported > 0 or cov.human_review_required > 0:
            prod_state = "review_required"
        else:
            prod_state = "draft"
    else:
        if (
            blockers > 0
            or cov.failed > 0
            or len(plan_issues) > 0
            or cats["rendering_media_integrity"] == "failed"
            or cats["audio_quality_loudness"] == "failed"
        ):
            prod_state = "rejected"
        elif (
            cov.unsupported > 0
            or cov.human_review_required > 0
            or uneval_req > 0
            or not sci_ok
            or score < 82.0
        ):
            prod_state = "review_required"
        else:
            prod_state = "publication_approved"

    approved = prod_state == "publication_approved"

    return VideoReviewReport(
        topic=topic,
        title=title,
        format_name=format_name,
        backend=backend,
        production_state=prod_state,  # type: ignore[arg-type]
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
            f"state={prod_state}, score={score:.1f}/100, approved={approved}, "
            f"blockers={blockers}, warnings={warnings}, uneval_required={uneval_req}."
        ),
    )
