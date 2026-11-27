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
    """Measure actual integrated loudness and true peak via FFmpeg ebur128."""
    ff = find_ffmpeg()
    try:
        proc = subprocess.run(
            [ff, "-hide_banner", "-nostats", "-i", str(video_path), "-af", "ebur128=peak=true", "-f", "null", "-"],
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


def audit_educational_quality(
    *,
    topic: str,
    title: str,
    plan: object,
    format_name: str,
    backend: str = "manimgx",
    video_path: Path | None = None,
    storyboard_image: Path | None = None,
    script: object | None = None,
    look_name: str = "midnight_studio",
    chapter_plans: list[object] | None = None,
) -> object:
    """Comprehensive pre/post-render quality audit returning a VideoReviewReport."""
    from studio.ai.validate_v3 import verify_v3_plan_detailed
    from studio.creative.script_ir import validate_script_ir
    from studio.storyboard.model import VerificationCoverageSummary, VideoReviewReport, VisualReviewNote

    notes: list[VisualReviewNote] = []
    metrics: list[str] = []

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
        det = verify_v3_plan_detailed(p_item, require_claims=False)  # type: ignore[arg-type]
        cov_raw = det.get("verification_coverage", {})
        if isinstance(cov_raw, dict):
            for k in cov_acc:
                cov_acc[k] += int(cov_raw.get(k, 0))
        plan_issues.extend(str(m) for m in det.get("issues", []))  # type: ignore[arg-type]

    cov = VerificationCoverageSummary(**cov_acc)
    for msg in plan_issues[:20]:
        notes.append(VisualReviewNote(shot_id="plan", severity="blocker", category="math_grounding", finding=str(msg)))

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
    metrics.append(f"scenes={len(scenes)}, unique_object_kinds={len(obj_kinds)}, compositions={sorted(compositions)}")

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

    audio_ok = True
    if video_path is not None and video_path.exists():
        loud = verify_audio_loudness(video_path)
        streams = verify_media_streams(video_path)
        metrics.append(f"loudness: {loud.detail}")
        metrics.append(f"streams: {streams.detail}")
        if not (loud.passed and streams.passed):
            audio_ok = False
            notes.append(
                VisualReviewNote(
                    shot_id="media",
                    severity="blocker",
                    category="audio",
                    finding=f"Media QA issue: {loud.detail}; {streams.detail}",
                )
            )

    if storyboard_image is not None and storyboard_image.exists():
        sb_chk = verify_storyboard_image(storyboard_image)
        metrics.append(sb_chk.detail)
        if sb_chk.status == "failed":
            layout_ok = False
            notes.append(VisualReviewNote(shot_id="storyboard", severity="blocker", category="layout", finding=sb_chk.detail))

    math_ok = len(plan_issues) == 0 and cov.failed == 0
    blockers = sum(1 for n in notes if n.severity == "blocker")
    warnings = sum(1 for n in notes if n.severity == "warning")
    score = max(0.0, 100.0 - blockers * 22.0 - warnings * 6.0)
    approved = blockers == 0 and score >= 75.0

    return VideoReviewReport(
        topic=topic,
        title=title,
        format_name=format_name,
        backend=backend,
        approved=approved,
        overall_score=score,
        script_clarity_ok=script_ok,
        math_grounding_ok=math_ok,
        layout_safe_zone_ok=layout_ok,
        audio_sync_ok=audio_ok,
        visual_variety_ok=variety_ok,
        verification_summary=cov,
        notes=notes,
        metrics=metrics,
        summary=f"Audited {len(scenes)} scenes ({look_name}, {format_name}, {backend}): score={score:.1f}/100, blockers={blockers}, warnings={warnings}.",
    )
