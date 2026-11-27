"""Immutable per-render ProductionManifest creation and cryptographic artifact verification."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from studio.storyboard.model import ProductionState

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ProductionManifest(BaseModel):
    """Cryptographically bound manifest tying a single rendered MP4 to exact code, script, math, and QA."""

    schema_version: str = "3.9"
    production_id: str
    topic: str
    title: str = ""
    git_commit_sha: str = "unknown"
    approved_script_revision: int | None = None
    approved_script_sha256: str = ""
    video_plan_sha256: str = ""
    math_model_sha256: str = ""
    renderer_backend: str = "manimgx"
    renderer_config: dict[str, Any] = Field(default_factory=dict)
    format_name: str = "widescreen_hd"
    look_name: str = "light"
    caption_style: str = "editorial"
    font_versions: dict[str, str] = Field(default_factory=dict)
    audio_timeline_sha256: str = ""
    caption_ir_sha256: str = ""
    final_mp4_path: str
    final_mp4_sha256: str
    file_size_bytes: int = 0
    duration_s: float = 0.0
    stream_info: dict[str, Any] = Field(default_factory=dict)
    production_state: ProductionState = "rendered"
    qa_overall_score: float = 0.0
    qa_approved: bool = False
    qa_review_path: str = ""
    generated_at_utc: str = ""
    provider_provenance: list[dict[str, Any]] = Field(default_factory=list)


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of an existing file cleanly in streaming 64 KiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    """Compute canonical deterministic SHA-256 of a JSON-serializable structure."""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def current_git_commit(repo_root: Path | None = None) -> str:
    root = repo_root or PROJECT_ROOT
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True, timeout=5, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


def load_font_versions(repo_root: Path | None = None) -> dict[str, str]:
    root = repo_root or PROJECT_ROOT
    mf = root / "assets" / "fonts" / "FONT_MANIFEST.json"
    if not mf.exists():
        return {}
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
        out: dict[str, str] = {}
        for item in data.get("fonts", []) if isinstance(data, dict) else []:
            if isinstance(item, dict) and item.get("filename"):
                out[str(item["filename"])] = str(item.get("sha256", ""))[:16]
        return out
    except Exception:
        return {}


def probe_mp4_streams(mp4_path: Path) -> dict[str, Any]:
    """Read exact container duration, resolution, fps, and audio properties via ffprobe."""
    from studio.audio.tts import find_ffmpeg

    ff = find_ffmpeg()
    ffprobe = str(Path(ff).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe"))
    if not Path(ffprobe).exists():
        ffprobe = "ffprobe"
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(mp4_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode != 0:
            return {}
        raw = json.loads(proc.stdout)
        streams = raw.get("streams", [])
        vs = next((s for s in streams if s.get("codec_type") == "video"), {})
        aus = next((s for s in streams if s.get("codec_type") == "audio"), {})
        fps_str = str(vs.get("r_frame_rate") or "0/1")
        fps = 0.0
        if "/" in fps_str:
            num, den = fps_str.split("/", 1)
            fps = float(num) / max(1.0, float(den))
        v_dur = float(vs.get("duration") or raw.get("format", {}).get("duration") or 0.0)
        a_dur = float(aus.get("duration") or 0.0)
        return {
            "width": int(vs.get("width") or 0),
            "height": int(vs.get("height") or 0),
            "fps": round(fps, 3),
            "nb_frames": int(vs.get("nb_frames") or 0),
            "sample_rate": int(aus.get("sample_rate") or 0),
            "channels": int(aus.get("channels") or 0),
            "v_dur": round(v_dur, 4),
            "a_dur": round(a_dur, 4),
            "drift_s": round(abs(v_dur - a_dur), 4) if a_dur > 0 else 0.0,
        }
    except Exception:
        return {}


def build_and_write_manifest(
    *,
    slug: str,
    topic: str,
    title: str,
    mp4_path: Path,
    plan: Any,
    backend: str,
    format_name: str,
    look_name: str,
    caption_style: str,
    build_dir: Path | None = None,
    script_ir: Any | None = None,
    audio_timeline: Any | None = None,
    timeline_path: Path | None = None,
    caption_ir_path: Path | None = None,
    review_report: Any | None = None,
    review_path: Path | None = None,
    renderer_config: dict[str, Any] | None = None,
    provider_provenance: list[dict[str, Any]] | None = None,
) -> tuple[ProductionManifest, Path]:
    """Construct immutable ProductionManifest from exact rendered bytes and write sidecar JSON files."""
    if not mp4_path.exists() or mp4_path.stat().st_size == 0:
        raise FileNotFoundError(f"Cannot build manifest for missing/empty MP4: {mp4_path}")

    mp4_sha = sha256_file(mp4_path)
    now_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prod_id = f"{slug}_{now_utc}_{mp4_sha[:10]}"

    plan_dump = plan.model_dump(mode="json") if hasattr(plan, "model_dump") else (plan or {})
    plan_sha = sha256_json(plan_dump)
    math_payload = {
        "math_models": plan_dump.get("math_models", []) if isinstance(plan_dump, dict) else [],
        "constructions": [sc.get("constructions", []) for sc in (plan_dump.get("scenes", []) if isinstance(plan_dump, dict) else [])],
        "claims": [sc.get("claims", []) for sc in (plan_dump.get("scenes", []) if isinstance(plan_dump, dict) else [])],
    }
    math_sha = sha256_json(math_payload)

    script_rev = getattr(script_ir, "approved_revision", None) if script_ir is not None else None
    script_sha = sha256_json(script_ir.model_dump(mode="json")) if script_ir is not None and hasattr(script_ir, "model_dump") else ""

    if timeline_path is not None and timeline_path.exists():
        tl_sha = sha256_file(timeline_path)
    elif audio_timeline is not None:
        tl_sha = sha256_json(audio_timeline)
    else:
        tl_sha = ""

    cap_sha = sha256_file(caption_ir_path) if (caption_ir_path is not None and caption_ir_path.exists()) else ""
    streams = probe_mp4_streams(mp4_path)
    dur_s = float(streams.get("v_dur") or 0.0)

    prod_state: ProductionState = "rendered"
    qa_score = 0.0
    qa_app = False
    if review_report is not None:
        prod_state = getattr(review_report, "production_state", "rendered")
        qa_score = float(getattr(review_report, "overall_score", 0.0))
        qa_app = bool(getattr(review_report, "approved", False))

    manifest = ProductionManifest(
        production_id=prod_id,
        topic=topic,
        title=title,
        git_commit_sha=current_git_commit(),
        approved_script_revision=script_rev,
        approved_script_sha256=script_sha,
        video_plan_sha256=plan_sha,
        math_model_sha256=math_sha,
        renderer_backend=backend,
        renderer_config=renderer_config or {},
        format_name=format_name,
        look_name=look_name,
        caption_style=caption_style,
        font_versions=load_font_versions(),
        audio_timeline_sha256=tl_sha,
        caption_ir_sha256=cap_sha,
        final_mp4_path=str(mp4_path.resolve()),
        final_mp4_sha256=mp4_sha,
        file_size_bytes=int(mp4_path.stat().st_size),
        duration_s=dur_s,
        stream_info=streams,
        production_state=prod_state,
        qa_overall_score=qa_score,
        qa_approved=qa_app,
        qa_review_path=str(review_path.resolve()) if review_path and review_path.exists() else (str(review_path) if review_path else ""),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        provider_provenance=provider_provenance or [],
    )

    payload_text = manifest.model_dump_json(indent=2)
    out_manifest = mp4_path.with_suffix(".manifest.json")
    tmp_m = out_manifest.with_suffix(".tmp")
    tmp_m.write_text(payload_text, encoding="utf-8")
    os.replace(tmp_m, out_manifest)

    if build_dir is not None:
        build_dir.mkdir(parents=True, exist_ok=True)
        b_manifest = build_dir / "production_manifest.json"
        b_manifest.write_text(payload_text, encoding="utf-8")
    return manifest, out_manifest


def verify_manifest_matches_mp4(manifest_path: Path, mp4_path: Path | None = None) -> tuple[bool, list[str]]:
    """Independently verify that MP4 bytes, stream duration, and QA artifacts match ProductionManifest."""
    issues: list[str] = []
    if not manifest_path.exists():
        return False, [f"Missing manifest: {manifest_path}"]
    mf = ProductionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    target_mp4 = mp4_path or Path(mf.final_mp4_path)
    if not target_mp4.exists():
        return False, [f"Manifest targets missing MP4: {target_mp4}"]
    actual_sha = sha256_file(target_mp4)
    if actual_sha != mf.final_mp4_sha256:
        issues.append(f"Stale/mismatched MP4 SHA-256: manifest={mf.final_mp4_sha256[:12]} vs actual={actual_sha[:12]}")
    actual_size = int(target_mp4.stat().st_size)
    if mf.file_size_bytes and actual_size != mf.file_size_bytes:
        issues.append(f"File size mismatch: manifest={mf.file_size_bytes} vs actual={actual_size}")
    streams = probe_mp4_streams(target_mp4)
    if streams and mf.duration_s > 0:
        actual_dur = float(streams.get("v_dur") or 0.0)
        if abs(actual_dur - mf.duration_s) > 0.25:
            issues.append(f"Duration mismatch: manifest={mf.duration_s:.2f}s vs actual={actual_dur:.2f}s")
    if mf.qa_review_path and not Path(mf.qa_review_path).exists():
        issues.append(f"Linked review report missing: {mf.qa_review_path}")
    return len(issues) == 0, issues
