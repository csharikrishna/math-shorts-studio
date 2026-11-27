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

    schema_version: str = "4.0"
    production_id: str
    topic: str
    title: str = ""
    git_commit_sha: str = "unknown"
    git_dirty: bool = False
    working_tree_fingerprint: str = ""
    approved_script_revision: int | None = None
    approved_script_sha256: str = ""
    video_plan_sha256: str = ""
    math_model_sha256: str = ""
    renderer_backend: str = "manimgx"
    renderer_config: dict[str, Any] = Field(default_factory=dict)
    renderer_dependencies: dict[str, str] = Field(default_factory=dict)
    format_name: str = "widescreen_hd"
    look_name: str = "light"
    caption_style: str = "editorial"
    font_versions: dict[str, str] = Field(default_factory=dict)
    audio_timeline_sha256: str = ""
    caption_ir_sha256: str = ""
    final_mp4_path: str
    relative_mp4_path: str = ""
    final_mp4_sha256: str
    file_size_bytes: int = 0
    duration_s: float = 0.0
    stream_info: dict[str, Any] = Field(default_factory=dict)
    production_state: ProductionState = "rendered"
    identity_verified: bool = False
    production_certified: bool = False
    qa_overall_score: float = 0.0
    qa_approved: bool = False
    qa_review_path: str = ""
    telemetry_summary: dict[str, Any] = Field(default_factory=dict)
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


def current_working_tree_state(repo_root: Path | None = None) -> tuple[bool, str]:
    """Return (git_dirty, sha256_fingerprint) capturing uncommitted tracked changes without temporary binary bloat."""
    root = repo_root or PROJECT_ROOT
    try:
        st = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(root),
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not st:
            return False, "clean"
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"],
            cwd=str(root),
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
        fp = hashlib.sha256((st + "\n").encode("utf-8") + diff).hexdigest()[:16]
        return True, fp
    except Exception:
        return False, "unavailable"


def load_font_versions(repo_root: Path | None = None) -> dict[str, str]:
    root = repo_root or PROJECT_ROOT
    out: dict[str, str] = {}
    mf = root / "assets" / "fonts" / "FONT_MANIFEST.json"
    if mf.exists():
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if isinstance(data.get("fonts"), list):
                    for item in data["fonts"]:
                        if isinstance(item, dict) and item.get("filename"):
                            out[str(item["filename"])] = str(item.get("sha256", ""))[:16]
                else:
                    for fname, meta in data.items():
                        if isinstance(meta, dict) and (fname.endswith((".ttf", ".otf", ".woff2")) or "sha256" in meta):
                            out[str(fname)] = str(meta.get("sha256", ""))[:16]
        except Exception:
            pass
    kmf = root / "assets" / "vendor" / "katex" / "KATEX_MANIFEST.json"
    if kmf.exists():
        try:
            kd = json.loads(kmf.read_text(encoding="utf-8"))
            k_ver = str(kd.get("version") or "vendored")
            k_files = kd.get("files") or {}
            if "katex.min.js" in k_files:
                out[f"katex@{k_ver}/katex.min.js"] = str(k_files["katex.min.js"])[:16]
            if "katex.min.css" in k_files:
                out[f"katex@{k_ver}/katex.min.css"] = str(k_files["katex.min.css"])[:16]
        except Exception:
            pass
    return out


def collect_renderer_dependencies() -> dict[str, str]:
    """Capture exact local runtime dependency versions deterministically."""
    import sys as _sys

    deps: dict[str, str] = {"python": _sys.version.split()[0]}
    for mod_name in ("numpy", "sympy", "pydantic", "manimgx", "kokoro"):
        try:
            mod = __import__(mod_name)
            deps[mod_name] = str(getattr(mod, "__version__", "installed"))
        except Exception:
            pass
    kmf = PROJECT_ROOT / "assets" / "vendor" / "katex" / "KATEX_MANIFEST.json"
    if kmf.exists():
        try:
            deps["katex"] = str(json.loads(kmf.read_text(encoding="utf-8")).get("version", "vendored"))
        except Exception:
            pass
    return deps


def probe_mp4_streams(mp4_path: Path) -> dict[str, Any]:
    """Read exact container duration, resolution, fps, packet count, and audio properties via ffprobe or ffmpeg fallback."""
    import re as _re
    import shutil as _shutil
    from studio.audio.tts import find_ffmpeg

    ff = find_ffmpeg()
    ffprobe = _shutil.which("ffprobe") or ""
    if not ffprobe:
        cand = Path(ff).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if cand.exists():
            ffprobe = str(cand)
    if ffprobe:
        try:
            proc = subprocess.run(
                [ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(mp4_path)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.returncode == 0:
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
                if int(vs.get("width") or 0) > 0 and v_dur > 0.0:
                    return {
                        "width": int(vs.get("width") or 0),
                        "height": int(vs.get("height") or 0),
                        "fps": round(fps, 3),
                        "nb_frames": int(vs.get("nb_frames") or round(v_dur * fps)),
                        "sample_rate": int(aus.get("sample_rate") or 0),
                        "channels": int(aus.get("channels") or 0),
                        "v_dur": round(v_dur, 4),
                        "a_dur": round(a_dur, 4),
                        "drift_s": round(abs(v_dur - a_dur), 4) if a_dur > 0 else 0.0,
                    }
        except Exception:
            pass

    # Reliable cross-platform FFmpeg fallback when ffprobe binary is absent (e.g. imageio-ffmpeg on Windows)
    try:
        info = subprocess.run([ff, "-hide_banner", "-i", str(mp4_path)], capture_output=True, text=True, timeout=25)
        err = info.stderr or ""
        wh_m = _re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})\b", err)
        fps_m = _re.search(r"(\d+(?:\.\d+)?)\s*fps\b", err)
        sr_m = _re.search(r"Audio:.*?,\s*(\d+)\s*Hz,\s*(\w+)", err)
        cont_m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
        cont_dur = (int(cont_m.group(1)) * 3600 + int(cont_m.group(2)) * 60 + float(cont_m.group(3))) if cont_m else 0.0
        width = int(wh_m.group(1)) if wh_m else 0
        height = int(wh_m.group(2)) if wh_m else 0
        fps = float(fps_m.group(1)) if fps_m else 0.0
        sr = int(sr_m.group(1)) if sr_m else 0
        ch_lbl = sr_m.group(2).lower() if sr_m else ""
        channels = 2 if ch_lbl == "stereo" else (1 if ch_lbl == "mono" else (2 if sr > 0 else 0))

        def _probe_stream_copy(sel: str) -> tuple[float, int]:
            p = subprocess.run(
                [ff, "-hide_banner", "-i", str(mp4_path), "-map", sel, "-c", "copy", "-f", "null", "-"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            times = _re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr or "")
            frames = _re.findall(r"frame=\s*(\d+)", p.stderr or "")
            dur_v = (int(times[-1][0]) * 3600 + int(times[-1][1]) * 60 + float(times[-1][2])) if times else 0.0
            n_f = int(frames[-1]) if frames else 0
            return dur_v, n_f

        v_dur, nb_frames = _probe_stream_copy("0:v:0")
        a_dur, _ = _probe_stream_copy("0:a:0") if sr > 0 else (0.0, 0)
        if v_dur <= 0.0 and cont_dur > 0.0:
            v_dur = cont_dur
        if nb_frames <= 0 and v_dur > 0.0 and fps > 0.0:
            nb_frames = int(round(v_dur * fps))
        if width <= 0 or v_dur <= 0.0:
            return {}
        return {
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "nb_frames": nb_frames,
            "sample_rate": sr,
            "channels": channels,
            "v_dur": round(v_dur, 4),
            "a_dur": round(a_dur, 4),
            "drift_s": round(abs(v_dur - a_dur), 4) if a_dur > 0 else 0.0,
        }
    except Exception:
        return {}


def get_process_rss_mb() -> float:
    """Return current process resident set size in MiB across Windows/POSIX with zero extra deps."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
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

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return round(float(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0), 2)
        else:
            import resource

            return round(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, 2)
    except Exception:
        pass
    return 0.0


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
    telemetry_summary: dict[str, Any] | None = None,
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

    prov_log = provider_provenance or []
    dirty, tree_fp = current_working_tree_state()
    try:
        rel_mp4 = mp4_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        rel_mp4 = mp4_path.name
    try:
        rev_path_str = review_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix() if review_path else ""
    except Exception:
        rev_path_str = str(review_path) if review_path else ""

    prod_state: ProductionState = "rendered"
    qa_score = 0.0
    qa_app = False
    if review_report is not None:
        prod_state = getattr(review_report, "production_state", "rendered")
        qa_score = float(getattr(review_report, "overall_score", 0.0))
        qa_app = bool(getattr(review_report, "approved", False))
    id_ok = bool(streams.get("width") and dur_s > 0.0)
    cert_ok = bool(id_ok and prod_state == "publication_approved" and qa_app and qa_score >= 82.0)

    manifest = ProductionManifest(
        production_id=prod_id,
        topic=topic,
        title=title,
        git_commit_sha=current_git_commit(),
        git_dirty=dirty,
        working_tree_fingerprint=tree_fp,
        approved_script_revision=script_rev,
        approved_script_sha256=script_sha,
        video_plan_sha256=plan_sha,
        math_model_sha256=math_sha,
        renderer_backend=backend,
        renderer_config=renderer_config or {},
        renderer_dependencies=collect_renderer_dependencies(),
        format_name=format_name,
        look_name=look_name,
        caption_style=caption_style,
        font_versions=load_font_versions(),
        audio_timeline_sha256=tl_sha,
        caption_ir_sha256=cap_sha,
        final_mp4_path=str(mp4_path.resolve()),
        relative_mp4_path=rel_mp4,
        final_mp4_sha256=mp4_sha,
        file_size_bytes=int(mp4_path.stat().st_size),
        duration_s=dur_s,
        stream_info=streams,
        production_state=prod_state,
        identity_verified=id_ok,
        production_certified=cert_ok,
        qa_overall_score=qa_score,
        qa_approved=qa_app,
        qa_review_path=rev_path_str,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        provider_provenance=prov_log,
        telemetry_summary=telemetry_summary or {},
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


def finalize_manifest_qa(
    manifest_path: Path,
    review_report: Any,
    *,
    mp4_path: Path | None = None,
    review_path: Path | None = None,
    build_dir: Path | None = None,
    telemetry_summary: dict[str, Any] | None = None,
) -> ProductionManifest:
    """Bind completed QA report & telemetry to exact verified MP4 and finalize production certification state."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    mf = ProductionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    target_mp4 = mp4_path or Path(mf.final_mp4_path)
    if not target_mp4.exists() and mf.relative_mp4_path:
        rel_cand = PROJECT_ROOT / mf.relative_mp4_path
        if rel_cand.exists():
            target_mp4 = rel_cand
    if not target_mp4.exists():
        raise FileNotFoundError(f"Cannot certify missing MP4: {target_mp4}")
    actual_sha = sha256_file(target_mp4)
    if actual_sha != mf.final_mp4_sha256:
        raise ValueError(f"Refusing to attach QA to mismatched MP4 bytes: {actual_sha[:12]} != {mf.final_mp4_sha256[:12]}")

    prod_state = getattr(review_report, "production_state", mf.production_state)
    qa_score = float(getattr(review_report, "overall_score", mf.qa_overall_score))
    qa_app = bool(getattr(review_report, "approved", mf.qa_approved))
    try:
        rev_p_str = review_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix() if review_path else mf.qa_review_path
    except Exception:
        rev_p_str = str(review_path) if review_path else mf.qa_review_path

    streams = mf.stream_info or probe_mp4_streams(target_mp4)
    dur_s = float(streams.get("v_dur") or mf.duration_s or 0.0)
    id_ok = bool(streams.get("width") and dur_s > 0.0 and mf.font_versions)
    cert_ok = bool(id_ok and prod_state == "publication_approved" and qa_app and qa_score >= 82.0)

    updated = mf.model_copy(
        update={
            "stream_info": streams,
            "duration_s": dur_s,
            "production_state": prod_state,
            "qa_overall_score": qa_score,
            "qa_approved": qa_app,
            "qa_review_path": rev_p_str,
            "identity_verified": id_ok,
            "production_certified": cert_ok,
            "telemetry_summary": telemetry_summary if telemetry_summary is not None else mf.telemetry_summary,
        }
    )
    txt = updated.model_dump_json(indent=2)
    tmp_m = manifest_path.with_suffix(".tmp")
    tmp_m.write_text(txt, encoding="utf-8")
    os.replace(tmp_m, manifest_path)
    if build_dir is not None:
        (build_dir / "production_manifest.json").write_text(txt, encoding="utf-8")
    return updated


def inspect_manifest_verification(
    manifest_path: Path,
    mp4_path: Path | None = None,
    *,
    require_certification: bool = False,
) -> dict[str, Any]:
    """Verify MP4 byte identity, mandatory stream properties, font pins, and optionally full QA certification."""
    identity_issues: list[str] = []
    cert_issues: list[str] = []
    if not manifest_path.exists():
        return {"identity_verified": False, "production_certified": False, "issues": [f"Missing manifest: {manifest_path}"], "certification_issues": []}

    mf = ProductionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    target_mp4 = mp4_path or Path(mf.final_mp4_path)
    if not target_mp4.exists() and mf.relative_mp4_path:
        cand = PROJECT_ROOT / mf.relative_mp4_path
        if cand.exists():
            target_mp4 = cand
    if not target_mp4.exists():
        return {"identity_verified": False, "production_certified": False, "issues": [f"Manifest targets missing MP4: {target_mp4}"], "certification_issues": []}

    actual_sha = sha256_file(target_mp4)
    if actual_sha != mf.final_mp4_sha256:
        identity_issues.append(f"Stale/mismatched MP4 SHA-256: manifest={mf.final_mp4_sha256[:12]} vs actual={actual_sha[:12]}")
    actual_size = int(target_mp4.stat().st_size)
    if mf.file_size_bytes and actual_size != mf.file_size_bytes:
        identity_issues.append(f"File size mismatch: manifest={mf.file_size_bytes} vs actual={actual_size}")

    if not mf.font_versions:
        identity_issues.append("Manifest missing font_versions pins")
    streams = probe_mp4_streams(target_mp4)
    enforce_streams = require_certification or bool(streams) or mf.duration_s > 0.0 or bool(mf.stream_info)
    req_stream_keys = ("width", "height", "fps", "sample_rate", "v_dur")
    missing_sk = [k for k in req_stream_keys if not mf.stream_info.get(k)]
    if enforce_streams:
        if mf.duration_s <= 0.0:
            identity_issues.append("Manifest missing measured duration_s (> 0 required)")
        if missing_sk:
            identity_issues.append(f"Manifest missing required stream_info fields: {missing_sk}")
        if not streams:
            identity_issues.append("Failed to probe live MP4 video/audio streams")
        else:
            actual_dur = float(streams.get("v_dur") or 0.0)
            if mf.duration_s > 0 and abs(actual_dur - mf.duration_s) > 0.25:
                identity_issues.append(f"Duration mismatch: manifest={mf.duration_s:.2f}s vs actual={actual_dur:.2f}s")
            for sk in ("width", "height", "sample_rate"):
                if mf.stream_info.get(sk) and int(mf.stream_info[sk]) != int(streams.get(sk) or 0):
                    identity_issues.append(f"Stream mismatch {sk}: manifest={mf.stream_info[sk]} vs actual={streams.get(sk)}")

    if mf.qa_review_path:
        rp = Path(mf.qa_review_path)
        if not rp.is_absolute():
            rp = PROJECT_ROOT / rp
        if not rp.exists():
            identity_issues.append(f"Linked review report missing: {mf.qa_review_path}")

    if mf.production_state != "publication_approved" or not mf.qa_approved or mf.qa_overall_score < 82.0:
        cert_issues.append(
            f"Not production_certified (state={mf.production_state}, qa_approved={mf.qa_approved}, score={mf.qa_overall_score})"
        )

    id_ok = len(identity_issues) == 0 and bool(streams) and mf.duration_s > 0.0 and not missing_sk and bool(mf.font_versions)
    cert_ok = id_ok and len(cert_issues) == 0
    combined = list(identity_issues) + (cert_issues if require_certification else [])
    return {
        "identity_verified": len(identity_issues) == 0,
        "stream_and_assets_complete": id_ok,
        "production_certified": cert_ok,
        "issues": combined,
        "certification_issues": cert_issues,
    }


def verify_manifest_matches_mp4(
    manifest_path: Path,
    mp4_path: Path | None = None,
    *,
    require_certification: bool = False,
) -> tuple[bool, list[str]]:
    """Independently verify that MP4 bytes, stream metadata, fonts, and QA artifacts match ProductionManifest."""
    res = inspect_manifest_verification(manifest_path, mp4_path, require_certification=require_certification)
    return (len(res["issues"]) == 0), list(res["issues"])
