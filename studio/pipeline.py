"""End-to-end pipeline: (Gemini plan) -> TTS beat sync -> SFX/BGM -> ManimGX render -> layout QA -> master.

Examples:
    # Fully autonomous: field or topic -> Gemini concept + beats -> mastered Short
    python -m studio.pipeline --topic "Why 0.999... equals 1" --format shorts_hd --ai
    python -m studio.pipeline --topic "Chaos Theory" --look cinematic
    # Explicit concept
    python -m studio.pipeline --prompt "Fourier series sharpening into a square wave"
    # Re-render a saved plan, or a hand-written scene
    python -m studio.pipeline --plan plans/zero_point_nine.json
    python -m studio.pipeline scenes/taylor_series.py --format widescreen_hd
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from studio.audio.bgm import generate_ambient_bgm
from studio.audio.master import LOOKS, master_video
from studio.audio.sfx import SfxLibrary
from studio.audio.tts import build_voice_timeline
from studio.config import FORMATS, THEMES
from studio.scene import Story3DScene, StoryScene, _StoryMixin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_SCENE_SCRIPT = PROJECT_ROOT / "scenes" / "ai_short.py"


def _load_scene_class(script_path: Path, scene_name: str | None) -> tuple[str, type[_StoryMixin]]:
    spec = importlib.util.spec_from_file_location("target_scene_module", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidates = {
        name: obj
        for name, obj in vars(module).items()
        if isinstance(obj, type)
        and issubclass(obj, _StoryMixin)
        and obj not in (_StoryMixin, StoryScene, Story3DScene)
        and obj.__module__ == module.__name__
    }
    if not candidates:
        raise ValueError(f"No StoryScene or Story3DScene subclasses found in {script_path}")
    if scene_name:
        if scene_name not in candidates:
            raise ValueError(f"Scene '{scene_name}' not found. Available: {list(candidates)}")
        return scene_name, candidates[scene_name]
    if len(candidates) > 1:
        raise ValueError(f"Multiple StoryScene classes found; specify one: {list(candidates)}")
    return next(iter(candidates.items()))


def run_pipeline(
    script_path: Path,
    scene_name: str | None = None,
    format_name: str = "shorts_hd",
    theme_name: str = "midnight_studio",
    voice: str = "en-US-AndrewNeural",
    rate: str = "-4%",
    pitch: str = "-2Hz",
    tts_engine: str = "kokoro",
    strict_layout: bool = True,
    look: str = "none",
    ui: str = "minimal",
    lut_path: Path | None = None,
    output_stem: str | None = None,
) -> Path:
    if format_name not in FORMATS:
        raise ValueError(f"Unknown format '{format_name}'. Options: {list(FORMATS)}")
    if theme_name not in THEMES:
        raise ValueError(f"Unknown theme '{theme_name}'. Options: {list(THEMES)}")

    profile = FORMATS[format_name]
    script_path = script_path.resolve()
    resolved_name, scene_cls = _load_scene_class(script_path, scene_name)
    stem = output_stem or resolved_name

    build_dir = PROJECT_ROOT / "build" / f"{stem}_{format_name}"
    build_dir.mkdir(parents=True, exist_ok=True)
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    from studio.storage.cache import JobStore
    from studio.timeline import MasterTimeline
    from studio.qa.engine import verify_audio_loudness

    jobs = JobStore(PROJECT_ROOT / "build" / "studio_jobs.db")
    job_id = f"{stem}_{format_name}"
    jobs.upsert_job(job_id, stem, "tts", "running")

    print(f"[1/5] Synthesizing beat-synced TTS ({len(scene_cls.BEATS)} beats, engine={tts_engine}, rate={rate})...")
    timeline = build_voice_timeline(
        scene_cls.BEATS, build_dir=build_dir, voice=voice, rate=rate, pitch=pitch, engine=tts_engine
    )
    mt = MasterTimeline(audio=timeline)
    tl_issues = mt.validate()
    if tl_issues:
        raise RuntimeError(f"Timeline validation failed: {tl_issues}")
    mt.export_subtitles(build_dir, stem)
    jobs.upsert_job(job_id, stem, "tts", "done", {"duration_s": timeline.total_duration})
    print(f"      -> Timeline duration: {timeline.total_duration:.2f}s")

    print("[2/5] Generating studio SFX library & duration-matched ambient BGM...")
    SfxLibrary(build_dir / "sfx").ensure_all()
    bgm_wav = generate_ambient_bgm(build_dir / "bgm.wav", duration=timeline.total_duration + 3.0)

    raw_mp4 = build_dir / f"{stem}_raw.mp4"
    print(f"[3/5] Rendering '{resolved_name}' via ManimGX GPU ({profile.resolution_str} @ {profile.fps}fps)...")

    env = os.environ.copy()
    env["STUDIO_FORMAT"] = format_name
    env["STUDIO_THEME"] = theme_name
    env["STUDIO_UI"] = ui
    env["STUDIO_BUILD_DIR"] = str(build_dir)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, "-m", "manimgx", "render", str(script_path), resolved_name,
        "-r", profile.resolution_str, "--fps", str(profile.fps), "-o", str(raw_mp4),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"ManimGX render failed with exit code {proc.returncode}")

    print("[4/5] Verifying automated layout diagnostics...")
    layout_match = re.search(r"layout:\s*(.+)", proc.stdout)
    layout_summary = layout_match.group(1).strip() if layout_match else "unknown"
    print(f"      -> {layout_summary}")
    layout_ok = "no problems" in layout_summary.lower() or "not checked (a 3d scene)" in layout_summary.lower()
    if strict_layout and not layout_ok:
        raise RuntimeError(f"Strict layout QA failed: {proc.stdout}")

    final_mp4 = out_dir / f"{stem}_{format_name}.mp4"
    print(f"[5/5] Mastering (sidechain ducking, look={look}{', LUT' if lut_path else ''}) -> {final_mp4.name}...")
    master_video(
        raw_video=raw_mp4,
        voice_wav=Path(timeline.master_voice_wav),
        bgm_wav=bgm_wav,
        output_mp4=final_mp4,
        look=look,
        lut_path=lut_path,
    )
    loud_chk = verify_audio_loudness(final_mp4)
    print(f"      -> QA {loud_chk.name}: {loud_chk.status} ({loud_chk.detail})")
    jobs.upsert_job(job_id, stem, "master", "done", {"output": str(final_mp4), "loudness": loud_chk.detail})
    jobs.close()
    print(f"DONE! Master video ready at: {final_mp4}")
    return final_mp4


def run_plan(plan_path: Path, **kwargs) -> Path:
    """Render a ShortPlan JSON through the template scene (3D class for surface plans)."""
    plan_path = plan_path.resolve()
    os.environ["STUDIO_PLAN"] = str(plan_path)  # read by scenes/ai_short.py at import (here and in the render)
    template = json.loads(plan_path.read_text(encoding="utf-8")).get("template")
    scene = {"surface": "AIShort3D", "polytope4d": "AIShortPolytope"}.get(template, "AIShort")
    return run_pipeline(AI_SCENE_SCRIPT, scene_name=scene, output_stem=f"AI_{plan_path.stem}", **kwargs)


def main() -> None:
    # Progress must be visible live even when output goes to a log file or pipe
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Math Shorts Studio: autonomous AI math Shorts engine")
    parser.add_argument("script", type=Path, nargs="?", help="Hand-written scene file (omit when using --topic/--prompt/--plan)")
    parser.add_argument("--scene", type=str, default=None, help="Scene class name")
    parser.add_argument("--topic", type=str, help="Math field (e.g. 'Chaos Theory') or topic; Gemini picks the hook")
    parser.add_argument("--prompt", type=str, help="Explicit concept for Gemini to script and stage")
    parser.add_argument("--ai", action="store_true", help="Use the Gemini director (implied by --topic/--prompt)")
    parser.add_argument("--plan", type=Path, help="Render an existing ShortPlan JSON (no Gemini call)")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "openrouter", "ollama", "llamacpp"], help="AI provider (default: gemini)")
    parser.add_argument("--allow-paid", action="store_true", help="Disable strict free-only model enforcement")
    parser.add_argument("--model", type=str, default=None, help="AI model override")
    parser.add_argument("--format", type=str, default="shorts_hd", choices=list(FORMATS))
    parser.add_argument("--theme", type=str, default="midnight_studio", choices=list(THEMES))
    parser.add_argument("--voice", type=str, default="en-US-AndrewNeural")
    parser.add_argument("--rate", type=str, default="-4%")
    parser.add_argument("--pitch", type=str, default="-2Hz", help="edge-tts pitch offset, e.g. -2Hz for a warmer read")
    parser.add_argument("--tts", type=str, default="kokoro", choices=["edge", "kokoro"], help="kokoro (local 82M neural TTS, falls back to edge if not installed)")
    parser.add_argument("--look", type=str, default="none", choices=list(LOOKS), help="Cinematic grade + vignette + grain")
    parser.add_argument("--ui", type=str, default="minimal", choices=["minimal", "card", "neon"], help="minimal: box-free outlined captions; card: frosted-glass pill; neon: fullwidth gradient bar with glow")
    parser.add_argument("--lut", type=Path, default=None, help="Optional .cube LUT applied before the look")
    parser.add_argument("--allow-layout-warnings", action="store_true")
    args = parser.parse_args()

    common = dict(
        format_name=args.format, theme_name=args.theme, voice=args.voice, rate=args.rate, pitch=args.pitch,
        tts_engine=args.tts, strict_layout=not args.allow_layout_warnings, look=args.look, lut_path=args.lut, ui=args.ui,
    )

    if args.topic or args.prompt or (args.ai and not args.plan):
        if not (args.topic or args.prompt):
            parser.error("--ai needs --topic or --prompt")
        from studio.ai.director import create_plan

        print(f"[0/5] {args.provider} creative director: concept -> beat plan -> validation...")
        plan_path = create_plan(
            topic=args.topic,
            prompt=args.prompt,
            model=args.model,
            provider_name=args.provider,
            free_only=not args.allow_paid,
        )
        print(f"      -> Plan saved: {plan_path}")
        run_plan(plan_path, **common)
    elif args.plan:
        run_plan(args.plan, **common)
    elif args.script:
        run_pipeline(script_path=args.script, scene_name=args.scene, **common)
    else:
        parser.error("give a scene script, --plan, or --topic/--prompt")


if __name__ == "__main__":
    main()
