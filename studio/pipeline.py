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
import threading
from studio.audio.bgm import generate_ambient_bgm
from studio.audio.master import LOOKS, master_video
from studio.audio.sfx import SfxLibrary
from studio.audio.tts import build_voice_timeline
from studio.config import CAPTION_STYLE_CHOICES, FORMATS, THEMES
from studio.design.spec import resolve_look
from studio.scene import Story3DScene, StoryScene, _StoryMixin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_SCENE_SCRIPT = PROJECT_ROOT / "scenes" / "ai_short.py"
_V3_SCENE_ENV_LOCK = threading.Lock()


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
    allow_edge_fallback: bool = True,
    strict_layout: bool = True,
    look: str = "none",
    ui: str = "minimal",
    caption_style: str | None = None,
    lut_path: Path | None = None,
    output_stem: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> Path:
    if format_name not in FORMATS:
        raise ValueError(f"Unknown format '{format_name}'. Options: {list(FORMATS)}")
    theme_name = resolve_look(theme_name).name
    if theme_name not in THEMES:
        raise ValueError(f"Unknown theme '{theme_name}'. Options: {list(THEMES)}")

    profile = FORMATS[format_name]
    resolved_cap_style = caption_style or ("editorial" if not profile.is_vertical else "minimal")
    script_path = script_path.resolve()
    prev_env_vals: dict[str, str | None] = {}
    with _V3_SCENE_ENV_LOCK:
        if extra_env:
            for ek, ev in extra_env.items():
                prev_env_vals[ek] = os.environ.get(ek)
                os.environ[ek] = ev
        try:
            resolved_name, scene_cls = _load_scene_class(script_path, scene_name)
        finally:
            for ek, old_v in prev_env_vals.items():
                if old_v is None:
                    os.environ.pop(ek, None)
                else:
                    os.environ[ek] = old_v
    stem = output_stem or resolved_name

    build_dir = PROJECT_ROOT / "build" / f"{stem}_{format_name}"
    build_dir.mkdir(parents=True, exist_ok=True)
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    from studio.storage.cache import JobStore
    from studio.timeline import MasterTimeline
    from studio.qa.engine import verify_audio_loudness, verify_media_streams

    jobs = JobStore(PROJECT_ROOT / "build" / "studio_jobs.db")
    job_id = f"{stem}_{format_name}"
    jobs.upsert_job(job_id, stem, "tts", "running")

    print(f"[1/5] Synthesizing beat-synced TTS ({len(scene_cls.BEATS)} beats, engine={tts_engine}, rate={rate})...")
    from studio.creative.dialogue import resolve_speaker_voices

    sp_voices = resolve_speaker_voices(voice_override=voice if voice != "en-US-AndrewNeural" else None)
    timeline = build_voice_timeline(
        scene_cls.BEATS,
        build_dir=build_dir,
        voice=voice,
        rate=rate,
        pitch=pitch,
        engine=tts_engine,
        allow_edge_fallback=allow_edge_fallback,
        speaker_voices=sp_voices,
    )
    mt = MasterTimeline(audio=timeline)
    tl_issues = mt.validate()
    if tl_issues:
        raise RuntimeError(f"Timeline validation failed: {tl_issues}")
    mt.export_subtitles(build_dir, stem, caption_style=resolved_cap_style, format_name=format_name, look_name=theme_name)
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
    env["STUDIO_CAPTION_STYLE"] = resolved_cap_style
    env["STUDIO_BUILD_DIR"] = str(build_dir)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable, "-m", "manimgx", "render", str(script_path), resolved_name,
        "-r", profile.resolution_str, "--fps", str(profile.fps),
        "--preset", "ultrafast", "--crf", "12",
        "-o", str(raw_mp4),
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

    voice_wav_path = Path(timeline.master_voice_wav)
    v3_env = (extra_env or {}).get("STUDIO_V3_PLAN") or os.environ.get("STUDIO_V3_PLAN")
    if v3_env and Path(v3_env).exists():
        try:
            from studio.ai.v3_schema import VideoPlanV3
            from studio.audio.demo_signals import apply_plan_audio_demos

            p3_obj = VideoPlanV3.model_validate_json(Path(v3_env).read_text(encoding="utf-8"))
            voice_wav_path = apply_plan_audio_demos(p3_obj, timeline, build_dir)
        except Exception as exc:
            print(f"      (audio demo synthesis skipped: {exc})")

    final_mp4 = out_dir / f"{stem}_{format_name}.mp4"
    print(f"[5/5] Mastering (sidechain ducking, look={look}{', LUT' if lut_path else ''}) -> {final_mp4.name}...")
    master_video(
        raw_video=raw_mp4,
        voice_wav=voice_wav_path,
        bgm_wav=bgm_wav,
        output_mp4=final_mp4,
        look=look,
        lut_path=lut_path,
    )
    loud_chk = verify_audio_loudness(final_mp4)
    stream_chk = verify_media_streams(final_mp4)
    print(f"      -> QA {loud_chk.name}: {loud_chk.status} ({loud_chk.detail})")
    print(f"      -> QA {stream_chk.name}: {stream_chk.status} ({stream_chk.detail})")
    if not loud_chk.passed or not stream_chk.passed:
        jobs.upsert_job(job_id, stem, "qa", "failed", {"loudness": loud_chk.detail, "streams": stream_chk.detail})
        jobs.close()
        raise RuntimeError(f"Audio/Media QA failed: {loud_chk.detail}; {stream_chk.detail}")
    jobs.upsert_job(job_id, stem, "master", "done", {"output": str(final_mp4), "loudness": loud_chk.detail})
    jobs.close()
    print(f"DONE! Master video ready at: {final_mp4}")
    return final_mp4


V2_SCENE_SCRIPT = PROJECT_ROOT / "scenes" / "multi_scene.py"
V3_SCENE_SCRIPT = PROJECT_ROOT / "scenes" / "v3_scene.py"


def run_plan(plan_path: Path, **kwargs) -> Path:
    """Schema-version-aware plan router: V3 -> V3Scene, V2/2.1 -> MultiSceneVideo, V1 -> AIShort."""
    plan_path = plan_path.resolve()
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    # Clear stale plan env vars before routing
    os.environ.pop("STUDIO_PLAN", None)
    os.environ.pop("STUDIO_V2_PLAN", None)
    os.environ.pop("STUDIO_V3_PLAN", None)
    try:
        if raw.get("schema_version") == "3.0":
            from studio.ai.v3_schema import VideoPlanV3
            from studio.ai.validate_v3 import validate_v3_plan

            plan3 = VideoPlanV3.model_validate(raw)
            issues = validate_v3_plan(plan3)
            if issues:
                raise ValueError("Invalid VideoPlanV3:\n- " + "\n- ".join(issues))
            stem = kwargs.pop("output_stem", f"V3_{plan_path.stem}")
            mp4 = run_pipeline(
                V3_SCENE_SCRIPT,
                scene_name="V3Scene",
                output_stem=stem,
                extra_env={"STUDIO_V3_PLAN": str(plan_path)},
                **kwargs,
            )
            fmt_p = str(kwargs.get("format_name", "shorts_hd"))
            thm_p = str(kwargs.get("theme_name", "midnight_studio"))
            cap_p = str(kwargs.get("caption_style") or "editorial")
            bdir_p = PROJECT_ROOT / "build" / f"{stem}_{fmt_p}"
            from studio.provenance import build_and_write_manifest

            build_and_write_manifest(
                slug=plan_path.stem,
                topic=plan3.topic,
                title=plan3.title,
                mp4_path=mp4,
                plan=plan3,
                backend="manimgx",
                format_name=fmt_p,
                look_name=thm_p,
                caption_style=cap_p,
                build_dir=bdir_p if bdir_p.exists() else None,
                timeline_path=(bdir_p / "timeline.json") if (bdir_p / "timeline.json").exists() else None,
                caption_ir_path=(bdir_p / "caption_ir.json") if (bdir_p / "caption_ir.json").exists() else None,
            )
            return mp4

        if raw.get("schema_version") in ("2.0", "2.1") or "scenes" in raw:
            from studio.ai.multi_schema import VideoPlan
            from studio.ai.validate_v2 import validate_video_plan

            plan = VideoPlan.model_validate(raw)
            issues = validate_video_plan(plan)
            if issues:
                raise ValueError("Invalid VideoPlan V2:\n- " + "\n- ".join(issues))
            os.environ["STUDIO_V2_PLAN"] = str(plan_path)
            return run_pipeline(V2_SCENE_SCRIPT, scene_name="MultiSceneVideo", output_stem=f"V2_{plan_path.stem}", **kwargs)

        os.environ["STUDIO_PLAN"] = str(plan_path)
        template = raw.get("template")
        scene = {"surface": "AIShort3D", "polytope4d": "AIShortPolytope"}.get(template, "AIShort")
        return run_pipeline(AI_SCENE_SCRIPT, scene_name=scene, output_stem=f"AI_{plan_path.stem}", **kwargs)
    finally:
        os.environ.pop("STUDIO_PLAN", None)
        os.environ.pop("STUDIO_V2_PLAN", None)
        os.environ.pop("STUDIO_V3_PLAN", None)


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
    parser.add_argument("--engine", type=str, default="v3", choices=["v3", "v2", "legacy"], help="v3: composable object-event VideoPlanV3 (default); v2: VideoPlan; legacy: ShortPlan")
    parser.add_argument("--allow-paid", action="store_true", help="Disable strict free-only model enforcement")
    parser.add_argument("--model", type=str, default=None, help="AI model override")
    parser.add_argument("--format", type=str, default="shorts_hd", choices=list(FORMATS))
    parser.add_argument("--theme", type=str, default="midnight_studio", choices=[*list(THEMES), "editorial_light", "cinematic_dark"])
    parser.add_argument("--voice", type=str, default="en-US-AndrewNeural")
    parser.add_argument("--rate", type=str, default="-4%")
    parser.add_argument("--pitch", type=str, default="-2Hz", help="edge-tts pitch offset, e.g. -2Hz for a warmer read")
    parser.add_argument("--tts", type=str, default="kokoro", choices=["edge", "kokoro"], help="kokoro (local 82M neural TTS, falls back to edge if not installed)")
    parser.add_argument("--no-edge-fallback", action="store_true", help="Fail instead of falling back to network edge-tts if kokoro fails")
    parser.add_argument("--longform", action="store_true", help="Generate autonomous multi-chapter long-form video from --topic")
    parser.add_argument("--duration", type=float, default=300.0, help="Target duration in seconds for --longform (e.g. 300, 600, 1200)")
    parser.add_argument("--audience", type=str, default="general", choices=["school", "beginner", "undergraduate", "advanced", "general"])
    parser.add_argument("--look", type=str, default="none", choices=list(LOOKS), help="Cinematic grade + vignette + grain")
    parser.add_argument("--backend", type=str, default="manimgx", choices=["manimgx", "hyperframes", "hybrid"], help="Video render backend (default: manimgx GPU; hybrid routes per scene)")
    parser.add_argument(
        "--narration-mode",
        type=str,
        default="single_educator",
        choices=["single_educator", "conversational_explainer", "socratic", "discussion", "documentary", "technical_instructor"],
        help="Pedagogical narration or multi-speaker dialogue style",
    )
    parser.add_argument("--derive-short", action="store_true", help="Derive a native recomposed 9:16 Short when running --longform")
    parser.add_argument("--benchmark-backends", action="store_true", help="Render both ManimGX and HyperFrames for --plan comparison")
    parser.add_argument("--no-creative-director", action="store_true", help="Skip separate CreativeDirector brief/script stages")
    parser.add_argument("--ui", type=str, default="minimal", choices=["minimal", "card", "neon"], help="Legacy UI style preset")
    parser.add_argument(
        "--caption-style",
        type=str,
        default=None,
        choices=CAPTION_STYLE_CHOICES,
        help="Unified caption style preset (default: editorial for widescreen, minimal/kinetic for Shorts)",
    )
    parser.add_argument("--lut", type=Path, default=None, help="Optional .cube LUT applied before the look")
    parser.add_argument("--allow-layout-warnings", action="store_true")
    parser.add_argument("--output-stem", type=str, default=None, help="Optional custom output file stem")
    args = parser.parse_args()

    theme_resolved = resolve_look(args.theme).name
    common = dict(
        format_name=args.format,
        theme_name=theme_resolved,
        voice=args.voice,
        rate=args.rate,
        pitch=args.pitch,
        tts_engine=args.tts,
        allow_edge_fallback=not args.no_edge_fallback,
        strict_layout=not args.allow_layout_warnings,
        look=args.look,
        lut_path=args.lut,
        ui=args.ui,
        caption_style=args.caption_style,
        output_stem=args.output_stem,
    )

    if args.benchmark_backends and args.plan:
        from studio.ai.v3_schema import VideoPlanV3
        from studio.ai.world_model import materialize_constructions
        from studio.renderers import benchmark_backends

        p3 = materialize_constructions(VideoPlanV3.model_validate_json(args.plan.read_text(encoding="utf-8")))
        res_list = benchmark_backends(p3, slug=args.plan.stem, format_name=args.format, look_name=args.theme, out_dir=PROJECT_ROOT / "output")
        for r in res_list:
            print(f"[benchmark] backend={r.backend} time={r.render_time_s}s size={r.file_size_bytes}B rss={r.peak_rss_mb}MB dur={r.duration_s}s out={r.video_path}")
        return

    if args.longform:
        if not (args.topic or args.prompt):
            parser.error("--longform needs --topic or --prompt")
        from studio.ai.director import slugify
        from studio.longform import build_longform_video, create_longform_plan, derive_recomposed_short

        topic_str = args.topic or args.prompt or "math_documentary"
        lf_mode = "documentary" if args.narration_mode == "single_educator" else args.narration_mode
        lf_plan, ch_plans = create_longform_plan(
            topic_str,
            target_duration_s=args.duration,
            audience=args.audience,
            narration_mode=lf_mode,
            model=args.model,
            provider_name=args.provider,
            free_only=not args.allow_paid,
            on_progress=lambda m: print(f"[longform] {m}", flush=True),
        )
        res = build_longform_video(
            slugify(topic_str),
            lf_plan,
            ch_plans,
            format_name="widescreen_hd" if args.format == "shorts_hd" else args.format,
            tts_engine=args.tts,
            allow_edge_fallback=not args.no_edge_fallback,
            voice=args.voice,
            rate=args.rate,
            pitch=args.pitch,
            ui=args.ui,
            look=args.look,
            look_name=theme_resolved,
            backend=args.backend,
            caption_style=args.caption_style,
            on_progress=lambda m: print(f"[longform] {m}", flush=True),
        )
        print(f"DONE! Long-form video ready: {res['output_mp4']} ({res['total_duration_s']}s)")
        if args.derive_short:
            s_res = derive_recomposed_short(
                slugify(topic_str),
                lf_plan,
                ch_plans,
                format_name="shorts_hd",
                tts_engine=args.tts,
                allow_edge_fallback=not args.no_edge_fallback,
                voice=args.voice,
                rate=args.rate,
                pitch=args.pitch,
                ui=args.ui,
                look=args.look,
                look_name=theme_resolved,
                backend=args.backend,
                caption_style=args.caption_style,
            )
            print(f"DONE! Derived portrait Short ready: {s_res['output_mp4']} ({s_res['duration_s']}s)")
    elif args.topic or args.prompt or (args.ai and not args.plan):
        if not (args.topic or args.prompt):
            parser.error("--ai needs --topic or --prompt")
        from studio.ai.director import create_plan, create_v2_plan, create_v3_plan

        print(f"[0/5] {args.provider} creative director ({args.engine}): concept -> plan -> validation...")
        if args.engine == "v3":
            plan_path = create_v3_plan(
                topic=args.topic,
                prompt=args.prompt,
                model=args.model,
                provider_name=args.provider,
                free_only=not args.allow_paid,
                audience=args.audience,
                narration_mode=args.narration_mode,
                look_name=args.theme,
                format_name=args.format,
                backend=args.backend,
                use_creative_director=not args.no_creative_director,
            )
        elif args.engine == "v2":
            plan_path = create_v2_plan(
                topic=args.topic,
                prompt=args.prompt,
                model=args.model,
                provider_name=args.provider,
                free_only=not args.allow_paid,
            )
        else:
            plan_path = create_plan(
                topic=args.topic,
                prompt=args.prompt,
                model=args.model,
                provider_name=args.provider,
                free_only=not args.allow_paid,
            )
        print(f"      -> Plan saved: {plan_path}")
        if args.backend in ("hyperframes", "hybrid") and args.engine == "v3":
            from studio.ai.v3_schema import VideoPlanV3
            from studio.ai.world_model import materialize_constructions
            from studio.renderers import get_backend

            p3 = materialize_constructions(VideoPlanV3.model_validate_json(plan_path.read_text(encoding="utf-8")))
            r = get_backend(args.backend).render(
                p3,
                slug=args.output_stem or plan_path.stem,
                format_name=args.format,
                look_name=theme_resolved,
                caption_style=args.caption_style,
                tts_engine=args.tts,
                allow_edge_fallback=not args.no_edge_fallback,
                voice=args.voice,
                rate=args.rate,
                pitch=args.pitch,
            )
            print(f"DONE! {args.backend.title()} render ready: {r.video_path} ({r.duration_s}s)")
        else:
            run_plan(plan_path, **common)
    elif args.plan:
        if args.backend in ("hyperframes", "hybrid"):
            from studio.ai.v3_schema import VideoPlanV3
            from studio.ai.world_model import materialize_constructions
            from studio.renderers import get_backend

            p3 = materialize_constructions(VideoPlanV3.model_validate_json(args.plan.read_text(encoding="utf-8")))
            r = get_backend(args.backend).render(
                p3,
                slug=args.output_stem or args.plan.stem,
                format_name=args.format,
                look_name=theme_resolved,
                caption_style=args.caption_style,
                tts_engine=args.tts,
                allow_edge_fallback=not args.no_edge_fallback,
                voice=args.voice,
                rate=args.rate,
                pitch=args.pitch,
            )
            print(f"DONE! {args.backend.title()} render ready: {r.video_path} ({r.duration_s}s)")
        else:
            run_plan(args.plan, **common)
    elif args.script:
        run_pipeline(script_path=args.script, scene_name=args.scene, **common)
    else:
        parser.error("give a scene script, --plan, or --topic/--prompt")


if __name__ == "__main__":
    main()
