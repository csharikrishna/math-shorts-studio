"""Scene-level hybrid backend orchestration: single AudioTimeline across ManimGX & HyperFrames."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from studio.ai.env import PROJECT_ROOT
from studio.audio.bgm import generate_ambient_bgm
from studio.audio.master import master_video
from studio.audio.sfx import SfxLibrary
from studio.audio.tts import AudioTimeline, BeatSpec, SyncedBeat, build_voice_timeline, find_ffmpeg
from studio.captions import normalize_caption_style
from studio.config import FORMATS
from studio.creative.dialogue import resolve_speaker_voices
from studio.design.spec import resolve_look
from studio.registry import select_scene_backend
from studio.renderers.base import BackendRenderResult, RenderBackend
from studio.timeline import MasterTimeline

if TYPE_CHECKING:
    from studio.ai.v3_schema import SceneV3, VideoPlanV3


@dataclass
class SceneSegmentPartition:
    backend: str
    scenes: list["SceneV3"]
    beats: list[SyncedBeat]
    start_s: float
    end_s: float


def partition_plan_by_backend(
    plan: "VideoPlanV3",
    timeline: AudioTimeline,
) -> list[SceneSegmentPartition]:
    """Partition scenes into contiguous runs routed to optimal backends with preserved object closure."""
    beat_map = {b.id: b for b in timeline.beats}
    parts: list[SceneSegmentPartition] = []
    for sc in plan.scenes:
        chosen = select_scene_backend(sc, requested="auto")
        sb = beat_map.get(sc.scene_id)
        if sb is None:
            continue
        if parts and parts[-1].backend == chosen:
            parts[-1].scenes.append(sc)
            parts[-1].beats.append(sb)
            parts[-1].end_s = max(parts[-1].end_s, float(sb.end_time))
        else:
            parts.append(
                SceneSegmentPartition(
                    backend=chosen,
                    scenes=[sc],
                    beats=[sb],
                    start_s=float(sb.start_time),
                    end_s=float(sb.end_time),
                )
            )
    return parts


class HybridRenderBackend(RenderBackend):
    """Scene-level hybrid router synthesizing audio ONCE and muxing mixed ManimGX/HyperFrames segments."""

    name = "hybrid"

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
        caption_style: str | None = None,
        tts_engine: str = "kokoro",
        allow_edge_fallback: bool = True,
        voice: str = "en-US-AndrewNeural",
        rate: str = "-4%",
        pitch: str = "-2Hz",
    ) -> BackendRenderResult:
        from studio.ai.v3_schema import VideoPlanV3
        from studio.ai.world_model import materialize_constructions
        from studio.renderers.hyperframes import HyperFramesBackend
        from studio.renderers.hyperframes.browser import _current_process_rss_mb
        from studio.renderers.manimgx_backend import ManimGXBackend
        from studio.renderers.scene_state import (
            SceneRuntimeState,
            advance_scene_state,
            materialize_carry_objects,
        )

        plan = materialize_constructions(plan)
        t0 = time.perf_counter()
        prof = FORMATS.get(format_name, FORMATS["widescreen_hd"])
        resolved_look = resolve_look(look_name).name
        resolved_cap = normalize_caption_style(caption_style, default="kinetic" if prof.is_vertical else "editorial")

        bdir = build_dir or (PROJECT_ROOT / "build" / f"hybrid_{slug}_{format_name}")
        bdir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_path or (PROJECT_ROOT / "output" / f"HYB_{slug}_{format_name}.mp4")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        if audio_timeline is None:
            sp_voices = resolve_speaker_voices(voice_override=voice if voice != "en-US-AndrewNeural" else None)
            beats = [
                BeatSpec(id=s.scene_id, narration=s.narration, caption=s.purpose, pause_after=0.30, speaker=s.speaker)
                for s in plan.scenes
            ]
            audio_timeline = build_voice_timeline(
                beats,
                build_dir=bdir,
                voice=voice,
                rate=rate,
                pitch=pitch,
                engine=tts_engine,
                allow_edge_fallback=allow_edge_fallback,
                speaker_voices=sp_voices,
            )

        MasterTimeline(audio=audio_timeline).export_subtitles(
            bdir, slug, caption_style=resolved_cap, format_name=format_name, look_name=resolved_look
        )
        parts = partition_plan_by_backend(plan, audio_timeline)

        # Single-backend fast path avoids any unnecessary segment splitting or mux overhead
        if len(parts) <= 1:
            only_backend = parts[0].backend if parts else "manimgx"
            runner: RenderBackend = HyperFramesBackend() if only_backend == "hyperframes" else ManimGXBackend()
            res = runner.render(
                plan,
                slug=slug,
                format_name=format_name,
                look_name=resolved_look,
                audio_timeline=audio_timeline,
                out_path=out_mp4,
                build_dir=bdir,
                caption_style=resolved_cap,
                tts_engine=tts_engine,
                allow_edge_fallback=allow_edge_fallback,
                voice=voice,
                rate=rate,
                pitch=pitch,
            )
            res.notes = f"Hybrid router selected single optimal backend '{only_backend}'. " + res.notes
            return res

        # Multi-backend segmented render reusing exact audio_timeline slice timestamps & single master audio mix
        ffmpeg_bin = find_ffmpeg()
        SfxLibrary(bdir / "sfx").ensure_all()
        bgm_wav = generate_ambient_bgm(bdir / "bgm.wav", duration=audio_timeline.total_duration + 3.0)
        seg_raw_mp4s: list[Path] = []
        layout_issues: list[str] = []

        # Track canonical evaluated SceneRuntimeState across backend boundaries so transforms persist & removed items stay gone
        runtime_state = SceneRuntimeState()
        for s_idx, part in enumerate(parts):
            seg_dir = bdir / f"seg_{s_idx:02d}_{part.backend}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_scenes = [s.model_copy(deep=True) for s in part.scenes]
            if seg_scenes:
                seg_ids_in_part = {o.id for sc in seg_scenes for o in sc.objects}
                has_stage_reset = any(o.kind in ("axes", "table", "matrix", "bar_chart", "graph_network") for o in seg_scenes[0].objects)
                needed = (
                    {e.target_id for sc in seg_scenes for e in sc.events if e.op != "remove"}
                    | {e.source_id for sc in seg_scenes for e in sc.events if e.source_id}
                    | {o.axes_id for sc in seg_scenes for o in sc.objects if o.axes_id}
                ) - seg_ids_in_part - runtime_state.removed_ids
                carry_objs = [
                    co for co in materialize_carry_objects(runtime_state, needed, include_visible_persistent=not has_stage_reset)
                    if co.id not in seg_ids_in_part
                ]
                if carry_objs:
                    # Parent axes must precede dependent carried overlays so coordinate mapping matches
                    carry_objs.sort(key=lambda o: 0 if o.kind == "axes" else 1)
                    seg_scenes[0].objects = list(carry_objs) + list(seg_scenes[0].objects)

            (seg_dir / "seg_state_in.json").write_text(runtime_state.model_dump_json(indent=2), encoding="utf-8")
            for sc in part.scenes:
                runtime_state = advance_scene_state(runtime_state, sc)
            (seg_dir / "seg_state_out.json").write_text(runtime_state.model_dump_json(indent=2), encoding="utf-8")

            seg_plan = VideoPlanV3(
                schema_version="3.0",
                topic=plan.topic,
                title=plan.title,
                audience=plan.audience,
                strategy=plan.strategy,
                math_models=plan.math_models,
                scenes=seg_scenes,
            )
            # Re-base segment timeline cleanly to start at t=0.0 matching exact spoken beat lengths
            seg_voice = seg_dir / "seg_voice.wav"
            seg_dur = max(0.3, part.end_s - part.start_s)
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-ss",
                    f"{part.start_s:.4f}",
                    "-t",
                    f"{seg_dur:.4f}",
                    "-i",
                    str(audio_timeline.master_voice_wav),
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                    str(seg_voice),
                ],
                check=True,
                capture_output=True,
            )
            rebased_beats = [
                SyncedBeat(
                    id=b.id,
                    narration=b.narration,
                    caption=b.caption,
                    audio_path=b.audio_path,
                    start_time=round(b.start_time - part.start_s, 4),
                    end_time=round(b.end_time - part.start_s, 4),
                    speech_duration=b.speech_duration,
                    total_duration=b.total_duration,
                    words=[[w[0], round(w[1] - part.start_s, 4), round(w[2] - part.start_s, 4), *w[3:]] for w in b.words],
                    marks={k: round(v - part.start_s, 4) for k, v in b.marks.items()},
                    speaker=b.speaker,
                )
                for b in part.beats
            ]
            seg_tl = AudioTimeline(
                voice=audio_timeline.voice,
                rate=audio_timeline.rate,
                master_voice_wav=str(seg_voice),
                total_duration=round(seg_dur, 4),
                beats=rebased_beats,
            )
            seg_tl.save(seg_dir / "timeline.json")

            seg_mp4 = seg_dir / f"seg_{s_idx:02d}.mp4"
            if part.backend == "hyperframes":
                res = HyperFramesBackend().render(
                    seg_plan,
                    slug=f"{slug}_seg_{s_idx}",
                    format_name=format_name,
                    look_name=resolved_look,
                    audio_timeline=seg_tl,
                    out_path=seg_mp4,
                    build_dir=seg_dir,
                    caption_style=resolved_cap,
                    video_only=True,
                )
                layout_issues.extend(res.layout_issues)
            else:
                from studio.pipeline import V3_SCENE_SCRIPT

                seg_json = seg_dir / "seg_plan.json"
                seg_json.write_text(json.dumps(seg_plan.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
                env = os.environ.copy()
                env["STUDIO_V3_PLAN"] = str(seg_json.resolve())
                env["STUDIO_FORMAT"] = format_name
                env["STUDIO_THEME"] = resolved_look
                env["STUDIO_CAPTION_STYLE"] = resolved_cap
                env["STUDIO_BUILD_DIR"] = str(seg_dir)
                env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                raw_seg = seg_dir / f"seg_{s_idx}_raw.mp4"
                subprocess.run(
                    [
                        os.sys.executable,
                        "-m",
                        "manimgx",
                        "render",
                        str(V3_SCENE_SCRIPT),
                        "V3Scene",
                        "-r",
                        prof.resolution_str,
                        "--fps",
                        str(prof.fps),
                        "--preset",
                        "ultrafast",
                        "--crf",
                        "12",
                        "-o",
                        str(raw_seg),
                    ],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                seg_mp4 = raw_seg
            seg_raw_mp4s.append(seg_mp4)

        merged_meta_hyb: dict[str, list] = {}
        for s_idx_m in range(len(parts)):
            m_p = bdir / f"seg_{s_idx_m:02d}" / "scene_frame_metadata.json"
            if m_p.exists():
                try:
                    merged_meta_hyb.update(json.loads(m_p.read_text(encoding="utf-8")))
                except Exception:
                    pass
        if merged_meta_hyb:
            (bdir / "scene_frame_metadata.json").write_text(json.dumps(merged_meta_hyb, indent=2), encoding="utf-8")

        joined_video = bdir / "hybrid_video_only.mp4"
        # Use FFmpeg filter_complex concat with per-segment fps/PTS/yuv420p normalization
        # so heterogeneous container timebases (e.g. ManimGX 30 tbn vs HyperFrames 15360 tbn) never drift.
        ff_inputs: list[str] = []
        fc_parts: list[str] = []
        v_labels: list[str] = []
        for idx, p in enumerate(seg_raw_mp4s):
            ff_inputs.extend(["-i", str(p)])
            lbl = f"v{idx}"
            fc_parts.append(f"[{idx}:v]fps={prof.fps},setpts=PTS-STARTPTS,format=yuv420p,setsar=1[{lbl}]")
            v_labels.append(f"[{lbl}]")
        from studio.audio.master import _pick_encoder

        enc_name, _enc_flags = _pick_encoder()
        hyb_v_flags = (
            ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", "-cq", "18", "-b:v", "0", "-pix_fmt", "yuv420p"]
            if enc_name == "h264_nvenc"
            else ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p"]
        )
        n_segs = len(seg_raw_mp4s)
        fc_parts.append(f"{''.join(v_labels)}concat=n={n_segs}:v=1:a=0[vout]")
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                *ff_inputs,
                "-filter_complex",
                ";".join(fc_parts),
                "-map",
                "[vout]",
                "-an",
                *hyb_v_flags,
                "-r",
                str(prof.fps),
                str(joined_video),
            ],
            check=True,
            capture_output=True,
        )
        from studio.audio.demo_signals import apply_plan_audio_demos

        hyb_voice_wav = apply_plan_audio_demos(plan, audio_timeline, bdir)
        master_video(
            raw_video=joined_video,
            voice_wav=hyb_voice_wav,
            bgm_wav=bgm_wav,
            output_mp4=out_mp4,
            look="none",
        )

        dt = time.perf_counter() - t0
        from studio.longform import _probe_duration
        from studio.provenance import build_and_write_manifest

        dur = _probe_duration(out_mp4)
        cap_hyb = bdir / "caption_ir.json"
        if not cap_hyb.exists():
            MasterTimeline(audio=audio_timeline).export_subtitles(
                bdir, slug, caption_style=resolved_cap, format_name=format_name, look_name=resolved_look
            )
        build_and_write_manifest(
            slug=slug,
            topic=plan.topic,
            title=plan.title,
            mp4_path=out_mp4,
            plan=plan,
            backend=self.name,
            format_name=format_name,
            look_name=resolved_look,
            caption_style=resolved_cap,
            build_dir=bdir,
            audio_timeline=audio_timeline,
            timeline_path=(bdir / "timeline.json") if (bdir / "timeline.json").exists() else None,
            caption_ir_path=cap_hyb if cap_hyb.exists() else None,
            renderer_config={"tts_engine": tts_engine, "voice": voice, "rate": rate},
        )
        mix_desc = " -> ".join(f"{p.backend}({len(p.scenes)})" for p in parts)
        return BackendRenderResult(
            backend=self.name,
            video_path=out_mp4,
            duration_s=dur,
            render_time_s=round(dt, 3),
            peak_rss_mb=_current_process_rss_mb(),
            file_size_bytes=out_mp4.stat().st_size if out_mp4.exists() else 0,
            width=prof.pixel_width,
            height=prof.pixel_height,
            fps=prof.fps,
            audio_sample_rate=48000,
            layout_issues=layout_issues,
            notes=f"Hybrid scene-level mux ({mix_desc}) onto single canonical AudioTimeline.",
        )
