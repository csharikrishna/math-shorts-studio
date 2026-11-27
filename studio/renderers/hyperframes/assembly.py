"""HyperFramesBackend assembly orchestrator combining audio, HTML compiler, capture, and mastering."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from studio.ai.env import PROJECT_ROOT
from studio.audio.bgm import generate_ambient_bgm
from studio.audio.master import _pick_encoder, build_audio_mix_graph, measure_loudnorm_filter
from studio.audio.sfx import SfxLibrary
from studio.audio.tts import AudioTimeline, BeatSpec, build_voice_timeline, find_ffmpeg
from studio.captions import normalize_caption_style
from studio.config import FORMATS
from studio.design.spec import default_design_spec, resolve_look
from studio.renderers.base import BackendRenderResult, RenderBackend
from studio.renderers.hyperframes.browser import _current_process_rss_mb, _start_loopback_server
from studio.renderers.hyperframes.capture import capture_frames_to_ffmpeg
from studio.renderers.hyperframes.compiler import compile_plan_to_html
from studio.renderers.hyperframes.typography import stage_fonts_to_site

if TYPE_CHECKING:
    from studio.ai.v3_schema import VideoPlanV3


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
        caption_style: str | None = None,
        tts_engine: str = "kokoro",
        allow_edge_fallback: bool = True,
        voice: str = "en-US-AndrewNeural",
        rate: str = "-4%",
        pitch: str = "-2Hz",
    ) -> BackendRenderResult:
        t0 = time.perf_counter()
        prof = FORMATS.get(format_name, FORMATS["widescreen_hd"])
        resolved_look = resolve_look(look_name).name
        resolved_cap = normalize_caption_style(caption_style, default="kinetic" if prof.is_vertical else "editorial")
        bdir = build_dir or (PROJECT_ROOT / "build" / f"backend_hyperframes_{slug}_{format_name}")
        bdir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_path or (PROJECT_ROOT / "output" / f"HF_{slug}_{format_name}.mp4")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        if audio_timeline is None:
            from studio.creative.dialogue import resolve_speaker_voices

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

        SfxLibrary(bdir / "sfx").ensure_all()
        bgm_wav = generate_ambient_bgm(bdir / "bgm.wav", duration=audio_timeline.total_duration + 3.0)

        design = default_design_spec(format_name=format_name, look_name=resolved_look)
        html_doc, _total_dur = compile_plan_to_html(plan, design, audio_timeline=audio_timeline, caption_style=resolved_cap)
        site_dir = bdir / "site"
        site_dir.mkdir(parents=True, exist_ok=True)
        stage_fonts_to_site(site_dir)
        (site_dir / "index.html").write_text(html_doc, encoding="utf-8")

        fps = prof.fps
        ffmpeg_bin = find_ffmpeg()
        audio_mix = build_audio_mix_graph(1, 2, sfx_idx=None)
        ln_filter = measure_loudnorm_filter(
            ffmpeg_bin,
            ["-i", str(audio_timeline.master_voice_wav), "-i", str(bgm_wav)],
            audio_mix,
        )
        enc_name, codec_flags = _pick_encoder()
        filter_complex = f"{audio_mix},{ln_filter}[master_a];[0:v]tpad=stop_mode=clone:stop_duration=2.0,format=yuv420p[master_v]"

        ff_cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-framerate", str(fps),
            "-i", "-",
            "-i", str(audio_timeline.master_voice_wav),
            "-i", str(bgm_wav),
            "-filter_complex", filter_complex,
            "-map", "[master_v]",
            "-map", "[master_a]",
            *codec_flags,
            "-fps_mode", "cfr",
            "-shortest",
            "-c:a", "aac",
            "-ar", "48000",
            "-b:a", "160k",
            "-movflags", "+faststart",
            str(out_mp4),
        ]

        server, port = _start_loopback_server(site_dir)
        tmp_profile = Path(tempfile.mkdtemp(prefix="hf_chrome_"))
        try:
            captured_frames, total_frames, layout_issues = capture_frames_to_ffmpeg(
                port=port,
                prof=prof,
                audio_timeline=audio_timeline,
                ff_cmd=ff_cmd,
                tmp_profile=tmp_profile,
            )
        finally:
            server.shutdown()
            server.server_close()
            shutil.rmtree(tmp_profile, ignore_errors=True)

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
            notes=f"Rendered via modular HyperFrames CDP single-pass {enc_name} ({captured_frames}/{total_frames} unique frames captured).",
        )
