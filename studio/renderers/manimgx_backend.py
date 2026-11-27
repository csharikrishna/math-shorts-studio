"""ManimGX GPU render backend implementing the shared RenderBackend interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from studio.ai.env import PROJECT_ROOT
from studio.config import FORMATS
from studio.renderers.base import BackendRenderResult, RenderBackend

if TYPE_CHECKING:
    from studio.ai.v3_schema import VideoPlanV3
    from studio.audio.tts import AudioTimeline


class ManimGXBackend(RenderBackend):
    """Wraps ManimGX GPU rendering (`scenes/v3_scene.py`) + two-pass FFmpeg mastering."""

    name = "manimgx"

    def render(
        self,
        plan: "VideoPlanV3",
        *,
        slug: str,
        format_name: str = "widescreen_hd",
        look_name: str = "midnight_studio",
        audio_timeline: "AudioTimeline | None" = None,
        out_path: Path | None = None,
        build_dir: Path | None = None,
        subtitle_ass: Path | None = None,
    ) -> BackendRenderResult:
        from studio.pipeline import run_plan

        t0 = time.perf_counter()
        bdir = build_dir or (PROJECT_ROOT / "build" / f"backend_manimgx_{slug}_{format_name}")
        bdir.mkdir(parents=True, exist_ok=True)
        plan_file = bdir / f"{slug}.json"
        plan_file.write_text(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

        final_mp4 = run_plan(
            plan_file,
            format_name=format_name,
            theme_name=look_name if look_name in ("midnight_studio", "obsidian_chalk", "cartesian_stone", "editorial_paper") else "midnight_studio",
            strict_layout=False,
        )
        if out_path is not None and out_path.resolve() != final_mp4.resolve():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(final_mp4.read_bytes())
            final_mp4 = out_path

        dt = time.perf_counter() - t0
        prof = FORMATS.get(format_name, FORMATS["widescreen_hd"])
        from studio.longform import _probe_duration
        from studio.renderers.hyperframes_backend import _current_process_rss_mb

        dur = _probe_duration(final_mp4)
        size = final_mp4.stat().st_size if final_mp4.exists() else 0
        return BackendRenderResult(
            backend=self.name,
            video_path=final_mp4,
            duration_s=dur,
            render_time_s=round(dt, 3),
            peak_rss_mb=_current_process_rss_mb(),
            file_size_bytes=size,
            width=prof.pixel_width,
            height=prof.pixel_height,
            fps=prof.fps,
            audio_sample_rate=48000,
            layout_issues=[],
            notes="Rendered via ManimGX wgpu hardware pipeline + NVENC/libx264 mastering.",
        )
