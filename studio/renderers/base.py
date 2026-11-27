"""Backend-agnostic rendering interface and performance/telemetry result models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from studio.ai.v3_schema import VideoPlanV3
    from studio.audio.tts import AudioTimeline


class BackendRenderResult(BaseModel):
    """Normalized output and performance telemetry emitted by any render backend."""

    backend: str
    video_path: Path
    duration_s: float
    render_time_s: float
    peak_rss_mb: float = 0.0
    file_size_bytes: int = 0
    width: int
    height: int
    fps: int
    audio_sample_rate: int = 48000
    layout_issues: list[str] = Field(default_factory=list)
    notes: str = ""


class RenderBackend(ABC):
    """Abstract video composition & rendering backend."""

    name: str = "abstract"

    @abstractmethod
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
        """Render plan into an MP4 + telemetry using this backend."""
