"""Render backend selector and empirical cross-backend benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from studio.renderers.base import BackendRenderResult, RenderBackend
from studio.renderers.manimgx_backend import ManimGXBackend

if TYPE_CHECKING:
    from studio.ai.v3_schema import VideoPlanV3


def get_backend(name: str = "manimgx") -> RenderBackend:
    """Resolve named render backend ('manimgx' or 'hyperframes')."""
    key = name.strip().lower()
    if key == "manimgx":
        return ManimGXBackend()
    if key == "hyperframes":
        from studio.renderers.hyperframes_backend import HyperFramesBackend

        return HyperFramesBackend()
    raise ValueError(f"Unknown render backend {name!r}. Choose 'manimgx' or 'hyperframes'.")


def benchmark_backends(
    plan: "VideoPlanV3",
    *,
    slug: str,
    format_name: str = "widescreen_hd",
    look_name: str = "midnight_studio",
    out_dir: Path | None = None,
) -> list[BackendRenderResult]:
    """Run identical VideoPlanV3 on both ManimGX and HyperFrames and compare metrics."""
    results: list[BackendRenderResult] = []
    for name in ("manimgx", "hyperframes"):
        backend = get_backend(name)
        target = (out_dir / f"{slug}_{name}_{format_name}.mp4") if out_dir else None
        res = backend.render(
            plan,
            slug=f"{slug}_{name}",
            format_name=format_name,
            look_name=look_name,
            out_path=target,
        )
        results.append(res)
    return results


__all__ = ["BackendRenderResult", "RenderBackend", "get_backend", "benchmark_backends"]
