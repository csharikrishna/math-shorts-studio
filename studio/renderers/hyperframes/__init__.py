"""Modular HyperFrames browser-rendering package (OSI-clean HTML5 + SVG + headless Chromium CDP)."""

from __future__ import annotations

from studio.renderers.hyperframes.assembly import HyperFramesBackend
from studio.renderers.hyperframes.audit import AUDIT_JS_SNIPPET
from studio.renderers.hyperframes.browser import (
    _CDPClient,
    _LoopbackHandler,
    _current_process_rss_mb,
    _launch_chrome_cdp,
    _start_loopback_server,
    find_chromium_binary,
    validate_safe_relpath,
)
from studio.renderers.hyperframes.compiler import compile_plan_to_html
from studio.renderers.hyperframes.components import _role_color, _slot_box
from studio.renderers.hyperframes.svg_math import _sample_function_points, _sample_parametric_points
from studio.renderers.hyperframes.typography import _pretty_math_label, stage_fonts_to_site

__all__ = [
    "AUDIT_JS_SNIPPET",
    "HyperFramesBackend",
    "_CDPClient",
    "_LoopbackHandler",
    "_current_process_rss_mb",
    "_launch_chrome_cdp",
    "_pretty_math_label",
    "_role_color",
    "_sample_function_points",
    "_sample_parametric_points",
    "_slot_box",
    "_start_loopback_server",
    "compile_plan_to_html",
    "find_chromium_binary",
    "stage_fonts_to_site",
    "validate_safe_relpath",
]
