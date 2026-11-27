"""Backward-compatible facade re-exporting modular HyperFrames renderer components."""

from __future__ import annotations

from studio.renderers.hyperframes import (
    AUDIT_JS_SNIPPET,
    HyperFramesBackend,
    _CDPClient,
    _LoopbackHandler,
    _current_process_rss_mb,
    _launch_chrome_cdp,
    _pretty_math_label,
    _role_color,
    _sample_function_points,
    _sample_parametric_points,
    _slot_box,
    _start_loopback_server,
    compile_plan_to_html,
    find_chromium_binary,
    stage_fonts_to_site,
    validate_safe_relpath,
)

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
