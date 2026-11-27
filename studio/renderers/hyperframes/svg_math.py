"""Deterministic Python SVG curve sampling for HyperFrames (zero browser eval)."""

from __future__ import annotations

import math

from studio.ai.expr import compile_expr


def _sample_function_points(expr_str: str, x_range: list[float], y_range: list[float], n: int = 160) -> list[list[float]]:
    """Safely evaluate 1D function curves in Python."""
    fn = compile_expr(expr_str or "0", ("x",))
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    y_span = max(y1 - y0, 1e-6)
    pts: list[list[float]] = []
    for i in range(n):
        xv = x0 + (x1 - x0) * i / max(1, n - 1)
        try:
            yv = float(fn(x=xv))
        except Exception:
            continue
        if math.isfinite(yv) and (y0 - 0.5 * y_span) <= yv <= (y1 + 0.5 * y_span):
            pts.append([round(xv, 5), round(max(y0 - 0.1 * y_span, min(y1 + 0.1 * y_span, yv)), 5)])
    return pts


def _sample_parametric_points(expr_str: str, t_range: list[float], n: int = 180) -> list[list[float]]:
    if ";" not in expr_str:
        return []
    p0, p1 = expr_str.split(";", 1)
    fx, fy = compile_expr(p0, ("t",)), compile_expr(p1, ("t",))
    t0, t1 = float(t_range[0]), float(t_range[1])
    pts: list[list[float]] = []
    for i in range(n):
        tv = t0 + (t1 - t0) * i / max(1, n - 1)
        try:
            xv, yv = float(fx(t=tv)), float(fy(t=tv))
            if math.isfinite(xv) and math.isfinite(yv):
                pts.append([round(xv, 5), round(yv, 5)])
        except Exception:
            continue
    return pts
