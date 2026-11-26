"""Mathematical World Model & Deterministic Geometric Construction Operators."""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def dist(p1: Sequence[float], p2: Sequence[float]) -> float:
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def clean_polygon_vertices(vertices: Sequence[Sequence[float]]) -> list[list[float]]:
    """Drop closing duplicate vertex if first == last."""
    pts = [[float(v[0]), float(v[1])] for v in vertices]
    if len(pts) >= 2 and math.isclose(pts[0][0], pts[-1][0], abs_tol=1e-9) and math.isclose(pts[0][1], pts[-1][1], abs_tol=1e-9):
        pts = pts[:-1]
    return pts


def polygon_area(vertices: Sequence[Sequence[float]]) -> float:
    """Shoelace formula area (always non-negative)."""
    pts = clean_polygon_vertices(vertices)
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def side_lengths(vertices: Sequence[Sequence[float]]) -> list[float]:
    pts = clean_polygon_vertices(vertices)
    n = len(pts)
    return [dist(pts[i], pts[(i + 1) % n]) for i in range(n)]


def is_square(vertices: Sequence[Sequence[float]], tol: float = 1e-4) -> bool:
    """Verify 4 distinct vertices, 4 equal nonzero sides, and 4 right angles."""
    pts = clean_polygon_vertices(vertices)
    if len(pts) != 4:
        return False
    sides = side_lengths(pts)
    s0 = sides[0]
    if s0 <= tol or any(abs(s - s0) > tol for s in sides):
        return False
    for i in range(4):
        p_prev = pts[(i - 1) % 4]
        p_curr = pts[i]
        p_next = pts[(i + 1) % 4]
        v1 = (p_prev[0] - p_curr[0], p_prev[1] - p_curr[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        if abs(dot) > tol * s0 * s0:
            return False
    return True


def is_right_triangle(vertices: Sequence[Sequence[float]], tol: float = 1e-4) -> bool:
    pts = clean_polygon_vertices(vertices)
    if len(pts) != 3 or polygon_area(pts) <= tol:
        return False
    s = sorted(side_lengths(pts))
    return abs(s[0] ** 2 + s[1] ** 2 - s[2] ** 2) <= tol * (s[2] ** 2 + 1.0)


def triangles_congruent(t1: Sequence[Sequence[float]], t2: Sequence[Sequence[float]], tol: float = 1e-4) -> bool:
    p1, p2 = clean_polygon_vertices(t1), clean_polygon_vertices(t2)
    if len(p1) != 3 or len(p2) != 3:
        return False
    s1, s2 = sorted(side_lengths(p1)), sorted(side_lengths(p2))
    return all(abs(a - b) <= tol for a, b in zip(s1, s2))


def right_triangle(a: float, b: float, origin: tuple[float, float] = (0.0, 0.0)) -> list[list[float]]:
    ox, oy = origin
    return [[ox, oy], [ox + a, oy], [ox, oy + b]]


def square_on_segment(p1: Sequence[float], p2: Sequence[float], outward: bool = True) -> list[list[float]]:
    """Construct exact square on segment p1 -> p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    nx, ny = (dy, -dx) if outward else (-dy, dx)
    return [
        [float(p1[0]), float(p1[1])],
        [float(p2[0]), float(p2[1])],
        [float(p2[0] + nx), float(p2[1] + ny)],
        [float(p1[0] + nx), float(p1[1] + ny)],
    ]


def translate_polygon(vertices: Sequence[Sequence[float]], dx: float, dy: float) -> list[list[float]]:
    return [[float(v[0] + dx), float(v[1] + dy)] for v in vertices]


def rotate_polygon(vertices: Sequence[Sequence[float]], angle_rad: float, center: tuple[float, float] = (0.0, 0.0)) -> list[list[float]]:
    cx, cy = center
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    out: list[list[float]] = []
    for x, y in vertices:
        rx, ry = x - cx, y - cy
        out.append([cx + rx * c - ry * s, cy + rx * s + ry * c])
    return out


def pythagorean_dissection(a: float = 3.0, b: float = 4.0) -> dict[str, object]:
    """Deterministically construct the (a+b)x(a+b) Pythagorean dissection proof."""
    if a <= 0 or b <= 0:
        raise ValueError("Leg lengths must be positive.")
    c = math.hypot(a, b)
    s = a + b
    # Center around (0, 0) for clean full-stage layout
    hx, hy = -s * 0.5, -s * 0.5
    outer = [[hx, hy], [hx + s, hy], [hx + s, hy + s], [hx, hy + s]]
    t1 = [[hx, hy], [hx + a, hy], [hx, hy + b]]
    t2 = [[hx + a, hy], [hx + s, hy], [hx + s, hy + a]]
    t3 = [[hx + s, hy + a], [hx + s, hy + s], [hx + b, hy + s]]
    t4 = [[hx + b, hy + s], [hx, hy + s], [hx, hy + b]]
    inner_c = [[hx + a, hy], [hx + s, hy + a], [hx + b, hy + s], [hx, hy + b]]
    return {
        "a": a,
        "b": b,
        "c": c,
        "outer_square": outer,
        "triangles": [t1, t2, t3, t4],
        "inner_square": inner_c,
        "outer_area": s * s,
        "triangles_area": 4.0 * 0.5 * a * b,
        "inner_area": c * c,
    }
