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


def all_coords_finite(pts: Sequence[Sequence[float]]) -> bool:
    return all(len(v) >= 2 and math.isfinite(float(v[0])) and math.isfinite(float(v[1])) for v in pts)


def is_convex_polygon(vertices: Sequence[Sequence[float]], tol: float = 1e-6) -> bool:
    """Verify polygon has >=3 vertices and consistent turn sign (convex)."""
    pts = clean_polygon_vertices(vertices)
    n = len(pts)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        x3, y3 = pts[(i + 2) % n]
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if abs(cross) <= tol:
            continue
        cur = 1 if cross > 0 else -1
        if sign == 0:
            sign = cur
        elif cur != sign:
            return False
    return sign != 0


def has_self_intersection(vertices: Sequence[Sequence[float]], tol: float = 1e-6) -> bool:
    """Detect non-adjacent edge intersections in a polygon."""
    pts = clean_polygon_vertices(vertices)
    n = len(pts)
    if n < 4:
        return False

    def _ccw(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    def _seg_inter(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> bool:
        d1, d2 = _ccw(a, b, c), _ccw(a, b, d)
        d3, d4 = _ccw(c, d, a), _ccw(c, d, b)
        return (d1 * d2 < -tol) and (d3 * d4 < -tol)

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _seg_inter(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                return True
    return False


def _ensure_ccw(pts: list[list[float]]) -> list[list[float]]:
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return pts if s >= 0 else list(reversed(pts))


def _clip_convex(subject: list[list[float]], clipper: list[list[float]]) -> list[list[float]]:
    """Sutherland-Hodgman convex polygon intersection."""
    output = _ensure_ccw(clean_polygon_vertices(subject))
    clip = _ensure_ccw(clean_polygon_vertices(clipper))
    if len(output) < 3 or len(clip) < 3:
        return []
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        inp = output
        output = []
        if not inp:
            break

        def _inside(p: Sequence[float]) -> bool:
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-9

        def _intersect(p1: Sequence[float], p2: Sequence[float]) -> list[float]:
            dc = [a[0] - b[0], a[1] - b[1]]
            dp = [p1[0] - p2[0], p1[1] - p2[1]]
            n1 = a[0] * b[1] - a[1] * b[0]
            n2 = p1[0] * p2[1] - p1[1] * p2[0]
            denom = dc[0] * dp[1] - dc[1] * dp[0]
            if abs(denom) < 1e-12:
                return [float(p2[0]), float(p2[1])]
            return [(n1 * dp[0] - n2 * dc[0]) / denom, (n1 * dp[1] - n2 * dc[1]) / denom]

        s = inp[-1]
        for e in inp:
            if _inside(e):
                if not _inside(s):
                    output.append(_intersect(s, e))
                output.append([float(e[0]), float(e[1])])
            elif _inside(s):
                output.append(_intersect(s, e))
            s = e
    return output


def polygons_overlap(p1: Sequence[Sequence[float]], p2: Sequence[Sequence[float]], tol: float = 1e-4) -> bool:
    """Return True if two convex polygons have positive-area interior intersection (> tol)."""
    inter = _clip_convex([list(v) for v in p1], [list(v) for v in p2])
    return polygon_area(inter) > tol


def polygon_inside(inner: Sequence[Sequence[float]], outer: Sequence[Sequence[float]], tol: float = 1e-3) -> bool:
    """Return True if convex polygon `inner` is contained within convex polygon `outer`."""
    c_in = clean_polygon_vertices(inner)
    a_in = polygon_area(c_in)
    if a_in <= 0:
        return False
    inter = _clip_convex(c_in, [list(v) for v in outer])
    return abs(polygon_area(inter) - a_in) <= tol


def verify_partition(pieces: Sequence[Sequence[Sequence[float]]], outer: Sequence[Sequence[float]], tol: float = 1e-3) -> list[str]:
    """Verify pieces lie inside `outer`, do not overlap pairwise, and conserve total area."""
    issues: list[str] = []
    for i, pc in enumerate(pieces):
        if not polygon_inside(pc, outer, tol=tol):
            issues.append(f"Partition piece {i} extends outside outer container.")
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if polygons_overlap(pieces[i], pieces[j], tol=tol):
                issues.append(f"Partition pieces {i} and {j} overlap.")
    total_pieces = sum(polygon_area(pc) for pc in pieces)
    outer_a = polygon_area(outer)
    if abs(total_pieces - outer_a) > tol:
        issues.append(f"Partition area conservation failed: pieces={total_pieces:.4f} vs outer={outer_a:.4f}.")
    return issues


def materialize_constructions(plan: "VideoPlanV3") -> "VideoPlanV3":
    """Deterministically materialize typed ConstructionOps into SceneObject definitions."""
    from studio.ai.math_truth import bayes_posterior, discrete_convolution, extract_taylor_coefficients, gradient_descent_trajectory
    from studio.ai.v3_schema import SceneObject

    out = plan.model_copy(deep=True)
    registry: dict[str, list[list[float]]] = {}

    for sc in out.scenes:
        for obj in sc.objects:
            if obj.kind == "polygon" and obj.vertices:
                registry[obj.id] = clean_polygon_vertices(obj.vertices)
        if not sc.constructions:
            continue

        existing_ids = {o.id for o in sc.objects}
        for c in sc.constructions:
            p = c.params
            if c.op == "right_triangle":
                a = float(p.a if p.a is not None else 3.0)
                b = float(p.b if p.b is not None else 4.0)
                origin = tuple(float(x) for x in (p.origin or [0.0, 0.0]))
                verts = right_triangle(a, b, (origin[0], origin[1]))
                oid = c.output_ids[0] if c.output_ids else "right_tri"
                registry[oid] = verts
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="polygon", persist=c.persist, color_role=c.color_role, vertices=verts, label=c.label or "triangle"))
                    existing_ids.add(oid)
            elif c.op == "square_on_segment":
                role_map = {"leg_a": 0, "hypotenuse": 1, "leg_b": 2}
                idx0 = role_map[p.edge_role] if p.edge_role in role_map else int(p.edge_index if p.edge_index is not None else 0)
                outward = bool(p.outward if p.outward is not None else True)
                if c.source_id:
                    if c.source_id not in registry:
                        raise ValueError(f"square_on_segment missing source_id {c.source_id!r} in registry.")
                    src = registry[c.source_id]
                    p1, p2 = src[idx0 % len(src)], src[(idx0 + 1) % len(src)]
                elif p.p1 and p.p2:
                    p1 = [float(x) for x in p.p1]
                    p2 = [float(x) for x in p.p2]
                else:
                    raise ValueError("square_on_segment requires valid source_id or explicit p1, p2.")
                verts = square_on_segment(p1, p2, outward=outward)
                oid = c.output_ids[0] if c.output_ids else f"sq_{idx0}"
                registry[oid] = verts
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="polygon", persist=c.persist, color_role=c.color_role, vertices=verts, label=c.label or "square"))
                    existing_ids.add(oid)
            elif c.op == "pythagorean_dissection":
                a = float(p.a if p.a is not None else 3.0)
                b = float(p.b if p.b is not None else 4.0)
                d = pythagorean_dissection(a, b)
                ids = c.output_ids or ["outer_square", "tri_1", "tri_2", "tri_3", "tri_4", "inner_square"]
                items: list[tuple[str, list[list[float]], str, str]] = [
                    (ids[0], d["outer_square"], "primary", "square"),  # type: ignore[arg-type]
                    (ids[1], d["triangles"][0], "secondary", "triangle"),  # type: ignore[index]
                    (ids[2], d["triangles"][1], "tertiary", "triangle"),  # type: ignore[index]
                    (ids[3], d["triangles"][2], "secondary", "triangle"),  # type: ignore[index]
                    (ids[4], d["triangles"][3], "tertiary", "triangle"),  # type: ignore[index]
                    (ids[5], d["inner_square"], "highlight", "square"),  # type: ignore[arg-type]
                ]
                for oid, verts, role, lbl in items:
                    registry[oid] = verts
                    if oid not in existing_ids:
                        sc.objects.append(SceneObject(id=oid, kind="polygon", persist=c.persist, color_role=role, vertices=verts, label=lbl))  # type: ignore[arg-type]
                        existing_ids.add(oid)
            elif c.op == "translate_polygon":
                if not c.source_id or c.source_id not in registry:
                    raise ValueError(f"translate_polygon missing source_id {c.source_id!r} in registry.")
                dx = float(p.dx if p.dx is not None else 0.0)
                dy = float(p.dy if p.dy is not None else 0.0)
                verts = translate_polygon(registry[c.source_id], dx, dy)
                oid = c.output_ids[0] if c.output_ids else f"{c.source_id}_shifted"
                registry[oid] = verts
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="polygon", persist=c.persist, color_role=c.color_role, vertices=verts, label=c.label or "polygon"))
                    existing_ids.add(oid)
            elif c.op == "rotate_polygon":
                if not c.source_id or c.source_id not in registry:
                    raise ValueError(f"rotate_polygon missing source_id {c.source_id!r} in registry.")
                ang = math.radians(float(p.angle_deg if p.angle_deg is not None else 90.0))
                ctr = tuple(float(x) for x in (p.center or [0.0, 0.0]))
                verts = rotate_polygon(registry[c.source_id], ang, (ctr[0], ctr[1]))
                oid = c.output_ids[0] if c.output_ids else f"{c.source_id}_rot"
                registry[oid] = verts
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="polygon", persist=c.persist, color_role=c.color_role, vertices=verts, label=c.label or "polygon"))
                    existing_ids.add(oid)
            elif c.op == "gradient_descent_path":
                expr = p.expr or "x**2"
                x0 = float(p.x0 if p.x0 is not None else 2.0)
                lr = float(p.lr if p.lr is not None else 0.2)
                steps = int(p.steps if p.steps is not None else 5)
                traj = gradient_descent_trajectory(expr, x0, lr, steps)
                for idx_pt, (xv, yv, _) in enumerate(traj):
                    oid = c.output_ids[idx_pt] if idx_pt < len(c.output_ids) else f"gd_pt_{idx_pt}"
                    if oid not in existing_ids:
                        role = "highlight" if idx_pt == len(traj) - 1 else c.color_role
                        sc.objects.append(SceneObject(id=oid, kind="point", persist=c.persist, color_role=role, coords=[xv, yv], axes_id=p.axes_id, label=f"step_{idx_pt}"))
                        existing_ids.add(oid)
            elif c.op == "bayes_area_model":
                prior = float(p.prior if p.prior is not None else 0.2)
                peh = float(p.p_e_given_h if p.p_e_given_h is not None else 0.9)
                penh = float(p.p_e_given_not_h if p.p_e_given_not_h is not None else 0.15)
                bayes_posterior(prior, peh, penh)
                ids = c.output_ids or ["bayes_space", "bayes_prior", "bayes_tp", "bayes_fp"]
                boxes: list[tuple[str, list[list[float]], str, str]] = [
                    (ids[0], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "primary", "sample_space"),
                    (ids[1], [[0.0, 0.0], [prior, 0.0], [prior, 1.0], [0.0, 1.0]], "secondary", "prior_region"),
                    (ids[2], [[0.0, 0.0], [prior, 0.0], [prior, peh], [0.0, peh]], "highlight", "true_positive"),
                    (ids[3], [[prior, 0.0], [1.0, 0.0], [1.0, penh], [prior, penh]], "danger", "false_positive"),
                ]
                for oid, verts, role, lbl in boxes:
                    registry[oid] = verts
                    if oid not in existing_ids:
                        sc.objects.append(SceneObject(id=oid, kind="polygon", persist=c.persist, color_role=role, vertices=verts, axes_id=p.axes_id, label=lbl))  # type: ignore[arg-type]
                        existing_ids.add(oid)
            elif c.op == "dice_sum_distribution":
                n_dice = int(p.n_dice if p.n_dice is not None else 2)
                support, probs = discrete_convolution([1.0 / 6.0] * 6, n_dice, offset_per_trial=1)
                oid = c.output_ids[0] if c.output_ids else f"dice_{n_dice}_dist"
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="bar_chart", persist=c.persist, color_role=c.color_role, values=probs, bar_labels=[str(s) for s in support], label=c.label or f"{n_dice}_dice_sum"))
                    existing_ids.add(oid)
            elif c.op == "gaussian_approx":
                mu = float(p.mu if p.mu is not None else 0.0)
                sigma = max(1e-3, float(p.sigma if p.sigma is not None else 1.0))
                oid = c.output_ids[0] if c.output_ids else "gaussian_curve"
                expr = f"exp(-0.5*((x-({mu}))/({sigma}))**2)/({sigma}*sqrt(2*pi))"
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="function_plot", persist=c.persist, color_role=c.color_role, expr=expr, axes_id=p.axes_id, label=c.label or "gaussian"))
                    existing_ids.add(oid)
            elif c.op == "taylor_approximation":
                expr = p.expr or "cos(x)"
                deg = int(p.degree if p.degree is not None else 4)
                x0 = float(p.x0 if p.x0 is not None else 0.0)
                coeffs = extract_taylor_coefficients(expr, deg, x0=x0)
                terms = [f"({c_k:.8g})*(x-({x0}))**{k}" for k, c_k in enumerate(coeffs) if abs(c_k) > 1e-12]
                poly_expr = " + ".join(terms) if terms else "0"
                oid = c.output_ids[0] if c.output_ids else f"taylor_{deg}"
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="function_plot", persist=c.persist, color_role=c.color_role, expr=poly_expr, axes_id=p.axes_id, label=c.label or f"taylor_{deg}"))
                    existing_ids.add(oid)
            elif c.op == "fourier_partial_sum":
                n_terms = max(1, int(p.n_terms if p.n_terms is not None else 5))
                stop = 2 * n_terms - 1
                expr = f"4/pi*sum(k, 1, {stop}, 2, sin(k*x)/k)"
                oid = c.output_ids[0] if c.output_ids else f"fourier_{n_terms}"
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="function_plot", persist=c.persist, color_role=c.color_role, expr=expr, axes_id=p.axes_id, label=c.label or f"fourier_{n_terms}"))
                    existing_ids.add(oid)
    return out
