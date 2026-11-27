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


def _bind_object_from_model(obj: "SceneObject", model_data: dict[str, Any], sweep_cls: type) -> None:
    """Populate geometry, curves, vectors, tables, labels & keyframes directly from evaluated LinkedMathModelSpec."""
    out_key = (obj.model_output or "").strip()
    if not out_key:
        return
    if out_key in (
        "time_signal",
        "winding_curve",
        "spectrum",
        "continuous_fourier_spectrum",
        "normalized_spectrum",
        "dft_single_sided_spectrum",
        "dft_filtered_spectrum",
        "filtered_spectrum",
        "filtered_signal",
        "curve",
        "derivative_curve",
        "trajectory",
        "loss_curve",
        "phase_orbit",
        "time_series_x",
        "time_series_y",
        "energy_total",
        "response_curve",
        "phase_curve",
        "mean_curve",
        "upper_std_curve",
        "lower_std_curve",
        "target_curve",
        "approx_curve",
        "error_curve",
    ):
        pts = model_data.get(out_key) or []
        if pts and not obj.points_list:
            obj.points_list = [list(p) for p in pts]
    elif out_key in ("centroid_point", "spectrum_peak", "end_point", "body_point", "mechanical_body"):
        pt = model_data.get("body_point" if out_key == "mechanical_body" else out_key)
        if pt and len(pt) >= 2 and obj.coords is None:
            obj.coords = [float(pt[0]), float(pt[1])]
    elif out_key in ("centroid_vector", "restoring_force"):
        vec = model_data.get("centroid_vector")
        if vec and len(vec) == 2:
            if obj.coords is None:
                obj.coords = [float(vec[0][0]), float(vec[0][1])]
            if obj.end_coords is None:
                obj.end_coords = [float(vec[1][0]), float(vec[1][1])]
    elif out_key in ("spectrum_bars", "posterior_bars", "energy_bars"):
        labels = model_data.get(f"{out_key}_labels") or model_data.get("spectrum_bars_labels") or model_data.get("bar_labels") or []
        vals = model_data.get(f"{out_key}_values") or model_data.get("spectrum_bars_values") or model_data.get("bar_values") or []
        if vals and not obj.values:
            obj.values = [float(v) for v in vals]
        if labels and not obj.bar_labels:
            obj.bar_labels = [str(x) for x in labels]
    elif out_key == "posterior_table":
        if not obj.table_headers:
            obj.table_headers = list(model_data.get("table_headers") or [])
        if not obj.table_rows:
            obj.table_rows = [list(r) for r in (model_data.get("table_rows") or [])]
    elif out_key == "eigen_matrix":
        if not obj.matrix_rows:
            obj.matrix_rows = [list(r) for r in (model_data.get("matrix_rows") or [])]
    elif out_key == "readout":
        if not obj.text:
            obj.text = str(model_data.get("readout_text") or "")

    if not obj.sweep_keyframes and model_data.get("sweep_frames"):
        if out_key in (
            "winding_curve", "centroid_point", "centroid_vector", "restoring_force",
            "spectrum_peak", "phase_orbit", "mechanical_body", "body_point",
            "energy_bars", "trajectory", "loss_curve", "end_point", "readout",
        ):
            obj.sweep_keyframes = [sweep_cls.model_validate(kf) for kf in model_data["sweep_frames"]]
    if "selected_sweep_u" in model_data and out_key in (
        "winding_curve", "centroid_point", "centroid_vector", "restoring_force",
        "spectrum_peak", "phase_orbit", "mechanical_body", "energy_bars", "readout",
    ):
        obj.param_init = float(model_data["selected_sweep_u"])


def materialize_constructions(plan: "VideoPlanV3") -> "VideoPlanV3":
    """Deterministically materialize typed ConstructionOps into SceneObject definitions."""
    from studio.ai.linked_models import LinkedMathModelSpec, evaluate_linked_math_model, evaluate_signal_spectral_pipeline
    from studio.ai.math_truth import (
        bayes_posterior,
        definite_integral,
        discrete_pmf_convolution,
        extract_taylor_coefficients,
        fourier_partial_sum_terms,
        gradient_descent_trajectory,
        markov_chain_trajectory,
        matrix_transform_and_eigen,
        newton_raphson_trajectory,
        rk4_ode_trajectory,
        transform_points_2x2,
    )
    from studio.ai.v3_schema import SceneObject, SweepKeyframe

    out = plan.model_copy(deep=True)
    computed_models: dict[str, dict[str, Any]] = {}
    for m_spec in getattr(out, "math_models", []) or []:
        computed_models[m_spec.model_id] = evaluate_linked_math_model(m_spec)

    registry: dict[str, list[list[float]]] = {}
    obj_axes_map: dict[str, str | None] = {}

    for sc in out.scenes:
        for obj in sc.objects:
            if obj.model_id and obj.model_id in computed_models and obj.model_output:
                _bind_object_from_model(obj, computed_models[obj.model_id], SweepKeyframe)
            obj_axes_map[obj.id] = obj.axes_id
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
                support, probs, _, _ = discrete_pmf_convolution([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0 / 6.0] * 6, n_trials=n_dice)
                oid = c.output_ids[0] if c.output_ids else f"dice_{n_dice}_dist"
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="bar_chart", persist=c.persist, color_role=c.color_role, values=probs, bar_labels=[str(int(s)) for s in support], label=c.label or f"{n_dice}_dice_sum"))
                    existing_ids.add(oid)
            elif c.op == "discrete_pmf_convolution":
                supp = p.support_vals or [0.0, 1.0]
                prbs = p.prob_vals or [0.5, 0.5]
                nt = int(p.n_trials if p.n_trials is not None else 1)
                xs_conv, ps_conv, _, _ = discrete_pmf_convolution(supp, prbs, n_trials=nt)
                oid = c.output_ids[0] if c.output_ids else f"pmf_conv_{nt}"
                if oid not in existing_ids:
                    lbls = [str(int(x)) if abs(x - round(x)) < 1e-6 else f"{x:.2f}" for x in xs_conv]
                    sc.objects.append(SceneObject(id=oid, kind="bar_chart", persist=c.persist, color_role=c.color_role, values=ps_conv, bar_labels=lbls, label=c.label or f"conv_{nt}"))
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
            elif c.op == "fourier_general_synthesis":
                expr_in = p.expr or "x"
                period = float(p.period if p.period is not None else 2.0 * math.pi)
                n_harm = int(p.n_terms if p.n_terms is not None else 5)
                f_res = fourier_partial_sum_terms(expr_in, period=period, n_harmonics=n_harm)
                oid = c.output_ids[0] if c.output_ids else f"fourier_gen_{n_harm}"
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="function_plot", persist=c.persist, color_role=c.color_role, expr=str(f_res["expr"]), axes_id=p.axes_id, label=c.label or f"fourier_{n_harm}"))
                    existing_ids.add(oid)
            elif c.op == "riemann_integral":
                expr_in = p.expr or "x**2"
                a_b = float(p.a_bound if p.a_bound is not None else 0.0)
                b_b = float(p.b_bound if p.b_bound is not None else 2.0)
                n_r = int(p.n_rects if p.n_rects is not None else 8)
                meth = p.riemann_method or "midpoint"
                res = definite_integral(expr_in, a_b, b_b, n=n_r, method=meth)
                boxes = res["boxes"]  # type: ignore[assignment]
                for i, (rx, rw, ry) in enumerate(boxes):  # type: ignore[misc]
                    if abs(ry) <= 1e-6:
                        continue
                    oid = c.output_ids[i] if i < len(c.output_ids) else f"rect_{i}"
                    y0, y1 = (0.0, ry) if ry >= 0 else (ry, 0.0)
                    verts = [[rx, y0], [rx + rw, y0], [rx + rw, y1], [rx, y1]]
                    registry[oid] = verts
                    if oid not in existing_ids:
                        sc.objects.append(SceneObject(id=oid, kind="polygon", persist=c.persist, color_role=c.color_role, vertices=verts, axes_id=p.axes_id, label=f"riemann_{i}"))
                        existing_ids.add(oid)
            elif c.op == "ode_phase_trajectory":
                dx_e = p.dx_expr or "y"
                dy_e = p.dy_expr or "-x - 0.25*y"
                init_s = p.initial_state or []
                x0 = float(p.x0 if p.x0 is not None else (init_s[0] if len(init_s) >= 1 else 1.5))
                y0 = float(p.y0 if p.y0 is not None else (init_s[1] if len(init_s) >= 2 else 0.0))
                dt = float(p.dt if p.dt is not None else 0.08)
                st = int(p.steps if p.steps is not None else 60)
                traj = rk4_ode_trajectory(dx_e, dy_e, x0=x0, y0=y0, dt=dt, steps=st)
                pts = [[xt, yt] for (_, xt, yt) in traj]
                oid = next((cand for cand in c.output_ids if cand not in existing_ids), c.output_ids[0] if c.output_ids else "ode_orbit")
                ax_ref = p.axes_id or next((cand for cand in c.output_ids if any(o.id == cand and o.kind == "axes" for o in sc.objects)), None)
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="data_series", persist=c.persist, color_role=c.color_role, points_list=pts, axes_id=ax_ref, label=c.label or "phase_orbit"))
                    existing_ids.add(oid)
            elif c.op == "linear_transform_2x2":
                mat = p.matrix_2x2 or [[1.0, 0.0], [0.0, 1.0]]
                src_verts = p.vertices_list or registry.get(c.source_id or "", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
                t_verts = transform_points_2x2(src_verts, mat)
                cleaned = clean_polygon_vertices(t_verts)
                oid = c.output_ids[0] if c.output_ids else "transformed_shape"
                ax_ref = p.axes_id or obj_axes_map.get(c.source_id or "")
                registry[oid] = cleaned
                obj_axes_map[oid] = ax_ref
                if oid not in existing_ids:
                    sc.objects.append(SceneObject(id=oid, kind="polygon", persist=c.persist, color_role=c.color_role, vertices=cleaned, axes_id=ax_ref, label=c.label or "linear_image"))
                    existing_ids.add(oid)
                elif ax_ref:
                    for existing_o in sc.objects:
                        if existing_o.id == oid and existing_o.axes_id is None:
                            existing_o.axes_id = ax_ref
                            obj_axes_map[oid] = ax_ref
            elif c.op == "eigen_decomposition_2x2":
                mat = p.matrix_2x2 or [[2.0, 1.0], [1.0, 2.0]]
                eig = matrix_transform_and_eigen(mat)
                mat_id = c.output_ids[0] if c.output_ids else "eigen_matrix"
                if mat_id not in existing_ids:
                    rows = [[f"{mat[0][0]:.2g}", f"{mat[0][1]:.2g}"], [f"{mat[1][0]:.2g}", f"{mat[1][1]:.2g}"]]
                    sc.objects.append(SceneObject(id=mat_id, kind="matrix", persist=c.persist, color_role=c.color_role, matrix_rows=rows, label=str(eig["regime"])))
                    existing_ids.add(mat_id)
                vecs = eig["real_eigenvectors"]  # type: ignore[assignment]
                vals = eig["real_eigenvalues"]  # type: ignore[assignment]
                for i, (vec, val) in enumerate(zip(vecs, vals)):  # type: ignore[arg-type]
                    vid = c.output_ids[i + 1] if (i + 1) < len(c.output_ids) else f"eigvec_{i}"
                    if vid not in existing_ids:
                        sc.objects.append(SceneObject(id=vid, kind="vector", persist=c.persist, color_role="highlight" if i == 0 else "secondary", coords=[0.0, 0.0], end_coords=[vec[0] * val, vec[1] * val], axes_id=p.axes_id, label=f"lambda={val:.2f}"))
                        existing_ids.add(vid)
            elif c.op == "newton_root_path":
                expr_in = p.expr or "x**2 - 2"
                x0 = float(p.x0 if p.x0 is not None else 2.5)
                steps = int(p.steps if p.steps is not None else 4)
                traj = newton_raphson_trajectory(expr_in, x0, steps=steps)
                for idx_pt, (xv, yv, _) in enumerate(traj):
                    oid = c.output_ids[idx_pt] if idx_pt < len(c.output_ids) else f"newton_pt_{idx_pt}"
                    if oid not in existing_ids:
                        sc.objects.append(SceneObject(id=oid, kind="point", persist=c.persist, color_role="highlight" if idx_pt == len(traj) - 1 else c.color_role, coords=[xv, yv], axes_id=p.axes_id, label=f"iter_{idx_pt}"))
                        existing_ids.add(oid)
            elif c.op == "markov_chain_evolution":
                P = p.transition_matrix or [[0.8, 0.2], [0.3, 0.7]]
                p0 = p.initial_state or [1.0 / len(P)] * len(P)
                steps = int(p.steps if p.steps is not None else 8)
                mk = markov_chain_trajectory(P, p0, steps=steps)
                oid = c.output_ids[0] if c.output_ids else "markov_stationary"
                stat = mk["stationary"]  # type: ignore[assignment]
                if oid not in existing_ids:
                    lbls = [f"S{i}" for i in range(len(stat))]  # type: ignore[arg-type]
                    sc.objects.append(
                        SceneObject(
                            id=oid,
                            kind="bar_chart",
                            persist=c.persist,
                            color_role=c.color_role,
                            values=stat,  # type: ignore[arg-type]
                            bar_labels=lbls,
                            points_list=list(mk["states"]),  # type: ignore[arg-type]
                            label=c.label or "stationary_dist",
                        )
                    )
                    existing_ids.add(oid)
            elif c.op in ("winding_complex_curve", "spectral_transform_from_signal", "filtered_signal_reconstruction"):
                spec = LinkedMathModelSpec(
                    model_id=c.source_id or f"_op_{c.op}",
                    kind="signal_spectral_pipeline",
                    expr=p.expr or "0.85*cos(2*pi*2*t) + 0.55*cos(2*pi*5*t)",
                    t_range=p.t_range or [0.0, 2.0],
                    freq_range=p.freq_range or [0.0, 7.0],
                    selected_freq=p.selected_freq,
                    cutoff_freq=p.cutoff_freq,
                    high_cutoff_freq=p.high_cutoff_freq,
                    filter_kind=p.filter_kind or "lowpass",
                    angular_units=p.angular_units or "hz",
                )
                pipe = computed_models.get(c.source_id or "") or evaluate_signal_spectral_pipeline(spec)
                if c.op == "winding_complex_curve":
                    curve_id = c.output_ids[0] if len(c.output_ids) >= 1 else "winding_curve"
                    pt_id = c.output_ids[1] if len(c.output_ids) >= 2 else "winding_centroid"
                    vec_id = c.output_ids[2] if len(c.output_ids) >= 3 else "winding_vector"
                    kfs = [SweepKeyframe.model_validate(kf) for kf in pipe.get("sweep_frames", [])]
                    if curve_id not in existing_ids:
                        sc.objects.append(SceneObject(id=curve_id, kind="data_series", persist=c.persist, color_role=c.color_role, points_list=pipe["winding_curve"], axes_id=p.axes_id, sweep_keyframes=kfs, label=c.label or "z(t)=x(t)e^{-iwt}"))
                        existing_ids.add(curve_id)
                    if pt_id not in existing_ids:
                        sc.objects.append(SceneObject(id=pt_id, kind="point", persist=c.persist, color_role="highlight", coords=pipe["centroid_point"], axes_id=p.axes_id, sweep_keyframes=kfs, label="C(w)"))
                        existing_ids.add(pt_id)
                    if len(c.output_ids) >= 3 and vec_id not in existing_ids:
                        vec = pipe["centroid_vector"]
                        sc.objects.append(SceneObject(id=vec_id, kind="vector", persist=c.persist, color_role="highlight", coords=vec[0], end_coords=vec[1], axes_id=p.axes_id, sweep_keyframes=kfs, label="C_vec"))
                        existing_ids.add(vec_id)
                elif c.op == "spectral_transform_from_signal":
                    spec_id = c.output_ids[0] if c.output_ids else "derived_spectrum"
                    if spec_id not in existing_ids:
                        sc.objects.append(SceneObject(id=spec_id, kind="data_series", persist=c.persist, color_role=c.color_role, points_list=pipe["spectrum"], axes_id=p.axes_id, label=c.label or "|F(w)|"))
                        existing_ids.add(spec_id)
                elif c.op == "filtered_signal_reconstruction":
                    filt_id = c.output_ids[0] if len(c.output_ids) >= 1 else "filtered_signal"
                    if filt_id not in existing_ids:
                        sc.objects.append(SceneObject(id=filt_id, kind="data_series", persist=c.persist, color_role=c.color_role, points_list=pipe["filtered_signal"], axes_id=p.axes_id, label=c.label or "x_filtered(t)"))
                        existing_ids.add(filt_id)
                    if len(c.output_ids) >= 2 and c.output_ids[1] not in existing_ids:
                        fspec_id = c.output_ids[1]
                        sc.objects.append(SceneObject(id=fspec_id, kind="data_series", persist=c.persist, color_role="secondary", points_list=pipe["filtered_spectrum"], axes_id=p.axes_id, label="F_filtered(w)"))
                        existing_ids.add(fspec_id)
    return out
