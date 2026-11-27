"""Pre-render state-simulating validator for VideoPlan V3."""

from __future__ import annotations

import math
import re
import numpy as np

from studio.ai.expr import ExprError, compile_expr
from studio.ai.math_truth import (
    bayes_posterior,
    definite_integral,
    distribution_stats,
    eval_function,
    extract_taylor_coefficients,
    functions_equivalent,
    is_local_minimum,
    is_stationary_point,
    matrix_transform_and_eigen,
    is_row_stochastic_matrix,
    symbolic_derivative,
    verify_eigenvalue_stability_regime,
    verify_graph_stochastic_weights,
    verify_polynomial_degree,
    verify_stationary_distribution,
)
from studio.ai.schema import BOOKMARK_RE
from studio.ai.v3_schema import SceneObject, VideoPlanV3
from studio.ai.world_model import (
    all_coords_finite,
    clean_polygon_vertices,
    has_self_intersection,
    is_right_triangle,
    is_square,
    materialize_constructions,
    polygon_area,
    rotate_polygon,
    translate_polygon,
    triangles_congruent,
    verify_partition,
)


def _finite_seq(vals: list[float] | None) -> bool:
    return vals is None or all(math.isfinite(float(v)) for v in vals)


def verify_v3_plan_detailed(plan: VideoPlanV3, require_claims: bool = False) -> dict[str, object]:
    """Simulate object lifecycle + post-transform math state; return issues and claim evidence."""
    issues: list[str] = []
    verified_claims: list[str] = []
    failed_claims: list[str] = []
    unsupported_claims: list[str] = []

    if plan.schema_version != "3.0":
        issues.append(f"Expected schema_version '3.0', got {plan.schema_version!r}.")
    if not plan.scenes:
        issues.append("VideoPlanV3 must contain at least one scene.")
        return {"issues": issues, "verified": verified_claims, "failed": failed_claims, "unsupported": unsupported_claims}

    try:
        plan = materialize_constructions(plan)
    except ValueError as exc:
        issues.append(f"Construction error: {exc}")
        return {"issues": issues, "verified": verified_claims, "failed": failed_claims, "unsupported": unsupported_claims}

    state: dict[str, str] = {}
    live_objs: dict[str, SceneObject] = {}
    live_polys: dict[str, list[list[float]]] = {}
    seen_scenes: set[str] = set()
    total_claims = 0

    for idx, sc in enumerate(plan.scenes, 1):
        where = f"scene {idx} ({sc.scene_id})"
        if not sc.scene_id or sc.scene_id in seen_scenes:
            issues.append(f"{where}: scene_id must be non-empty and unique.")
        seen_scenes.add(sc.scene_id)
        if not sc.narration.strip():
            issues.append(f"{where}: narration must not be empty.")

        raw_marks = BOOKMARK_RE.findall(sc.narration)
        if len(raw_marks) != len(set(raw_marks)):
            issues.append(f"{where}: duplicate bookmark names in narration.")
        marks = set(raw_marks)
        mark_order = {m_name: i for i, m_name in enumerate(raw_marks)}

        scene_declared: set[str] = set()
        for obj in sc.objects:
            if not obj.id or not obj.id.replace("_", "").isalnum():
                issues.append(f"{where}: invalid object id {obj.id!r}.")
            if obj.id in scene_declared:
                issues.append(f"{where}: duplicate object id {obj.id!r} in same scene.")
            if state.get(obj.id) in ("declared", "visible"):
                issues.append(f"{where}: object {obj.id!r} already exists as persistent; cannot redeclare without removing first.")
            scene_declared.add(obj.id)
            state[obj.id] = "declared"
            live_objs[obj.id] = obj.model_copy(deep=True)

            for seq_name, seq in (("coords", obj.coords), ("end_coords", obj.end_coords), ("x_range", obj.x_range), ("y_range", obj.y_range), ("t_range", obj.t_range)):
                if not _finite_seq(seq):
                    issues.append(f"{where} obj {obj.id}: {seq_name} contains non-finite values.")
            if obj.radius is not None and (not math.isfinite(obj.radius) or obj.radius <= 0):
                issues.append(f"{where} obj {obj.id}: circle/arc radius must be finite and positive.")

            if obj.axes_id and state.get(obj.axes_id) not in ("declared", "visible"):
                issues.append(f"{where} obj {obj.id}: axes_id {obj.axes_id!r} is not live.")

            if obj.kind == "function_plot":
                if not obj.expr:
                    issues.append(f"{where} obj {obj.id}: function_plot requires expr.")
                else:
                    try:
                        compile_expr(obj.expr, ("x",))
                    except ExprError as exc:
                        issues.append(f"{where} obj {obj.id}: {exc}")
            elif obj.kind == "parametric_curve":
                if not obj.expr or ";" not in obj.expr:
                    issues.append(f"{where} obj {obj.id}: parametric_curve requires 'X(t); Y(t)'.")
                else:
                    try:
                        p0, p1 = obj.expr.split(";", 1)
                        compile_expr(p0, ("t",))
                        compile_expr(p1, ("t",))
                    except ExprError as exc:
                        issues.append(f"{where} obj {obj.id}: {exc}")
            elif obj.kind == "bar_chart":
                if not obj.values or any(not np.isfinite(v) for v in obj.values):
                    issues.append(f"{where} obj {obj.id}: bar_chart requires finite non-empty values.")
                if obj.bar_labels and len(obj.bar_labels) != len(obj.values):
                    issues.append(f"{where} obj {obj.id}: bar_labels length must match values.")
                if obj.log_scale and any(v <= 0 for v in obj.values):
                    issues.append(f"{where} obj {obj.id}: log_scale requires strictly positive values.")
            elif obj.kind == "polygon":
                if not all_coords_finite(obj.vertices):
                    issues.append(f"{where} obj {obj.id}: polygon vertices contain non-finite or malformed coordinates.")
                    continue
                cleaned = clean_polygon_vertices(obj.vertices)
                if len(obj.vertices) != len(cleaned):
                    issues.append(f"{where} obj {obj.id}: polygon has redundant closing duplicate vertex.")
                if len(cleaned) < 3 or polygon_area(cleaned) <= 1e-6:
                    issues.append(f"{where} obj {obj.id}: polygon must have >= 3 distinct vertices and nonzero area.")
                if has_self_intersection(cleaned):
                    issues.append(f"{where} obj {obj.id}: polygon has self-intersecting edges.")
                tag = f"{obj.id} {obj.label}".lower()
                if "square" in tag and not is_square(cleaned):
                    issues.append(f"{where} obj {obj.id}: claimed square fails geometric square check (4 equal sides & 4 right angles).")
                if "triangle" in tag and len(cleaned) != 3:
                    issues.append(f"{where} obj {obj.id}: claimed triangle must have 3 vertices (got {len(cleaned)}).")
                live_polys[obj.id] = cleaned
            elif obj.kind in ("data_series", "particle_system", "waveform", "graph_network"):
                if not all_coords_finite(obj.points_list):
                    issues.append(f"{where} obj {obj.id}: points_list contains non-finite coordinates.")
            elif obj.kind == "matrix":
                if not obj.matrix_rows:
                    issues.append(f"{where} obj {obj.id}: matrix requires non-empty matrix_rows.")
            elif obj.kind == "table":
                if not (obj.table_headers or obj.table_rows):
                    issues.append(f"{where} obj {obj.id}: table requires table_headers or table_rows.")

        # Validate event dependencies, ordering, and simulate geometry transformations
        has_unanchored_main = any(not e.anchor_mark for e in sc.events)
        if not has_unanchored_main and sc.objects:
            anchored_ids = {e.target_id for e in sc.events if e.anchor_mark}
            for o in sc.objects:
                if o.id not in anchored_ids and state.get(o.id) == "declared":
                    state[o.id] = "visible"

        last_mark_idx = -1
        for ev in sc.events:
            st = state.get(ev.target_id)
            if ev.op != "duplicate" and st not in ("declared", "visible"):
                issues.append(f"{where}: event target_id {ev.target_id!r} is not live (missing or removed).")
            if ev.op in ("move", "rotate", "scale", "highlight", "focus", "remove", "animate_param", "zoom", "iterate_states", "interpolate_values") and st == "declared":
                issues.append(f"{where}: event {ev.op} on {ev.target_id!r} before it is created/drawn.")
            if ev.op == "rotate" and st in ("declared", "visible") and ev.target_id in live_objs:
                t_kind = live_objs[ev.target_id].kind
                if t_kind in ("matrix", "table", "equation", "axes", "annotation", "bar_chart"):
                    issues.append(f"{where}: cannot rotate non-geometric object {ev.target_id!r} of kind {t_kind!r}.")
            if ev.op in ("create", "draw", "reveal", "trace", "sequence", "parallel", "group"):
                state[ev.target_id] = "visible"
            elif ev.op in ("transform", "morph", "duplicate"):
                if not ev.source_id or state.get(ev.source_id) not in ("declared", "visible"):
                    issues.append(f"{where}: {ev.op} requires live source_id (got {ev.source_id!r}).")
                state[ev.target_id] = "visible"
                if ev.op == "duplicate" and ev.source_id in live_objs:
                    live_objs.setdefault(ev.target_id, live_objs[ev.source_id].model_copy(update={"id": ev.target_id}))
                    if ev.source_id in live_polys:
                        live_polys.setdefault(ev.target_id, [list(p) for p in live_polys[ev.source_id]])
            elif ev.op == "move" and ev.shift and ev.target_id in live_polys:
                live_polys[ev.target_id] = translate_polygon(live_polys[ev.target_id], float(ev.shift[0]), float(ev.shift[1]))
            elif ev.op == "rotate" and ev.angle_deg and ev.target_id in live_polys:
                pts = live_polys[ev.target_id]
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                live_polys[ev.target_id] = rotate_polygon(pts, math.radians(float(ev.angle_deg)), (cx, cy))
                if has_self_intersection(live_polys[ev.target_id]):
                    issues.append(f"{where}: rotated polygon {ev.target_id!r} became self-intersecting.")

            if not _finite_seq(ev.shift) or not _finite_seq(ev.param_range):
                issues.append(f"{where}: event {ev.op} on {ev.target_id!r} has non-finite shift/param_range.")
            if ev.scale_factor is not None and not math.isfinite(ev.scale_factor):
                issues.append(f"{where}: event scale on {ev.target_id!r} has non-finite scale_factor.")
            if ev.angle_deg is not None and not math.isfinite(ev.angle_deg):
                issues.append(f"{where}: event rotate on {ev.target_id!r} has non-finite angle_deg.")
            if ev.op == "animate_param" and (not ev.param_range or len(ev.param_range) != 2):
                issues.append(f"{where}: animate_param requires param_range=[start, end].")
            if ev.anchor_mark:
                if ev.anchor_mark not in marks:
                    issues.append(f"{where}: anchor_mark {ev.anchor_mark!r} missing from narration.")
                elif mark_order[ev.anchor_mark] < last_mark_idx:
                    issues.append(f"{where}: anchor_mark {ev.anchor_mark!r} appears out of narration order.")
                else:
                    last_mark_idx = mark_order[ev.anchor_mark]
            if ev.op == "remove":
                state[ev.target_id] = "removed"

        # Structured mathematical claim verification (evaluated on post-event state)
        for cl in sc.claims:
            total_claims += 1
            missing = [oid for oid in cl.object_ids if oid not in live_objs]
            if missing:
                msg = f"{where} claim {cl.kind}: references missing object(s) {missing}."
                issues.append(msg)
                failed_claims.append(msg)
                continue

            if cl.kind in ("right_triangle", "square", "congruent_triangles", "nonzero_area", "partition_of", "area_conservation", "area_equality"):
                non_polys = [oid for oid in cl.object_ids if live_objs[oid].kind != "polygon"]
                if non_polys:
                    msg = f"{where} claim {cl.kind}: incompatible non-polygon object(s) {non_polys}."
                    issues.append(msg)
                    failed_claims.append(msg)
                    continue
                polys = [live_polys[oid] for oid in cl.object_ids if oid in live_polys]
                if cl.kind == "right_triangle":
                    bad = [oid for oid, verts in zip(cl.object_ids, polys) if not is_right_triangle(verts)]
                    if bad:
                        msg = f"{where} claim right_triangle: {bad} not valid right triangle(s)."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: right_triangle({cl.object_ids})")
                elif cl.kind == "square":
                    bad = [oid for oid, verts in zip(cl.object_ids, polys) if not is_square(verts)]
                    if bad:
                        msg = f"{where} claim square: {bad} not valid square(s)."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: square({cl.object_ids})")
                elif cl.kind == "congruent_triangles":
                    if len(polys) < 2 or any(not triangles_congruent(polys[0], v) for v in polys[1:]):
                        msg = f"{where} claim congruent_triangles: triangles {cl.object_ids} are not congruent."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: congruent_triangles({cl.object_ids})")
                elif cl.kind == "nonzero_area":
                    if any(polygon_area(v) <= 1e-6 for v in polys):
                        msg = f"{where} claim nonzero_area: zero-area polygon in {cl.object_ids}."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: nonzero_area({cl.object_ids})")
                elif cl.kind == "partition_of":
                    cid = cl.container_id
                    if not cid or cid not in live_polys:
                        msg = f"{where} claim partition_of: container_id {cid!r} is not a live polygon."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        p_issues = verify_partition(polys, live_polys[cid])
                        if p_issues:
                            for pi in p_issues:
                                issues.append(f"{where} claim partition_of: {pi}")
                            failed_claims.append(f"{where}: partition_of({cl.object_ids} in {cid})")
                        else:
                            verified_claims.append(f"{where}: partition_of({cl.object_ids} in {cid})")
                elif cl.kind in ("area_conservation", "area_equality"):
                    cid = cl.container_id
                    if cid and cid in live_polys:
                        total_pieces = sum(polygon_area(pc) for pc in polys)
                        target_area = polygon_area(live_polys[cid])
                        if abs(total_pieces - target_area) > cl.tolerance:
                            msg = f"{where} claim {cl.kind}: sum={total_pieces:.4f} vs {cid}={target_area:.4f}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: {cl.kind}({cl.object_ids} == {cid})")
                    elif cl.expected_value is not None:
                        total_pieces = sum(polygon_area(pc) for pc in polys)
                        if abs(total_pieces - cl.expected_value) > cl.tolerance:
                            msg = f"{where} claim {cl.kind}: sum={total_pieces:.4f} vs expected={cl.expected_value:.4f}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: {cl.kind}({cl.object_ids} == {cl.expected_value})")
                    else:
                        msg = f"{where} claim {cl.kind}: requires container_id or expected_value."
                        issues.append(msg); unsupported_claims.append(msg)

            elif cl.kind == "point_on_function":
                pts = [live_objs[oid] for oid in cl.object_ids if live_objs[oid].kind == "point"]
                fns = [live_objs[oid] for oid in cl.object_ids if live_objs[oid].kind == "function_plot"]
                expr = cl.expr or (fns[0].expr if fns else None)
                if not pts or not expr:
                    msg = f"{where} claim point_on_function: requires a point object and function_plot/expr."
                    issues.append(msg); failed_claims.append(msg)
                else:
                    ok_all = True
                    for pt in pts:
                        if not pt.coords or len(pt.coords) < 2:
                            ok_all = False; break
                        try:
                            yv = eval_function(expr, pt.coords[0])
                        except Exception as exc:
                            ok_all = False
                            msg = f"{where} claim point_on_function: invalid expr {expr!r} ({exc})."
                            issues.append(msg); failed_claims.append(msg)
                            break
                        if abs(yv - pt.coords[1]) > cl.tolerance:
                            ok_all = False
                            msg = f"{where} claim point_on_function: point {pt.id}={pt.coords} does not lie on {expr} (f(x)={yv:.4f})."
                            issues.append(msg); failed_claims.append(msg)
                    if ok_all:
                        verified_claims.append(f"{where}: point_on_function({cl.object_ids})")

            elif cl.kind in ("stationary_point", "local_minimum"):
                expr = cl.expr or next((live_objs[oid].expr for oid in cl.object_ids if live_objs[oid].kind == "function_plot" and live_objs[oid].expr), None)
                xv = cl.x_val
                if xv is None:
                    pt = next((live_objs[oid] for oid in cl.object_ids if live_objs[oid].kind == "point" and live_objs[oid].coords), None)
                    xv = pt.coords[0] if pt and pt.coords else None
                if not expr or xv is None:
                    msg = f"{where} claim {cl.kind}: requires expr and x_val (or point)."
                    issues.append(msg); failed_claims.append(msg)
                else:
                    try:
                        ok = is_stationary_point(expr, xv, tol=cl.tolerance) if cl.kind == "stationary_point" else is_local_minimum(expr, xv, tol=cl.tolerance)
                        if not ok:
                            msg = f"{where} claim {cl.kind}: x={xv} is not a valid {cl.kind} of {expr}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: {cl.kind}({expr} at x={xv})")
                    except Exception as exc:
                        msg = f"{where} claim {cl.kind}: invalid expr {expr!r} ({exc})."
                        issues.append(msg); failed_claims.append(msg)

            elif cl.kind == "derivative_equality":
                if not cl.expr or not cl.expected_expr:
                    msg = f"{where} claim derivative_equality: requires expr and expected_expr."
                    issues.append(msg); failed_claims.append(msg)
                else:
                    try:
                        d = symbolic_derivative(cl.expr)
                        if not functions_equivalent(d, cl.expected_expr):
                            msg = f"{where} claim derivative_equality: d/dx({cl.expr})={d} != {cl.expected_expr}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: derivative_equality({cl.expr})")
                    except Exception as exc:
                        msg = f"{where} claim derivative_equality: invalid expression ({exc})."
                        issues.append(msg); failed_claims.append(msg)

            elif cl.kind == "function_equivalence":
                try:
                    if not cl.expr or not cl.expected_expr or not functions_equivalent(cl.expr, cl.expected_expr):
                        msg = f"{where} claim function_equivalence: {cl.expr} != {cl.expected_expr}."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: function_equivalence({cl.expr})")
                except Exception as exc:
                    msg = f"{where} claim function_equivalence: invalid expression ({exc})."
                    issues.append(msg); failed_claims.append(msg)

            elif cl.kind == "polynomial_degree":
                expr = cl.expr or next((live_objs[oid].expr for oid in cl.object_ids if live_objs[oid].expr), None)
                try:
                    if not expr or cl.expected_degree is None or not verify_polynomial_degree(expr, cl.expected_degree):
                        msg = f"{where} claim polynomial_degree: {expr} does not have degree {cl.expected_degree}."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: polynomial_degree({cl.expected_degree})")
                except Exception as exc:
                    msg = f"{where} claim polynomial_degree: {exc}"
                    issues.append(msg); failed_claims.append(msg)

            elif cl.kind == "coefficient_correctness":
                if not cl.expr or cl.expected_degree is None or cl.expected_value is None:
                    msg = f"{where} claim coefficient_correctness: requires expr, expected_degree, and expected_value."
                    issues.append(msg); unsupported_claims.append(msg)
                else:
                    try:
                        coeffs = extract_taylor_coefficients(cl.expr, cl.expected_degree)
                        if abs(coeffs[cl.expected_degree] - cl.expected_value) > cl.tolerance:
                            msg = f"{where} claim coefficient_correctness: coeff[{cl.expected_degree}]={coeffs[cl.expected_degree]:.5f} != {cl.expected_value}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: coefficient_correctness({cl.expr})")
                    except Exception as exc:
                        msg = f"{where} claim coefficient_correctness: invalid scalar expression {cl.expr!r} ({exc})."
                        issues.append(msg); failed_claims.append(msg)

            elif cl.kind == "probability_normalization":
                bars = [live_objs[oid] for oid in cl.object_ids if live_objs[oid].kind == "bar_chart"]
                if not bars:
                    msg = f"{where} claim probability_normalization: requires a bar_chart object."
                    issues.append(msg); failed_claims.append(msg)
                else:
                    s = sum(bars[0].values)
                    if abs(s - 1.0) > max(cl.tolerance, 1e-3) or any(v < 0 for v in bars[0].values):
                        msg = f"{where} claim probability_normalization: values sum to {s:.5f} (expected 1.0)."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: probability_normalization({bars[0].id})")

            elif cl.kind == "conditional_probability":
                if cl.prior is None or cl.likelihood is None or cl.false_positive is None or cl.expected_value is None:
                    msg = f"{where} claim conditional_probability: requires prior, likelihood, false_positive, expected_value."
                    issues.append(msg); unsupported_claims.append(msg)
                else:
                    res = bayes_posterior(cl.prior, cl.likelihood, cl.false_positive)
                    if abs(res["posterior"] - cl.expected_value) > max(cl.tolerance, 1e-3):
                        msg = f"{where} claim conditional_probability: posterior={res['posterior']:.4f} != {cl.expected_value:.4f}."
                        issues.append(msg); failed_claims.append(msg)
                    elif abs(res["delta"]) < 1e-4:
                        msg = f"{where} claim conditional_probability: evidence is independent of hypothesis (posterior == prior == {cl.prior:.4f})."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: conditional_probability(posterior={res['posterior']:.4f})")

            elif cl.kind == "distribution_stats":
                bars = [live_objs[oid] for oid in cl.object_ids if live_objs[oid].kind == "bar_chart"]
                if not bars or not bars[0].bar_labels or cl.mu is None or cl.sigma is None:
                    msg = f"{where} claim distribution_stats: requires labeled bar_chart, mu, and sigma."
                    issues.append(msg); unsupported_claims.append(msg)
                else:
                    try:
                        supp = [float(x) for x in bars[0].bar_labels]
                        mu_calc, sig_calc = distribution_stats(supp, bars[0].values)
                        if abs(mu_calc - cl.mu) > max(cl.tolerance, 1e-2) or abs(sig_calc - cl.sigma) > max(cl.tolerance, 1e-2):
                            msg = f"{where} claim distribution_stats: got (mu={mu_calc:.4f}, sigma={sig_calc:.4f}) != ({cl.mu}, {cl.sigma})."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: distribution_stats(mu={mu_calc:.2f}, sigma={sig_calc:.2f})")
                    except ValueError as exc:
                        msg = f"{where} claim distribution_stats: {exc}"
                        issues.append(msg); failed_claims.append(msg)

            elif cl.kind == "convergence_assumptions":
                if not cl.statement.strip():
                    msg = f"{where} claim convergence_assumptions: statement must explicitly specify assumptions."
                    issues.append(msg); failed_claims.append(msg)
                else:
                    verified_claims.append(f"{where}: convergence_assumptions")

            elif cl.kind == "definite_integral_value":
                if not cl.expr or cl.a_bound is None or cl.b_bound is None or cl.expected_value is None:
                    msg = f"{where} claim definite_integral_value: requires expr, a_bound, b_bound, expected_value."
                    issues.append(msg); unsupported_claims.append(msg)
                else:
                    try:
                        res = definite_integral(cl.expr, cl.a_bound, cl.b_bound)
                        val = float(res["integral_exact"])  # type: ignore[arg-type]
                        if abs(val - cl.expected_value) > max(cl.tolerance, 1e-2):
                            msg = f"{where} claim definite_integral_value: got {val:.5f} != {cl.expected_value}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: definite_integral_value({val:.4f})")
                    except Exception as exc:
                        msg = f"{where} claim definite_integral_value: {exc}"
                        issues.append(msg); failed_claims.append(msg)

            elif cl.kind == "eigenvalue_determinant":
                if not cl.matrix_2x2 or cl.expected_value is None:
                    msg = f"{where} claim eigenvalue_determinant: requires matrix_2x2 and expected_value (determinant)."
                    issues.append(msg); unsupported_claims.append(msg)
                else:
                    try:
                        eig = matrix_transform_and_eigen(cl.matrix_2x2)
                        det_v = float(eig["determinant"])  # type: ignore[arg-type]
                        if abs(det_v - cl.expected_value) > max(cl.tolerance, 1e-3):
                            msg = f"{where} claim eigenvalue_determinant: det={det_v:.4f} != {cl.expected_value}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: eigenvalue_determinant(det={det_v:.4f})")
                    except Exception as exc:
                        msg = f"{where} claim eigenvalue_determinant: {exc}"
                        issues.append(msg); failed_claims.append(msg)

            elif cl.kind == "eigenvalue_stability_regime":
                if not cl.matrix_2x2:
                    msg = f"{where} claim eigenvalue_stability_regime: requires matrix_2x2."
                    issues.append(msg); unsupported_claims.append(msg)
                else:
                    try:
                        ok_reg, reg_detail = verify_eigenvalue_stability_regime(
                            cl.matrix_2x2,
                            expected_regime=cl.expected_regime,
                            statement=cl.statement or sc.narration,
                        )
                        if not ok_reg:
                            msg = f"{where} claim eigenvalue_stability_regime: {reg_detail}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: eigenvalue_stability_regime({reg_detail})")
                    except Exception as exc:
                        msg = f"{where} claim eigenvalue_stability_regime: {exc}"
                        issues.append(msg); failed_claims.append(msg)

            elif cl.kind == "ode_equilibrium":
                if not cl.dx_expr or not cl.dy_expr:
                    msg = f"{where} claim ode_equilibrium: requires dx_expr and dy_expr."
                    issues.append(msg); unsupported_claims.append(msg)
                else:
                    try:
                        fx = compile_expr(cl.dx_expr, ("x", "y", "t"))
                        fy = compile_expr(cl.dy_expr, ("x", "y", "t"))
                        xv = float(cl.x_val or 0.0)
                        yv = float(cl.expected_value or 0.0)
                        dx_v, dy_v = float(fx(x=xv, y=yv, t=0.0)), float(fy(x=xv, y=yv, t=0.0))
                        if math.hypot(dx_v, dy_v) > max(cl.tolerance, 1e-3):
                            msg = f"{where} claim ode_equilibrium: |({dx_v:.4f}, {dy_v:.4f})| != 0 at ({xv}, {yv})."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: ode_equilibrium({xv}, {yv})")
                    except Exception as exc:
                        msg = f"{where} claim ode_equilibrium: {exc}"
                        issues.append(msg); failed_claims.append(msg)

            elif cl.kind == "markov_stochastic_invariant":
                mats = [live_objs[oid] for oid in cl.object_ids if live_objs[oid].kind == "matrix"]
                bars = [live_objs[oid] for oid in cl.object_ids if live_objs[oid].kind == "bar_chart"]
                graphs = [live_objs[oid] for oid in cl.object_ids if live_objs[oid].kind == "graph_network"]
                mk_con = next((c for c in sc.constructions if c.op == "markov_chain_evolution" and c.params.transition_matrix), None)
                checked_any = False
                mk_failed = False

                for mat_obj in mats:
                    try:
                        p_mat = [[float(cell) for cell in row] for row in mat_obj.matrix_rows]
                    except ValueError:
                        p_mat = []
                    if p_mat:
                        ok_m, m_msg = is_row_stochastic_matrix(p_mat, tol=max(cl.tolerance, 2e-2))
                        checked_any = True
                        if not ok_m:
                            msg = f"{where} claim markov_stochastic_invariant ({mat_obj.id}): {m_msg}."
                            issues.append(msg); failed_claims.append(msg); mk_failed = True
                for bar_obj in bars:
                    checked_any = True
                    if mk_con and mk_con.params.transition_matrix and bar_obj.values:
                        ok_s, s_msg = verify_stationary_distribution(mk_con.params.transition_matrix, bar_obj.values, tol=max(cl.tolerance, 2e-2))
                        if not ok_s:
                            msg = f"{where} claim markov_stochastic_invariant ({bar_obj.id}): {s_msg}."
                            issues.append(msg); failed_claims.append(msg); mk_failed = True
                    elif abs(sum(bar_obj.values) - 1.0) > max(cl.tolerance, 2e-2) or any(v < -1e-6 for v in bar_obj.values):
                        msg = f"{where} claim markov_stochastic_invariant ({bar_obj.id}): distribution sum={sum(bar_obj.values):.4f} != 1.0."
                        issues.append(msg); failed_claims.append(msg); mk_failed = True
                for g_obj in graphs:
                    checked_any = True
                    ok_g, g_msg = verify_graph_stochastic_weights(len(g_obj.points_list), g_obj.edge_pairs, tol=max(cl.tolerance, 5e-2))
                    if not ok_g:
                        msg = f"{where} claim markov_stochastic_invariant ({g_obj.id}): {g_msg}."
                        issues.append(msg); failed_claims.append(msg); mk_failed = True

                if not checked_any and mk_con and mk_con.params.transition_matrix:
                    ok_m, m_msg = is_row_stochastic_matrix(mk_con.params.transition_matrix, tol=max(cl.tolerance, 2e-2))
                    checked_any = True
                    if not ok_m:
                        msg = f"{where} claim markov_stochastic_invariant: {m_msg}."
                        issues.append(msg); failed_claims.append(msg); mk_failed = True

                if not checked_any:
                    msg = f"{where} claim markov_stochastic_invariant: no verifiable stochastic matrix, distribution bar_chart, or weighted graph linked in object_ids."
                    issues.append(msg); unsupported_claims.append(msg)
                elif not mk_failed:
                    verified_claims.append(f"{where}: markov_stochastic_invariant({cl.object_ids})")

        # Supplementary narration-to-visual consistency checks
        low_narr = sc.narration.lower()
        live_poly_vals = [v for k, v in live_polys.items() if state.get(k) in ("declared", "visible")]
        if "four triangles" in low_narr or "four congruent" in low_narr:
            tri_cnt = sum(1 for verts in live_poly_vals if len(verts) == 3)
            if tri_cnt < 4:
                issues.append(f"{where}: narration claims four triangles, but only {tri_cnt} live triangle(s) exist.")
        if "three squares" in low_narr:
            sq_cnt = sum(1 for verts in live_poly_vals if is_square(verts))
            if sq_cnt < 3:
                issues.append(f"{where}: narration claims three squares, but only {sq_cnt} valid live square(s) exist.")
        if live_poly_vals and re.search(r"\b(?:rotate|rotates|rotating)\b", low_narr) and not re.search(r"\b(?:never|without|not|no)\s+rotat", low_narr):
            if not any(e.op in ("rotate", "animate_param") for e in sc.events):
                issues.append(f"{where}: narration mentions rotation, but no rotate/animate_param event is present.")

        # Reject bare empty axes stages (every declared axes must host at least one plotted curve, vector, trajectory, point, or polygon)
        non_chrome_kinds = ("axes", "annotation", "equation", "matrix", "table")
        scene_axes_ids = [o.id for o in sc.objects if o.kind == "axes"]
        scene_non_chrome = [o for o in sc.objects if o.kind not in non_chrome_kinds]
        for ax_id in scene_axes_ids:
            has_local = any(o.axes_id == ax_id for o in scene_non_chrome) or any(
                getattr(c.params, "axes_id", None) == ax_id for c in sc.constructions
            )
            has_later = any(
                (o.kind not in non_chrome_kinds and o.axes_id == ax_id)
                for later_sc in plan.scenes[idx:]
                for o in later_sc.objects
            )
            if not (has_local or (not require_claims and has_later)):
                issues.append(f"{where}: bare axes {ax_id!r} has no plotted curve, vector, trajectory, point, or polygon in this scene.")
        live_stage_kinds = {live_objs[oid].kind for oid, st_v in state.items() if st_v in ("declared", "visible")}
        if require_claims and "axes" in live_stage_kinds and not scene_non_chrome and not any(o.kind in ("matrix", "table", "bar_chart", "graph_network") for o in sc.objects):
            if not any(o.axes_id for o in scene_non_chrome) and not scene_axes_ids:
                issues.append(f"{where}: empty axes stage (declare curves, vectors, polygons, or trajectories alongside axes).")

        for obj in sc.objects:
            if not obj.persist and state.get(obj.id) in ("declared", "visible"):
                state[obj.id] = "removed"

    if require_claims and total_claims == 0:
        issues.append("Mathematical plan must include at least one StructuredClaim in scenes[].claims.")

    # Generalized narration-to-claim coverage audit (catch unbacked math assertions even when scenes omit StructuredClaim)
    _MAT_OBLIGATION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (r"\b(?:rows?\s+sum\s+to\s+one|row[- ]stochastic|probabilities?\s+sum\s+to\s+(?:1|one)|stationary\s+(?:distribution|equilibrium))\b", ("markov_stochastic_invariant", "probability_normalization", "markov_chain_evolution")),
        (r"\b(?:determinant\s+(?:of|equals|is))\b", ("eigenvalue_determinant", "eigen_decomposition_2x2")),
        (r"\b(?:eigenvalues?\s+(?:have|with)\s+(?:negative|positive|zero)\s+real|asymptotically\s+stable)\b", ("eigenvalue_stability_regime", "eigen_decomposition_2x2", "ode_phase_trajectory")),
        (r"\b(?:derivative\s+(?:of|equals|is))\b", ("derivative_equality", "stationary_point", "local_minimum", "gradient_descent_path")),
        (r"\b(?:posterior\s+probability|given\s+evidence)\b", ("conditional_probability", "bayes_area_model")),
        (r"\b(?:area\s+(?:conserved|equals|conservation)|pythagorean)\b", ("area_conservation", "area_equality", "pythagorean_dissection", "right_triangle")),
    )
    narration_gaps: list[str] = []
    if require_claims:
        for sc in plan.scenes:
            cl_kinds = {c.kind for c in sc.claims}
            con_ops = {c.op for c in sc.constructions}
            backed = cl_kinds | con_ops
            low_n = sc.narration.lower()
            for pat, accepted in _MAT_OBLIGATION_PATTERNS:
                if re.search(pat, low_n) and not (backed & set(accepted)):
                    msg = f"scene {sc.scene_id}: narration asserts mathematical rule ({pat[:40]}...) without backed claim/construction {accepted[:2]}"
                    narration_gaps.append(msg)
                    unsupported_claims.append(msg)

    # Conceptual visual hooks (e.g. number-line intuition with no equations/matrices/claims text) are allowed ONLY if non-assertive
    unclaimed_math_scenes = 0
    unclaimed_math_msgs: list[str] = []
    for sc in plan.scenes:
        if sc.claims or sc.constructions:
            continue
        formal_ids = [o.id for o in sc.objects if o.kind in ("matrix", "equation", "function_plot", "probability_region", "table", "bar_chart")]
        if formal_ids:
            unclaimed_math_scenes += 1
            unclaimed_math_msgs.append(
                f"scene {sc.scene_id}: formal visual(s) {formal_ids} lack a verified StructuredClaim or ConstructionOp."
            )
    if require_claims and narration_gaps:
        issues.extend(narration_gaps)

    unclaimed_non_constructive = unclaimed_math_scenes
    explicit_ai_reviewed = sum(1 for sc in plan.scenes for cl in sc.claims if cl.verification_level == "ai_reviewed")
    coverage = {
        "deterministically_verified": len(verified_claims),
        "numerically_tested": sum(1 for sc in plan.scenes for c in sc.constructions if c.op in ("ode_phase_trajectory", "fourier_general_synthesis", "riemann_integral", "gradient_descent_path", "markov_chain_evolution", "discrete_pmf_convolution")),
        "source_supported": sum(1 for sc in plan.scenes for cl in sc.claims if cl.verification_level == "source_supported"),
        "ai_reviewed": explicit_ai_reviewed,
        "unsupported": len(unsupported_claims) + (unclaimed_non_constructive if require_claims else 0),
        "failed": len(failed_claims),
        "human_review_required": 1 if (failed_claims or unsupported_claims or (require_claims and unclaimed_non_constructive > 0)) else 0,
    }
    return {
        "issues": issues,
        "verified": verified_claims,
        "failed": failed_claims,
        "unsupported": unsupported_claims,
        "unclaimed_math_scenes": unclaimed_math_msgs,
        "narration_gaps": narration_gaps,
        "verification_coverage": coverage,
    }


def validate_v3_plan(plan: VideoPlanV3, require_claims: bool = False) -> list[str]:
    """Simulate object lifecycle and validate math, geometry, claims, bookmarks, and event ordering."""
    return list(verify_v3_plan_detailed(plan, require_claims=require_claims)["issues"])  # type: ignore[arg-type]
