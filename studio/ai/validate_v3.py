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

    for sc in plan.scenes:
        for con in sc.constructions:
            if con.op == "winding_complex_curve" and con.params.expr and ";" in con.params.expr:
                issues.append(
                    f"scene {sc.scene_id}: winding_complex_curve requires scalar signal expr(t) without ';'; "
                    "use dynamical_simulation or ode_phase_trajectory (or standalone parametric_curve) for 2D spirals."
                )
    if issues:
        return {"issues": issues, "verified": verified_claims, "failed": failed_claims, "unsupported": unsupported_claims}
    try:
        plan = materialize_constructions(plan)
    except Exception as exc:
        issues.append(f"Construction/model error: {exc}")
        return {"issues": issues, "verified": verified_claims, "failed": failed_claims, "unsupported": unsupported_claims}

    from studio.ai.linked_models import evaluate_linked_math_model
    computed_models: dict[str, dict[str, object]] = {}
    for m_spec in getattr(plan, "math_models", []) or []:
        if m_spec.model_id in computed_models:
            issues.append(f"Duplicate LinkedMathModelSpec model_id {m_spec.model_id!r}.")
            continue
        try:
            computed_models[m_spec.model_id] = evaluate_linked_math_model(m_spec)
        except Exception as exc:
            issues.append(f"LinkedMathModelSpec {m_spec.model_id!r} evaluation failed: {exc}")

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
            if obj.model_id and obj.model_id not in computed_models:
                issues.append(f"{where} obj {obj.id}: references unknown math_model {obj.model_id!r}.")

            if obj.kind == "function_plot":
                if not obj.expr:
                    if not (obj.model_id and obj.model_output) and not (obj.points_list and len(obj.points_list) >= 2):
                        issues.append(f"{where} obj {obj.id}: function_plot requires expr.")
                else:
                    try:
                        compile_expr(obj.expr, ("x",))
                    except ExprError as exc:
                        issues.append(f"{where} obj {obj.id}: {exc}")
            elif obj.kind == "parametric_curve":
                if not obj.expr or ";" not in obj.expr:
                    if not (obj.model_id and obj.model_output) and not (obj.points_list and len(obj.points_list) >= 2):
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
            elif obj.kind in ("data_series", "particle_system", "waveform"):
                if not all_coords_finite(obj.points_list):
                    issues.append(f"{where} obj {obj.id}: points_list contains non-finite coordinates.")
            elif obj.kind == "graph_network":
                if not all_coords_finite(obj.points_list):
                    issues.append(f"{where} obj {obj.id}: points_list contains non-finite coordinates.")
                elif len(obj.points_list) < 2 and len(obj.node_labels) < 2:
                    issues.append(f"{where} obj {obj.id}: graph_network requires >= 2 nodes in points_list or node_labels.")
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
                if ev.target_id in live_objs:
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
                has_linked_rng = ev.target_id in live_objs and bool(live_objs[ev.target_id].model_id)
                if not has_linked_rng:
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
                        from studio.ai.simulation import infer_phase_var_names

                        v0, v1 = infer_phase_var_names(cl.dx_expr, cl.dy_expr)
                        scope_e = tuple(dict.fromkeys([v0, v1, "t"]))
                        fx = compile_expr(cl.dx_expr, scope_e)
                        fy = compile_expr(cl.dy_expr, scope_e)
                        xv = float(cl.x_val or 0.0)
                        yv = float(cl.expected_value or 0.0)
                        kw_e = {v0: xv, v1: yv, "t": 0.0}
                        dx_v, dy_v = float(fx(**kw_e)), float(fy(**kw_e))
                        if math.hypot(dx_v, dy_v) > max(cl.tolerance, 1e-3):
                            msg = (
                                f"{where} claim ode_equilibrium: |({dx_v:.4f}, {dy_v:.4f})| != 0 at ({xv}, {yv}) "
                                f"— both dx_expr and dy_expr must vanish at a fixed point; use simulation_trajectory_bounds for oscillating/non-equilibrium states."
                            )
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

            elif cl.kind in (
                "fourier_winding_centroid",
                "spectral_derivation_consistency",
                "signal_filtering_equivalence",
                "parseval_energy_conservation",
            ):
                m_data = computed_models.get(cl.model_id or "") if cl.model_id else None
                if m_data is None:
                    # Also accept materialized signal_spectral_pipeline via objects sharing one model_id
                    inferred_mid = next((live_objs[oid].model_id for oid in cl.object_ids if live_objs[oid].model_id in computed_models), None)
                    m_data = computed_models.get(inferred_mid) if inferred_mid else None
                if m_data is None:
                    msg = f"{where} claim {cl.kind}: requires a valid LinkedMathModelSpec model_id (signal_spectral_pipeline)."
                    issues.append(msg); unsupported_claims.append(msg)
                elif cl.kind == "fourier_winding_centroid":
                    fake_1d = [oid for oid in cl.object_ids if live_objs[oid].kind == "function_plot"]
                    if fake_1d:
                        msg = f"{where} claim fourier_winding_centroid: rejects 1D Cartesian function_plot {fake_1d}; must bind genuine 2D complex winding curve."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        exp_c = list(m_data.get("centroid_point") or [0.0, 0.0])  # type: ignore[arg-type]
                        pts_objs = [live_objs[oid] for oid in cl.object_ids if live_objs[oid].kind in ("point", "vector")]
                        mismatch = False
                        for po in pts_objs:
                            got = po.end_coords if po.kind == "vector" else po.coords
                            if got and (abs(got[0] - exp_c[0]) > max(cl.tolerance, 2e-2) or abs(got[1] - exp_c[1]) > max(cl.tolerance, 2e-2)):
                                mismatch = True
                        if mismatch:
                            msg = f"{where} claim fourier_winding_centroid: visual centroid differs from trapezoidal integral C(w)={exp_c}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: fourier_winding_centroid(C={exp_c})")
                elif cl.kind == "spectral_derivation_consistency":
                    spec_objs = [live_objs[oid] for oid in cl.object_ids if live_objs[oid].kind == "data_series" and live_objs[oid].points_list]
                    exp_peak = list(m_data.get("spectrum_peak") or [0.0, 0.0])  # type: ignore[arg-type]
                    if not spec_objs:
                        msg = f"{where} claim spectral_derivation_consistency: requires materialized spectrum data_series object."
                        issues.append(msg); unsupported_claims.append(msg)
                    else:
                        peak_got = max(spec_objs[0].points_list, key=lambda p: p[1])
                        if abs(peak_got[1] - exp_peak[1]) > max(cl.tolerance, 5e-2):
                            msg = f"{where} claim spectral_derivation_consistency: curve peak {peak_got} != derived {exp_peak}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: spectral_derivation_consistency(peak={exp_peak})")
                elif cl.kind == "signal_filtering_equivalence":
                    meas = dict(m_data.get("measurements") or {})  # type: ignore[arg-type]
                    e_raw = float(meas.get("energy_time_raw", 0.0))
                    e_filt = float(meas.get("energy_time_filtered", 0.0))
                    if e_raw <= 0 or abs(e_raw - e_filt) < max(1e-4, 0.01 * e_raw):
                        msg = f"{where} claim signal_filtering_equivalence: filtered signal did not measurably alter spectral energy (raw={e_raw}, filt={e_filt})."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: signal_filtering_equivalence(e_raw={e_raw:.2f}, e_filt={e_filt:.2f})")
                elif cl.kind == "parseval_energy_conservation":
                    meas = dict(m_data.get("measurements") or {})  # type: ignore[arg-type]
                    rel_err = float(meas.get("parseval_rel_error", 1.0))
                    if rel_err > max(cl.tolerance, 5e-3):
                        msg = f"{where} claim parseval_energy_conservation: relative error {rel_err:.6f} exceeds tolerance."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: parseval_energy_conservation(rel_err={rel_err:.2e})")

            elif cl.kind in (
                "simulation_trajectory_bounds",
                "frequency_response_peak",
                "state_bounds",
                "positivity",
                "conservation_law",
                "energy_dissipation",
                "periodicity",
                "numerical_accuracy",
                "parameter_validity",
                "physical_units",
            ):
                m_spec = next((ms for ms in (getattr(plan, "math_models", []) or []) if ms.model_id == cl.model_id), None)
                m_data = computed_models.get(cl.model_id or "") if cl.model_id else None
                if m_data is None:
                    inferred_mid = next((live_objs[oid].model_id for oid in cl.object_ids if oid in live_objs and live_objs[oid].model_id in computed_models), None)
                    m_data = computed_models.get(inferred_mid) if inferred_mid else None
                    if m_spec is None and inferred_mid:
                        m_spec = next((ms for ms in (getattr(plan, "math_models", []) or []) if ms.model_id == inferred_mid), None)
                if m_data is None:
                    msg = f"{where} claim {cl.kind}: requires a valid LinkedMathModelSpec model_id."
                    issues.append(msg); unsupported_claims.append(msg)
                elif cl.kind in ("simulation_trajectory_bounds", "state_bounds"):
                    pts_l = list(m_data.get("phase_orbit") or m_data.get("time_series_x") or [])
                    if len(pts_l) < 4 or not all_coords_finite(pts_l):
                        msg = f"{where} claim {cl.kind}: simulation has empty or nonfinite states."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        s_vars_m = list(m_data.get("state_vars") or [])
                        states_m = list(m_data.get("states") or [])
                        col_idx = 0
                        if cl.state_var and cl.state_var in s_vars_m:
                            col_idx = s_vars_m.index(cl.state_var)
                        series = [float(st[col_idx]) for st in states_m if len(st) > col_idx] if states_m else [float(p[min(1, len(p) - 1)]) for p in pts_l]
                        lo_b = cl.lower_bound if cl.lower_bound is not None else cl.a_bound
                        hi_b = cl.upper_bound if cl.upper_bound is not None else cl.b_bound
                        mn_v, mx_v = min(series), max(series)
                        span_v = max(abs(st[i] - states_m[0][i]) for st in states_m for i in range(len(st))) if states_m else (mx_v - mn_v)
                        if cl.kind == "state_bounds" and lo_b is None and hi_b is None:
                            msg = f"{where} claim state_bounds: requires explicit lower_bound/upper_bound or a_bound/b_bound."
                            issues.append(msg); unsupported_claims.append(msg)
                        elif lo_b is not None and mn_v < float(lo_b) - cl.tolerance:
                            msg = f"{where} claim {cl.kind}: min({cl.state_var or 'state'})={mn_v:.4f} violates lower bound {lo_b}."
                            issues.append(msg); failed_claims.append(msg)
                        elif hi_b is not None and mx_v > float(hi_b) + cl.tolerance:
                            msg = f"{where} claim {cl.kind}: max({cl.state_var or 'state'})={mx_v:.4f} violates upper bound {hi_b}."
                            issues.append(msg); failed_claims.append(msg)
                        elif cl.expected_value is not None and mx_v > float(cl.expected_value) + cl.tolerance:
                            msg = f"{where} claim {cl.kind}: max={mx_v:.4f} exceeds asserted expected_value={cl.expected_value}."
                            issues.append(msg); failed_claims.append(msg)
                        elif span_v <= 1e-6 and "equilibrium" not in f"{cl.statement} {sc.narration}".lower():
                            msg = f"{where} claim {cl.kind}: simulation collapsed to constant state {states_m[0] if states_m else pts_l[0]}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: {cl.kind}(n={len(series)}, range=[{mn_v:.3f}, {mx_v:.3f}])")
                elif cl.kind == "positivity":
                    states_m = list(m_data.get("states") or [])
                    s_vars_m = list(m_data.get("state_vars") or [])
                    if not states_m:
                        msg = f"{where} claim positivity: simulation has no state trajectory."
                        issues.append(msg); unsupported_claims.append(msg)
                    else:
                        cols = [s_vars_m.index(cl.state_var)] if (cl.state_var and cl.state_var in s_vars_m) else list(range(len(states_m[0])))
                        min_val = min(float(st[ci]) for st in states_m for ci in cols)
                        if min_val <= 0.0:
                            msg = f"{where} claim positivity: state dropped to {min_val:.5f} <= 0 (positivity violated)."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: positivity(min={min_val:.4f} > 0)")
                elif cl.kind == "conservation_law":
                    inv_expr = (cl.expr or "").strip()
                    derived_m = dict(m_data.get("derived") or {})
                    seq: list[float] = []
                    if inv_expr and inv_expr in derived_m:
                        seq = [float(v) for v in derived_m[inv_expr]]
                    elif inv_expr:
                        from studio.ai.simulation import evaluate_derived_observables, SimulationResult

                        s_vars_m = list(m_data.get("state_vars") or ["x", "y"])
                        sim_tmp = SimulationResult(
                            solver="verify",
                            state_vars=s_vars_m,
                            t_axis=list(m_data.get("t_axis") or []),
                            states=list(m_data.get("states") or []),
                        )
                        try:
                            seq = evaluate_derived_observables(
                                sim_tmp,
                                {"inv": inv_expr},
                                params=(m_spec.params if m_spec else None),
                            )["inv"]
                        except Exception as exc:
                            msg = f"{where} claim conservation_law: failed evaluating {inv_expr!r}: {exc}"
                            issues.append(msg); failed_claims.append(msg)
                            continue
                    elif "conserved_invariant" in derived_m:
                        seq = [float(v) for v in derived_m["conserved_invariant"]]
                    elif "total_energy" in derived_m:
                        seq = [float(v) for v in derived_m["total_energy"]]
                    if len(seq) < 2:
                        msg = f"{where} claim conservation_law: requires expr or model derived conserved observable."
                        issues.append(msg); unsupported_claims.append(msg)
                    else:
                        ref_v = seq[0]
                        rel_drift = max(abs(v - ref_v) for v in seq) / max(1.0, abs(ref_v))
                        tol_use = max(cl.tolerance, 5e-3)
                        if rel_drift > tol_use:
                            msg = f"{where} claim conservation_law: quantity varied by rel_drift={rel_drift:.5f} > {tol_use}."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: conservation_law(drift={rel_drift:.2e})")
                elif cl.kind == "energy_dissipation":
                    derived_m = dict(m_data.get("derived") or {})
                    e_seq = list(derived_m.get("total_energy") or [])
                    if not e_seq and cl.expr:
                        from studio.ai.simulation import evaluate_derived_observables, SimulationResult

                        sim_tmp = SimulationResult(
                            solver="verify",
                            state_vars=list(m_data.get("state_vars") or ["x", "v"]),
                            t_axis=list(m_data.get("t_axis") or []),
                            states=list(m_data.get("states") or []),
                        )
                        e_seq = evaluate_derived_observables(sim_tmp, {"E": cl.expr}, params=(m_spec.params if m_spec else None))["E"]
                    if len(e_seq) < 4:
                        msg = f"{where} claim energy_dissipation: requires explicit total_energy derivation and physical assumptions."
                        issues.append(msg); unsupported_claims.append(msg)
                    else:
                        inc_max = max(float(e_seq[i + 1]) - float(e_seq[i]) for i in range(len(e_seq) - 1))
                        net_drop = float(e_seq[0]) - float(e_seq[-1])
                        if inc_max > max(cl.tolerance, 1e-3) or net_drop <= 1e-4:
                            msg = f"{where} claim energy_dissipation: E_init={e_seq[0]:.4f}, E_final={e_seq[-1]:.4f}, max_step_inc={inc_max:.5f} violates monotone dissipation."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            verified_claims.append(f"{where}: energy_dissipation(drop={net_drop:.4f})")
                elif cl.kind == "periodicity":
                    states_m = list(m_data.get("states") or [])
                    t_ax = list(m_data.get("t_axis") or [])
                    if len(states_m) < 16 or len(t_ax) != len(states_m):
                        msg = f"{where} claim periodicity: insufficient simulation samples to check periodicity."
                        issues.append(msg); unsupported_claims.append(msg)
                    else:
                        y0 = np.asarray(states_m[0], dtype=float)
                        scale_p = max(1.0, float(np.linalg.norm(y0)))
                        dists = [float(np.linalg.norm(np.asarray(st, dtype=float) - y0)) / scale_p for st in states_m]
                        span_orbit = max(dists)
                        # Must leave initial neighborhood significantly before returning close to initial state
                        left_idx = next((i for i, d in enumerate(dists) if d > 0.12), None)
                        ret_idx = next((i for i in range((left_idx or 0) + 4, len(dists)) if dists[i] <= max(cl.tolerance * 20.0, 0.05)), None) if left_idx is not None else None
                        if span_orbit <= 0.05 or ret_idx is None:
                            msg = f"{where} claim periodicity: trajectory does not complete a near-periodic return (max_disp={span_orbit:.4f}, min_post_dist={min(dists[left_idx+4:] or [1.0]):.4f})."
                            issues.append(msg); failed_claims.append(msg)
                        else:
                            est_period = float(t_ax[ret_idx]) - float(t_ax[0])
                            if cl.expected_value is not None and abs(est_period - float(cl.expected_value)) > max(cl.tolerance, 0.15 * float(cl.expected_value)):
                                msg = f"{where} claim periodicity: observed period {est_period:.3f} != expected {cl.expected_value}."
                                issues.append(msg); failed_claims.append(msg)
                            else:
                                verified_claims.append(f"{where}: periodicity(T~={est_period:.2f})")
                elif cl.kind == "numerical_accuracy":
                    meas = dict(m_data.get("measurements") or {})
                    max_err = float(meas.get("max_error_est", 0.0))
                    rtol_m = float(meas.get("rtol", 1e-4))
                    if max_err > max(cl.tolerance, rtol_m * 10.0):
                        msg = f"{where} claim numerical_accuracy: max_error_est={max_err:.2e} exceeds tolerance."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: numerical_accuracy(solver={meas.get('solver_algorithm', meas.get('solver'))}, err={max_err:.2e})")
                elif cl.kind == "parameter_validity":
                    p_dict = dict(m_spec.params or {}) if m_spec else {}
                    bad_p = [f"{k}={v}" for k, v in p_dict.items() if (not math.isfinite(float(v)) or float(v) < 0.0)]
                    if m_spec and m_spec.mass is not None and float(m_spec.mass) <= 0.0:
                        bad_p.append(f"mass={m_spec.mass}")
                    if m_spec and m_spec.natural_freq is not None and float(m_spec.natural_freq) <= 0.0:
                        bad_p.append(f"natural_freq={m_spec.natural_freq}")
                    if bad_p:
                        msg = f"{where} claim parameter_validity: invalid physical parameter(s) {bad_p}."
                        issues.append(msg); failed_claims.append(msg)
                    elif not (p_dict or (m_spec and (m_spec.mass is not None or m_spec.natural_freq is not None))):
                        msg = f"{where} claim parameter_validity: requires explicit declared model parameters."
                        issues.append(msg); unsupported_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: parameter_validity({list(p_dict.keys())})")
                elif cl.kind == "physical_units":
                    declared_u = dict(cl.units or {}) or dict(m_data.get("units") or {})
                    if not declared_u or not all(str(v).strip() for v in declared_u.values()):
                        msg = f"{where} claim physical_units: requires explicit non-empty variable unit declarations."
                        issues.append(msg); unsupported_claims.append(msg)
                    elif m_spec and m_spec.kind == "signal_spectral_pipeline" and m_spec.angular_units == "hz" and "rad/s" in str(declared_u.get("freq", "")).lower():
                        msg = f"{where} claim physical_units: model angular_units='hz' conflicts with declared unit {declared_u}."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: physical_units({declared_u})")
                elif cl.kind == "frequency_response_peak":
                    meas = dict(m_data.get("measurements") or {})
                    has_pk = bool(meas.get("has_nonzero_resonance_peak", False))
                    has_sing = bool(meas.get("has_singularity", False))
                    zeta_v = float(meas.get("damping_ratio", 0.0))
                    w_res = float(meas.get("resonance_freq", 0.0))
                    low_st = f"{cl.statement} {sc.narration}".lower()
                    claims_nonzero_pk = bool(re.search(r"\bnonzero\s+resonance\s+peak\b|\balways\s+has\s+a\s+resonance\s+peak\b", low_st))
                    if zeta_v >= (1.0 / math.sqrt(2.0)) - 1e-4 and (w_res > 1e-3 or claims_nonzero_pk):
                        msg = f"{where} claim frequency_response_peak: zeta={zeta_v:.3f} >= 1/sqrt(2) cannot yield a nonzero resonance peak."
                        issues.append(msg); failed_claims.append(msg)
                    elif has_sing and cl.expected_value is not None:
                        msg = f"{where} claim frequency_response_peak: undamped resonance at w0={meas.get('singularity_freq')} is unbounded (singularity), cannot equal finite expected_value={cl.expected_value}."
                        issues.append(msg); failed_claims.append(msg)
                    elif cl.expected_value is not None and abs(w_res - float(cl.expected_value)) > max(cl.tolerance, 5e-2):
                        msg = f"{where} claim frequency_response_peak: computed w_res={w_res:.4f} != expected {cl.expected_value}."
                        issues.append(msg); failed_claims.append(msg)
                    else:
                        verified_claims.append(f"{where}: frequency_response_peak(zeta={zeta_v:.2f}, w_res={w_res:.2f}, has_pk={has_pk}, singularity={has_sing})")

        # Validate ClaimLedgerEntry rows (exact spoken span + assumptions + backing)
        from studio.audio.tts import parse_script as _parse_scr

        norm_narr_plain = re.sub(r"\s+", " ", _parse_scr(BOOKMARK_RE.sub("", re.sub(r"\[m:[^\]]+\]", "", sc.narration)))[0]).strip().lower()
        for entry in getattr(sc, "claim_ledger", []) or []:
            span_clean = re.sub(r"\s+", " ", _parse_scr(BOOKMARK_RE.sub("", entry.exact_narration_span))[0]).strip().lower()
            if span_clean and span_clean not in norm_narr_plain:
                span_words = [w.strip(".,!?;:\"'()") for w in span_clean.split() if len(w.strip(".,!?;:\"'()")) > 2]
                narr_words = set(w.strip(".,!?;:\"'()") for w in norm_narr_plain.split())
                overlap_ratio = (sum(1 for w in span_words if w in narr_words) / len(span_words)) if span_words else 0.0
                if overlap_ratio < 0.85:
                    issues.append(f"{where} claim_ledger[{entry.claim_id}]: exact_narration_span not found in scene narration.")
            if entry.model_id and entry.model_id not in computed_models:
                issues.append(f"{where} claim_ledger[{entry.claim_id}]: references unknown model_id {entry.model_id!r}.")
            missing_l = [oid for oid in entry.object_ids if oid not in live_objs]
            if missing_l:
                issues.append(f"{where} claim_ledger[{entry.claim_id}]: references missing object(s) {missing_l}.")
            if entry.status == "failed":
                failed_claims.append(f"{where}: ledger {entry.claim_id}")
            elif entry.status == "unverified":
                unsupported_claims.append(f"{where}: ledger {entry.claim_id} unverified")

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
        if re.search(r"\b(?:wind|winding|wraps?\s+around)\b", low_narr) and re.search(r"\b(?:center\s+of\s+mass|centroid|frequency|fourier|complex)\b", low_narr):
            has_winding_model = any(o.model_id in computed_models and o.model_output == "winding_curve" for o in sc.objects)
            has_winding_op = any(c.op == "winding_complex_curve" for c in sc.constructions)
            has_fake_trig_plot = any(o.kind == "function_plot" and o.expr and re.search(r"\b(?:cos|sin)\(", o.expr) for o in sc.objects)
            if has_fake_trig_plot and not (has_winding_model or has_winding_op):
                issues.append(f"{where}: narration describes complex winding, but stage uses a 1D cos/sin function_plot instead of winding_complex_curve/model_output='winding_curve'.")
            elif not (has_winding_model or has_winding_op or any(o.kind in ("parametric_curve", "data_series") and o.points_list for o in sc.objects)):
                issues.append(f"{where}: narration describes winding center-of-mass without a 2D winding curve object.")

        # Reject bare empty axes & collapsed zero-span trajectory stages
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
        for o in sc.objects:
            if o.kind in ("data_series", "parametric_curve") and o.points_list and len(o.points_list) >= 2 and not getattr(o, "sweep_keyframes", None):
                xs_o = [float(p[0]) for p in o.points_list]
                ys_o = [float(p[1]) for p in o.points_list]
                if (max(xs_o) - min(xs_o)) < 1e-4 and (max(ys_o) - min(ys_o)) < 1e-4 and require_claims:
                    issues.append(
                        f"{where} obj {o.id}: trajectory collapsed to a single stationary point {o.points_list[0]}; "
                        "start ODE away from exact equilibrium or include surrounding vector_field/orbit."
                    )
        live_stage_kinds = {live_objs[oid].kind for oid, st_v in state.items() if oid in live_objs and st_v in ("declared", "visible")}
        live_non_chrome = [
            live_objs[oid]
            for oid, st_v in state.items()
            if oid in live_objs and st_v in ("declared", "visible") and live_objs[oid].kind not in non_chrome_kinds
        ]
        if require_claims and "axes" in live_stage_kinds and not scene_non_chrome and not live_non_chrome and not any(o.kind in ("matrix", "table", "bar_chart", "graph_network") for o in sc.objects):
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
    source_supported_unbacked = sum(
        1
        for sc in plan.scenes
        for entry in (getattr(sc, "claim_ledger", []) or [])
        if entry.status == "source_supported"
    ) + sum(
        1
        for sc in plan.scenes
        for cl in sc.claims
        if cl.verification_level == "source_supported"
    )
    # Track narration-to-demonstration dynamic motion coverage with substantive motion-capability verification
    _STATE_MOTION_OPS = {"transform", "morph", "move", "rotate", "scale", "trace", "animate_param", "set_param", "iterate_states", "interpolate_values"}
    _DYNAMIC_NARR_RE = re.compile(
        r"\b(?:oscillat\w*|swing\w*|vibrat\w*|evolv\w*|sweep\w*|converg\w*|spiral\w*|decay\w*|dissipat\w*|resonat\w*|trajector\w*|simulat\w*|phase\s+space|move|moves|moving|transform\w*|shift\w*|slide\w*|orbit\w*|step[- ]by[- ]step|iteration)\b"
    )
    dynamic_motion_gaps: list[str] = []
    parameter_sync_issues: list[str] = []
    hollow_motion_issues: list[str] = []
    animated_scenes_count = 0
    total_dynamic_scenes = 0
    _swept_models_seen: set[str] = set()
    _all_objs_seen: dict[str, SceneObject] = {o.id: o for sc in plan.scenes for o in sc.objects}

    def _event_produces_real_state_change(ev_item: Any) -> tuple[bool, str]:
        op_n = str(ev_item.op)
        if op_n not in _STATE_MOTION_OPS:
            return False, ""
        tgt_o = _all_objs_seen.get(ev_item.target_id)
        if op_n == "move":
            sh = ev_item.shift or []
            if len(sh) < 2 or math.hypot(float(sh[0]), float(sh[1])) <= 1e-4:
                return False, f"move on {ev_item.target_id!r} has zero displacement shift"
            return True, ""
        if op_n == "rotate":
            if abs(float(ev_item.angle_deg or 0.0)) <= 1e-2:
                return False, f"rotate on {ev_item.target_id!r} has zero angle_deg"
            return True, ""
        if op_n == "scale":
            if abs(float(ev_item.scale_factor or 1.0) - 1.0) <= 1e-3:
                return False, f"scale on {ev_item.target_id!r} has unit scale_factor"
            return True, ""
        if op_n in ("transform", "morph"):
            return bool(ev_item.source_id), ("" if ev_item.source_id else f"{op_n} missing source_id")
        if op_n == "trace":
            if tgt_o is not None and tgt_o.kind == "point":
                has_kfs = bool(getattr(tgt_o, "sweep_keyframes", None) and len(tgt_o.sweep_keyframes) >= 2)
                has_param = bool(tgt_o.param_name and tgt_o.expr and ";" in tgt_o.expr)
                if not (has_kfs or has_param):
                    return False, f"trace on static unlinked point {ev_item.target_id!r} produces zero visual motion"
            if tgt_o is not None and tgt_o.kind in ("data_series", "parametric_curve") and tgt_o.points_list:
                xs_t = [float(p[0]) for p in tgt_o.points_list]
                ys_t = [float(p[1]) for p in tgt_o.points_list]
                if (max(xs_t) - min(xs_t)) + (max(ys_t) - min(ys_t)) <= 1e-4 and not getattr(tgt_o, "sweep_keyframes", None):
                    return False, f"trace on constant zero-length curve {ev_item.target_id!r} produces zero visual motion"
            return True, ""
        if op_n in ("animate_param", "set_param", "iterate_states", "interpolate_values"):
            if tgt_o is not None:
                has_kfs = bool(getattr(tgt_o, "sweep_keyframes", None) and len(tgt_o.sweep_keyframes) >= 2)
                has_param = bool(tgt_o.param_name or tgt_o.model_id)
                has_bar_steps = bool(tgt_o.kind == "bar_chart" and tgt_o.points_list and len(tgt_o.points_list) >= 2)
                if not (has_kfs or has_param or has_bar_steps or tgt_o.kind in ("data_series", "parametric_curve", "function_plot")):
                    return False, f"{op_n} on unparameterized {tgt_o.kind} {ev_item.target_id!r} produces zero state change"
            return True, ""
        return True, ""

    for sc in plan.scenes:
        has_substantive_anim = False
        for e in sc.events:
            if e.op in _STATE_MOTION_OPS:
                ok_ev, reason_ev = _event_produces_real_state_change(e)
                if ok_ev:
                    has_substantive_anim = True
                elif reason_ev and require_claims:
                    hollow_motion_issues.append(f"scene {sc.scene_id}: {reason_ev}.")
            if e.op == "animate_param":
                obj_m = next((o for o in sc.objects if o.id == e.target_id), None) or _all_objs_seen.get(e.target_id)
                mid_e = str((obj_m.model_id if obj_m else None) or e.target_id)
                _swept_models_seen.add(mid_e)
        if has_substantive_anim:
            animated_scenes_count += 1
        low_n = sc.narration.lower()
        is_dynamic_beat = bool(_DYNAMIC_NARR_RE.search(low_n))
        if is_dynamic_beat:
            total_dynamic_scenes += 1
            if not has_substantive_anim:
                dynamic_motion_gaps.append(
                    f"scene {sc.scene_id}: narration describes dynamic evolution/motion without a state-changing animation event that visibly alters state."
                )
        # Check frequency parameter sync across subsequent beats mentioning explicit frequency values
        m_hz = re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b", low_n)
        if m_hz and len(m_hz) == 1:
            spoken_f = float(m_hz[0])
            sc_active_p = dict(getattr(sc, "active_model_params", {}) or {})
            for o in sc.objects:
                if o.model_id and o.model_id in computed_models and o.model_id in _swept_models_seen:
                    if not any(e.op in ("animate_param", "set_param") for e in sc.events) and o.model_id not in sc_active_p:
                        m_info = dict(computed_models[o.model_id].get("measurements") or {})
                        sel_f = float(m_info.get("selected_freq", spoken_f))
                        sw_frames = list(computed_models[o.model_id].get("sweep_frames") or [])
                        end_f = float(sw_frames[-1].get("param", sel_f)) if sw_frames else sel_f
                        if abs(sel_f - spoken_f) > 0.15 or abs(end_f - spoken_f) > 0.15:
                            parameter_sync_issues.append(
                                f"scene {sc.scene_id}: narration focuses on {spoken_f} Hz after sweep (ends at {end_f} Hz, selected_freq={sel_f}) while set_param/active_model_params omits {o.model_id!r}."
                            )
                            break

    coverage = {
        "deterministically_verified": len(verified_claims),
        "numerically_tested": sum(1 for sc in plan.scenes for c in sc.constructions if c.op in ("ode_phase_trajectory", "fourier_general_synthesis", "riemann_integral", "gradient_descent_path", "markov_chain_evolution", "discrete_pmf_convolution", "winding_complex_curve", "spectral_transform_from_signal", "filtered_signal_reconstruction", "general_simulation_engine", "driven_frequency_response")) + len(computed_models),
        "source_supported": source_supported_unbacked,
        "ai_reviewed": explicit_ai_reviewed,
        "unsupported": len(unsupported_claims) + (unclaimed_non_constructive if require_claims else 0),
        "failed": len(failed_claims),
        "human_review_required": 1 if (failed_claims or unsupported_claims or source_supported_unbacked > 0 or parameter_sync_issues or hollow_motion_issues or (require_claims and unclaimed_non_constructive > 0)) else 0,
    }
    issues.extend(parameter_sync_issues)
    issues.extend(hollow_motion_issues)
    return {
        "issues": issues,
        "verified": verified_claims,
        "failed": failed_claims,
        "unsupported": unsupported_claims,
        "unclaimed_math_scenes": unclaimed_math_msgs,
        "narration_gaps": narration_gaps,
        "dynamic_motion_gaps": dynamic_motion_gaps,
        "parameter_sync_issues": parameter_sync_issues,
        "animated_scenes": animated_scenes_count,
        "total_dynamic_scenes": total_dynamic_scenes,
        "verification_coverage": coverage,
    }


def validate_v3_plan(plan: VideoPlanV3, require_claims: bool = False) -> list[str]:
    """Simulate object lifecycle and validate math, geometry, claims, bookmarks, and event ordering."""
    return list(verify_v3_plan_detailed(plan, require_claims=require_claims)["issues"])  # type: ignore[arg-type]
