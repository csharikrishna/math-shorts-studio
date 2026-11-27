"""Pre-render state-simulating validator for VideoPlan V3."""

from __future__ import annotations

import math
import numpy as np

from studio.ai.expr import ExprError, compile_expr
from studio.ai.schema import BOOKMARK_RE
from studio.ai.v3_schema import VideoPlanV3
from studio.ai.world_model import (
    all_coords_finite,
    clean_polygon_vertices,
    is_right_triangle,
    is_square,
    materialize_constructions,
    polygon_area,
    triangles_congruent,
    verify_partition,
)


def _finite_seq(vals: list[float] | None) -> bool:
    return vals is None or all(math.isfinite(float(v)) for v in vals)


def validate_v3_plan(plan: VideoPlanV3) -> list[str]:
    """Simulate object lifecycle and validate math, geometry, claims, bookmarks, and event ordering."""
    issues: list[str] = []
    if plan.schema_version != "3.0":
        issues.append(f"Expected schema_version '3.0', got {plan.schema_version!r}.")
    if not plan.scenes:
        issues.append("VideoPlanV3 must contain at least one scene.")
        return issues

    plan = materialize_constructions(plan)

    # state[obj_id] in {"declared", "visible", "removed"}
    state: dict[str, str] = {}
    live_kinds: dict[str, tuple[str, list[list[float]]]] = {}  # id -> (kind, vertices)
    seen_scenes: set[str] = set()

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
                tag = f"{obj.id} {obj.label}".lower()
                if "square" in tag and not is_square(cleaned):
                    issues.append(f"{where} obj {obj.id}: claimed square fails geometric square check (4 equal sides & 4 right angles).")
                if "triangle" in tag and len(cleaned) != 3:
                    issues.append(f"{where} obj {obj.id}: claimed triangle must have 3 vertices (got {len(cleaned)}).")
                live_kinds[obj.id] = ("polygon", cleaned)

        # Structured mathematical claim verification
        for cl in sc.claims:
            missing = [oid for oid in cl.object_ids if state.get(oid) not in ("declared", "visible")]
            if missing:
                issues.append(f"{where} claim {cl.kind}: references non-live object(s) {missing}.")
                continue
            polys = [live_kinds[oid][1] for oid in cl.object_ids if oid in live_kinds and live_kinds[oid][0] == "polygon"]
            if cl.kind == "right_triangle":
                for oid, verts in zip(cl.object_ids, polys):
                    if not is_right_triangle(verts):
                        issues.append(f"{where} claim right_triangle: {oid!r} is not a valid right triangle.")
            elif cl.kind == "square":
                for oid, verts in zip(cl.object_ids, polys):
                    if not is_square(verts):
                        issues.append(f"{where} claim square: {oid!r} is not a valid square.")
            elif cl.kind == "congruent_triangles":
                if len(polys) < 2:
                    issues.append(f"{where} claim congruent_triangles: needs >= 2 triangles.")
                else:
                    ref = polys[0]
                    for oid, verts in zip(cl.object_ids[1:], polys[1:]):
                        if not triangles_congruent(ref, verts):
                            issues.append(f"{where} claim congruent_triangles: {oid!r} is not congruent to {cl.object_ids[0]!r}.")
            elif cl.kind == "nonzero_area":
                for oid, verts in zip(cl.object_ids, polys):
                    if polygon_area(verts) <= 1e-6:
                        issues.append(f"{where} claim nonzero_area: {oid!r} has zero area.")
            elif cl.kind == "partition_of":
                cid = cl.container_id
                if not cid or cid not in live_kinds:
                    issues.append(f"{where} claim {cl.kind}: container_id {cid!r} is not a live polygon.")
                else:
                    p_issues = verify_partition(polys, live_kinds[cid][1])
                    issues.extend(f"{where} claim {cl.kind}: {pi}" for pi in p_issues)
            elif cl.kind == "area_conservation":
                cid = cl.container_id
                if not cid or cid not in live_kinds:
                    issues.append(f"{where} claim {cl.kind}: container_id {cid!r} is not a live polygon.")
                else:
                    total_pieces = sum(polygon_area(pc) for pc in polys)
                    target_area = polygon_area(live_kinds[cid][1])
                    if abs(total_pieces - target_area) > 1e-3:
                        issues.append(f"{where} claim area_conservation: sum={total_pieces:.4f} vs {cid}={target_area:.4f}.")

        # Supplementary narration-to-visual consistency checks
        low_narr = sc.narration.lower()
        live_polys = [v for k, v in live_kinds.items() if state.get(k) in ("declared", "visible") and v[0] == "polygon"]
        if "four triangles" in low_narr or "four congruent" in low_narr:
            tri_cnt = sum(1 for _, verts in live_polys if len(verts) == 3)
            if tri_cnt < 4:
                issues.append(f"{where}: narration claims four triangles, but only {tri_cnt} live triangle(s) exist.")
        if "three squares" in low_narr:
            sq_cnt = sum(1 for _, verts in live_polys if is_square(verts))
            if sq_cnt < 3:
                issues.append(f"{where}: narration claims three squares, but only {sq_cnt} valid live square(s) exist.")
        if ("rotate" in low_narr or "rotates" in low_narr) and not any(e.op in ("rotate", "animate_param") for e in sc.events):
            issues.append(f"{where}: narration mentions rotation, but no rotate/animate_param event is present.")

        # Validate event dependencies, finite params, and monotonic bookmark order
        has_unanchored_main = any(not e.anchor_mark for e in sc.events)
        if not has_unanchored_main and sc.objects:
            # Auto-reveal will make unanchored objects visible at scene start
            anchored_ids = {e.target_id for e in sc.events if e.anchor_mark}
            for o in sc.objects:
                if o.id not in anchored_ids and state.get(o.id) == "declared":
                    state[o.id] = "visible"

        last_mark_idx = -1
        for ev in sc.events:
            st = state.get(ev.target_id)
            if st not in ("declared", "visible"):
                issues.append(f"{where}: event target_id {ev.target_id!r} is not live (missing or removed).")
            if ev.op in ("move", "rotate", "scale", "highlight", "focus", "remove", "animate_param") and st == "declared":
                # Allow if it was created earlier in this scene's event list; otherwise warn
                issues.append(f"{where}: event {ev.op} on {ev.target_id!r} before it is created/drawn.")
            if ev.op in ("create", "draw"):
                state[ev.target_id] = "visible"
            elif ev.op == "transform":
                if not ev.source_id or state.get(ev.source_id) not in ("declared", "visible"):
                    issues.append(f"{where}: transform requires live source_id (got {ev.source_id!r}).")
                state[ev.target_id] = "visible"
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

        for obj in sc.objects:
            if not obj.persist and state.get(obj.id) in ("declared", "visible"):
                state[obj.id] = "removed"

    return issues
