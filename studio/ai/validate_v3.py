"""Pre-render state-simulating validator for VideoPlan V3."""

from __future__ import annotations

import numpy as np

from studio.ai.expr import ExprError, compile_expr
from studio.ai.schema import BOOKMARK_RE
from studio.ai.v3_schema import VideoPlanV3


def validate_v3_plan(plan: VideoPlanV3) -> list[str]:
    """Simulate object lifecycle and validate math, geometry, bookmarks, and event dependencies."""
    issues: list[str] = []
    if plan.schema_version != "3.0":
        issues.append(f"Expected schema_version '3.0', got {plan.schema_version!r}.")
    if not plan.scenes:
        issues.append("VideoPlanV3 must contain at least one scene.")
        return issues

    # state[obj_id] in {"live", "removed"}
    state: dict[str, str] = {}
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

        scene_declared: set[str] = set()
        for obj in sc.objects:
            if not obj.id or not obj.id.replace("_", "").isalnum():
                issues.append(f"{where}: invalid object id {obj.id!r}.")
            if obj.id in scene_declared:
                issues.append(f"{where}: duplicate object id {obj.id!r} in same scene.")
            if state.get(obj.id) == "live":
                issues.append(f"{where}: object {obj.id!r} already exists as persistent; cannot redeclare without removing first.")
            scene_declared.add(obj.id)
            state[obj.id] = "live"

            if obj.axes_id and state.get(obj.axes_id) != "live":
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
            elif obj.kind == "polygon" and len(obj.vertices) < 3:
                issues.append(f"{where} obj {obj.id}: polygon requires >= 3 vertices.")
            elif obj.kind == "circle" and obj.radius is not None and obj.radius <= 0:
                issues.append(f"{where} obj {obj.id}: circle radius must be positive.")

        for ev in sc.events:
            if state.get(ev.target_id) != "live":
                issues.append(f"{where}: event target_id {ev.target_id!r} is not live (missing or removed).")
            if ev.op == "transform":
                if not ev.source_id or state.get(ev.source_id) != "live":
                    issues.append(f"{where}: transform requires live source_id (got {ev.source_id!r}).")
            if ev.op == "animate_param" and (not ev.param_range or len(ev.param_range) != 2):
                issues.append(f"{where}: animate_param requires param_range=[start, end].")
            if ev.anchor_mark and ev.anchor_mark not in marks:
                issues.append(f"{where}: anchor_mark {ev.anchor_mark!r} missing from narration.")
            if ev.op == "remove":
                state[ev.target_id] = "removed"

        for obj in sc.objects:
            if not obj.persist and state.get(obj.id) == "live":
                state[obj.id] = "removed"

    return issues
