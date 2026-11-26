"""Validator for VideoPlan V3 (objects, event dependencies, math expressions, bookmarks)."""

from __future__ import annotations

import numpy as np

from studio.ai.expr import ExprError, compile_expr
from studio.ai.schema import BOOKMARK_RE
from studio.ai.v3_schema import VideoPlanV3


def validate_v3_plan(plan: VideoPlanV3) -> list[str]:
    issues: list[str] = []
    if plan.schema_version != "3.0":
        issues.append(f"Expected schema_version '3.0', got {plan.schema_version!r}.")
    if not plan.scenes:
        issues.append("VideoPlanV3 must contain at least one scene.")
        return issues

    live_ids: set[str] = set()
    seen_scenes: set[str] = set()

    for idx, sc in enumerate(plan.scenes, 1):
        where = f"scene {idx} ({sc.scene_id})"
        if not sc.scene_id or sc.scene_id in seen_scenes:
            issues.append(f"{where}: scene_id must be non-empty and unique.")
        seen_scenes.add(sc.scene_id)
        if not sc.narration.strip():
            issues.append(f"{where}: narration must not be empty.")

        marks = set(BOOKMARK_RE.findall(sc.narration))
        scene_obj_ids: set[str] = set()

        for obj in sc.objects:
            if not obj.id or obj.id in scene_obj_ids:
                issues.append(f"{where}: duplicate or empty object id {obj.id!r}.")
            scene_obj_ids.add(obj.id)
            live_ids.add(obj.id)

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

        for ev in sc.events:
            if ev.target_id not in live_ids:
                issues.append(f"{where}: event target_id {ev.target_id!r} not defined.")
            if ev.source_id and ev.source_id not in live_ids:
                issues.append(f"{where}: event source_id {ev.source_id!r} not defined.")
            if ev.anchor_mark and ev.anchor_mark not in marks:
                issues.append(f"{where}: anchor_mark {ev.anchor_mark!r} missing from narration.")

        # Non-persisting objects leave scope after scene
        for obj in sc.objects:
            if not obj.persist:
                live_ids.discard(obj.id)

    return issues
