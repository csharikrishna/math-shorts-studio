"""Strict static validation for VideoPlan V2."""

from __future__ import annotations

import numpy as np

from studio.ai.expr import ExprError, compile_expr
from studio.ai.multi_schema import VideoPlan


def validate_video_plan(plan: VideoPlan) -> list[str]:
    """Validate VideoPlan V2 before any TTS or GPU rendering."""
    issues: list[str] = []
    if plan.schema_version != "2.0":
        issues.append(f"Unsupported schema_version {plan.schema_version!r}; expected '2.0'.")
    if not plan.scenes:
        issues.append("VideoPlan must contain at least one scene.")
        return issues

    if len(plan.x_range) != 2 or not np.all(np.isfinite(plan.x_range)) or plan.x_range[0] >= plan.x_range[1]:
        issues.append(f"Invalid x_range {plan.x_range}.")
    if len(plan.y_range) != 2 or not np.all(np.isfinite(plan.y_range)) or plan.y_range[0] >= plan.y_range[1]:
        issues.append(f"Invalid y_range {plan.y_range}.")

    seen_ids: set[str] = set()
    for idx, sc in enumerate(plan.scenes, 1):
        where = f"scene {idx} ({sc.scene_id})"
        if not sc.scene_id or sc.scene_id in seen_ids:
            issues.append(f"{where}: scene_id must be non-empty and unique.")
        seen_ids.add(sc.scene_id)
        if not sc.narration.strip():
            issues.append(f"{where}: narration must not be empty.")
        if sc.y_range is not None:
            if len(sc.y_range) != 2 or not np.all(np.isfinite(sc.y_range)) or sc.y_range[0] >= sc.y_range[1]:
                issues.append(f"{where}: invalid y_range {sc.y_range}.")

        prim = sc.primitive
        for field_name, expr_str in (("expr", prim.expr), ("secondary_expr", prim.secondary_expr)):
            if expr_str:
                try:
                    compile_expr(expr_str, ("x",))
                except ExprError as exc:
                    issues.append(f"{where} {field_name}: {exc}")

        if prim.kind == "spectrum_bars":
            if not prim.harmonics or any(k <= 0 for k in prim.harmonics):
                issues.append(f"{where}: spectrum_bars requires positive harmonics.")
        elif not prim.expr:
            issues.append(f"{where}: primitive {prim.kind!r} requires expr.")

    return issues
