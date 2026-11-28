"""Automated scene-level visual defect diagnosis and repair reusing preview renderer + FrameAuditResult."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from studio.ai.v3_schema import AnimationEvent, SceneObject, VideoPlanV3
from studio.ai.visual_intelligence import apply_visual_intelligence_and_fit
from studio.ai.world_model import materialize_constructions

_QUESTION_TOKEN_RE = re.compile(r"\?\s*$|\b[a-zA-Z]+\s*=\s*\?(?!\S)")


def _nonfinite_or_zero_span(points: list[list[float]]) -> bool:
    if len(points) < 2:
        return True
    xs = [float(p[0]) for p in points if len(p) >= 2]
    ys = [float(p[1]) for p in points if len(p) >= 2]
    if len(xs) < 2:
        return True
    return (max(xs) - min(xs) < 1e-6) and (max(ys) - min(ys) < 1e-6)


def diagnose_scene_visual_defects(plan: VideoPlanV3, frame_audit: Any | None = None) -> dict[str, list[str]]:
    """Return per-scene defect diagnostics covering empty axes, overlaps, clipped data, static points, and placeholder text."""
    defects: dict[str, list[str]] = {}
    declared_prior: dict[str, SceneObject] = {}
    removed: set[str] = set()

    for sc in plan.scenes:
        sc_issues: list[str] = []
        active: dict[str, SceneObject] = {k: v for k, v in declared_prior.items() if k not in removed}
        for obj in sc.objects:
            active[obj.id] = obj
        local_ids = {o.id for o in sc.objects}

        axes_objs = [o for o in active.values() if o.kind == "axes"]
        plotted_objs = [
            o
            for o in active.values()
            if o.kind in ("data_series", "parametric_curve", "function_plot", "vector_field", "bar_chart", "mechanical_body")
        ]

        # 1. Bare axes without any curve / plotted object or zero-span collapsed data series
        if axes_objs and not plotted_objs:
            pts_only = [o for o in active.values() if o.kind == "point"]
            sc_issues.append(
                f"bare_axes: scene '{sc.scene_id}' has axes {[a.id for a in axes_objs]} without curve/trajectory objects"
                + (f" (only points {[p.id for p in pts_only]})" if pts_only else "")
            )

        for o in sc.objects:
            pts_o = o.points_list or o.vertices
            if o.kind in ("data_series", "parametric_curve") and pts_o:
                if len(pts_o) >= 2 and _nonfinite_or_zero_span(pts_o):
                    sc_issues.append(f"collapsed_curve: '{o.id}' has zero spatial span (likely equilibrium initial condition)")
            elif o.kind == "data_series" and not pts_o and not o.model_id and not o.expr:
                sc_issues.append(f"empty_curve: '{o.id}' has no plotted points and no model_id")

        # 2. Curve exceeds owning axes bounds on non-zoomed views
        axes_by_id = {a.id: a for a in axes_objs}
        is_zoomed = (sc.composition == "spotlight_zoom") or any(ev.op in ("zoom", "spotlight") for ev in sc.events)
        for o in active.values():
            pts_c = o.points_list or o.vertices
            if o.kind in ("data_series", "parametric_curve") and pts_c and o.axes_id in axes_by_id and not is_zoomed:
                ax = axes_by_id[o.axes_id]
                if ax.x_range and ax.y_range and len(ax.x_range) >= 2 and len(ax.y_range) >= 2:
                    ax_x0, ax_x1 = float(ax.x_range[0]), float(ax.x_range[1])
                    ax_y0, ax_y1 = float(ax.y_range[0]), float(ax.y_range[1])
                    xs = [float(p[0]) for p in pts_c if len(p) >= 2]
                    ys = [float(p[1]) for p in pts_c if len(p) >= 2]
                    if xs and ys:
                        dx = max(1e-6, ax_x1 - ax_x0)
                        dy = max(1e-6, ax_y1 - ax_y0)
                        if min(xs) < ax_x0 - 0.08 * dx or max(xs) > ax_x1 + 0.08 * dx or min(ys) < ax_y0 - 0.08 * dy or max(ys) > ax_y1 + 0.08 * dy:
                            sc_issues.append(
                                f"clipped_curve: '{o.id}' span [{min(xs):.2f}..{max(xs):.2f}, {min(ys):.2f}..{max(ys):.2f}] exceeds axes '{ax.id}' [{ax_x0:.2f}..{ax_x1:.2f}, {ax_y0:.2f}..{ax_y1:.2f}]"
                            )

        # 3. Multiple large containers lacking disjoint panel slots
        containers = [
            o for o in active.values()
            if o.kind in ("axes", "matrix", "table", "bar_chart", "graph_network", "fraction_bar", "fraction_circle", "array_grid", "counting_group", "equation_balance", "concept_card")
            or (o.kind == "mechanical_body" and not o.axes_id)
        ]
        if len(containers) >= 2:
            slots = [getattr(c, "panel_slot", "auto") for c in containers]
            missing = [c.id for c in containers if getattr(c, "panel_slot", "auto") in ("auto", "")]
            non_null = [s for s in slots if s and s != "auto"]
            if missing or len(non_null) != len(set(non_null)):
                sc_issues.append(f"panel_slot_collision: containers {[c.id for c in containers]} share/lack panel_slots ({slots})")

        # 4. Dynamic trace/animate_param targeting an unlinked static point or unlinked mechanical_body
        for ev in sc.events:
            if ev.op in ("trace", "animate_param") and ev.target_id in active:
                tgt = active[ev.target_id]
                if tgt.kind == "point" and not (tgt.model_id and tgt.model_output) and not (tgt.param_name and tgt.expr and ";" in tgt.expr):
                    sc_issues.append(f"unlinked_motion_target: '{ev.op}' on static point '{tgt.id}' lacking model_id/model_output")
                elif tgt.kind == "mechanical_body" and not (tgt.model_id and tgt.model_output):
                    sc_issues.append(f"unlinked_motion_target: '{ev.op}' on mechanical_body '{tgt.id}' lacking model_id/model_output")

        # 5. Unresolved '?' placeholder text in scene labels/equations
        for o in sc.objects:
            if o.id in local_ids:
                txt = f"{o.text or ''} {o.tex or ''}".strip()
                if txt and _QUESTION_TOKEN_RE.search(txt):
                    sc_issues.append(f"placeholder_text: '{o.id}' contains unresolved '?' placeholder: '{txt[:40]}'")

        # 6. Merge optional rendered FrameAuditResult findings mentioning this scene_id
        if frame_audit is not None:
            for item in list(getattr(frame_audit, "findings", [])) + list(getattr(frame_audit, "warnings", [])):
                s_item = str(item)
                if sc.scene_id in s_item and s_item not in sc_issues:
                    sc_issues.append(f"frame_audit: {s_item}")

        if sc_issues:
            defects[sc.scene_id] = sc_issues

        for obj in sc.objects:
            declared_prior[obj.id] = obj
            removed.discard(obj.id)
        for ev in sc.events:
            if ev.op == "remove" and ev.target_id:
                removed.add(ev.target_id)

    return defects


def _infer_non_equilibrium_pair(x_v: float, y_v: float) -> tuple[float, float]:
    return (
        round(float(x_v) * 1.45 + (1.0 if abs(x_v) < 1e-4 else 0.0), 3),
        round(float(y_v) * 0.72 + (0.5 if abs(y_v) < 1e-4 else 0.0), 3),
    )


def repair_scene_deterministically(
    plan: VideoPlanV3,
    scene_id: str | None = None,
) -> tuple[VideoPlanV3, list[str]]:
    """Repair layout collisions, unlinked points/bodies, bare axes, collapsed equilibrium starts, and '?' placeholders deterministically."""
    out = plan.model_copy(deep=True)
    notes: list[str] = []
    models_by_id = {m.model_id: m for m in out.math_models}

    # First pass: fix collapsed equilibrium initial conditions on math_models AND ode_phase_trajectory constructions
    from studio.ai.linked_models import evaluate_linked_math_model

    for mod in out.math_models:
        if mod.kind == "dynamical_simulation" and len(mod.initial_state) >= 2:
            try:
                m_res = evaluate_linked_math_model(mod)
                pts_chk = m_res.get("phase_orbit") or m_res.get("time_series_x") or []
                if pts_chk and _nonfinite_or_zero_span(pts_chk):
                    nx, ny = _infer_non_equilibrium_pair(mod.initial_state[0], mod.initial_state[1])
                    mod.initial_state = [nx, ny]
                    notes.append(f"global: shifted collapsed equilibrium model '{mod.model_id}' initial_state -> {[nx, ny]}")
            except Exception:
                pass

    for sc in out.scenes:
        if scene_id and sc.scene_id != scene_id:
            continue
        collapsed_axes: set[str] = set()
        kept_objs: list[SceneObject] = []
        cons_out_ids = {oid for c in sc.constructions for oid in c.output_ids}
        for o in sc.objects:
            pts_o = o.points_list or o.vertices
            if o.kind in ("data_series", "parametric_curve") and pts_o and _nonfinite_or_zero_span(pts_o):
                if o.model_id and o.model_id in models_by_id:
                    o.points_list = []
                    kept_objs.append(o)
                else:
                    collapsed_axes.add(o.id)
                    if o.id not in cons_out_ids:
                        o.points_list = []
                        kept_objs.append(o)
                    continue
            kept_objs.append(o)
        sc.objects = kept_objs
        for c in sc.constructions:
            if c.op == "ode_phase_trajectory" and (collapsed_axes & set(c.output_ids) or not c.output_ids):
                p = c.params
                init_s = list(p.initial_state or [])
                x0 = float(p.x0 if p.x0 is not None else (init_s[0] if len(init_s) >= 1 else 1.5))
                y0 = float(p.y0 if p.y0 is not None else (init_s[1] if len(init_s) >= 2 else 0.0))
                nx, ny = _infer_non_equilibrium_pair(x0, y0)
                p.x0, p.y0 = nx, ny
                if len(init_s) >= 2:
                    p.initial_state = [nx, ny]
                notes.append(f"{sc.scene_id}: shifted collapsed construction '{c.op}' -> ({nx}, {ny})")

    out = materialize_constructions(out)
    models_by_id = {m.model_id: m for m in out.math_models}
    dyn_models = [m for m in out.math_models if m.kind == "dynamical_simulation"]
    any_models = list(out.math_models)

    # Second pass: track live carried objects across scenes so Scene 2 points/axes see Scene 1 models & curves
    carried_map: dict[str, SceneObject] = {}
    removed_ids: set[str] = set()
    created_ids: set[str] = set()

    for sc in out.scenes:
        active_map: dict[str, SceneObject] = {k: v for k, v in carried_map.items() if k not in removed_ids}
        for o in sc.objects:
            active_map[o.id] = o

        if not scene_id or sc.scene_id == scene_id:
            from studio.ai.visual_intelligence import humanize_visible_label
            for o in sc.objects:
                if o.label:
                    o.label = humanize_visible_label(o.label)
                if o.x_label:
                    o.x_label = humanize_visible_label(o.x_label)
                if o.y_label:
                    o.y_label = humanize_visible_label(o.y_label)
                if o.legend_items:
                    o.legend_items = [[humanize_visible_label(it[0]), *it[1:]] for it in o.legend_items if it]
                if o.text:
                    o.text = humanize_visible_label(o.text)
                if o.text and _QUESTION_TOKEN_RE.search(o.text.strip()):
                    cleaned = re.sub(r"\s*=\s*\?(?!\S)", "", o.text).strip(" ?:")
                    o.text = cleaned or o.id.replace("_", " ").title()
                    notes.append(f"{sc.scene_id}: cleaned placeholder text on '{o.id}' -> '{o.text}'")
                if o.tex and _QUESTION_TOKEN_RE.search(o.tex.strip()):
                    o.tex = re.sub(r"\s*=\s*\?(?!\S)", "", o.tex).strip(" ?")
                    notes.append(f"{sc.scene_id}: cleaned placeholder tex on '{o.id}'")
                if o.kind == "data_series" and not o.points_list and not o.expr and not o.model_id and (dyn_models or any_models):
                    ref_m = dyn_models[0] if dyn_models else any_models[0]
                    o.model_id = ref_m.model_id
                    o.model_output = "time_series_x" if ref_m.kind == "dynamical_simulation" else "response_curve"
                    notes.append(f"{sc.scene_id}: bound empty data_series '{o.id}' -> '{ref_m.model_id}.{o.model_output}'")

            # Add missing curve/trajectory onto bare axes when chapter has an active math_model
            axes_objs = [o for o in active_map.values() if o.kind == "axes"]
            plotted_objs = [
                o for o in active_map.values()
                if o.kind in ("data_series", "parametric_curve", "function_plot", "vector_field", "bar_chart", "mechanical_body")
            ]
            if axes_objs and not plotted_objs:
                pts_models = [o.model_id for o in active_map.values() if o.kind == "point" and o.model_id in models_by_id]
                ref_mid = pts_models[0] if pts_models else (dyn_models[0].model_id if dyn_models else (any_models[0].model_id if any_models else None))
                if ref_mid is not None:
                    ax_t = axes_objs[0]
                    cid_auto = f"{ax_t.id}_series"
                    if cid_auto not in active_map:
                        out_f = "time_series_x" if models_by_id[ref_mid].kind == "dynamical_simulation" else "response_curve"
                        new_curve = SceneObject(
                            id=cid_auto,
                            kind="data_series",
                            persist=True,
                            color_role="primary",
                            axes_id=ax_t.id,
                            model_id=ref_mid,
                            model_output=out_f,
                            label=models_by_id[ref_mid].label or out_f.replace("_", " "),
                        )
                        sc.objects.append(new_curve)
                        active_map[cid_auto] = new_curve
                        sc.events.insert(0, AnimationEvent(t=0.2, op="draw", target_id=cid_auto))
                        notes.append(f"{sc.scene_id}: added '{cid_auto}' ({ref_mid}.{out_f}) to populate bare axes '{ax_t.id}'")

            # Ensure mechanical_body or point targeted by trace/animate_param has valid model binding & create event
            active_curves = [o for o in active_map.values() if o.kind in ("data_series", "parametric_curve") and o.model_id]
            for ev in list(sc.events):
                if ev.target_id not in active_map and ev.target_id in models_by_id:
                    bound = next(
                        (o for o in active_map.values() if o.model_id == ev.target_id and o.kind in ("mechanical_body", "point", "data_series")),
                        next((o for o in active_map.values() if o.model_id == ev.target_id), None),
                    )
                    if bound is not None:
                        notes.append(f"{sc.scene_id}: remapped event target model_id '{ev.target_id}' -> '{bound.id}'")
                        ev.target_id = bound.id

                tgt = active_map.get(ev.target_id)
                if tgt is not None and ev.op in ("trace", "animate_param"):
                    if tgt.kind == "point" and not (tgt.model_id and tgt.model_output) and not (tgt.param_name and tgt.expr and ";" in tgt.expr):
                        ref_curve = next((c for c in active_curves if c.axes_id == tgt.axes_id), active_curves[0] if active_curves else None)
                        ref_mid = ref_curve.model_id if ref_curve else (dyn_models[0].model_id if dyn_models else None)
                        if ref_mid:
                            tgt.model_id = ref_mid
                            tgt.model_output = "time_point"
                            ev.op = "animate_param"
                            if not ev.param_range:
                                ev.param_range = [0.0, 1.0]
                            notes.append(f"{sc.scene_id}: linked point '{tgt.id}' -> '{ref_mid}.time_point'")
                        else:
                            ev.op = "focus"
                            notes.append(f"{sc.scene_id}: converted unlinked dot trace '{tgt.id}' -> 'focus'")
                    elif tgt.kind == "mechanical_body" and not (tgt.model_id and tgt.model_output):
                        ref_mid = tgt.model_id or (dyn_models[0].model_id if dyn_models else None)
                        if ref_mid:
                            tgt.model_id = ref_mid
                            tgt.model_output = "sweep"
                            if not ev.param_range:
                                ev.param_range = [0.0, 1.0]
                            notes.append(f"{sc.scene_id}: linked mechanical_body '{tgt.id}' -> '{ref_mid}.sweep'")

            # Auto-insert create event when scene animates/focuses an existing object before creation
            for ev in list(sc.events):
                if ev.op in ("create", "draw", "reveal", "trace", "sequence", "parallel"):
                    created_ids.add(ev.target_id)
                elif ev.target_id in active_map and ev.target_id not in created_ids:
                    sc.events.insert(0, AnimationEvent(t=0.0, op="create", target_id=ev.target_id))
                    created_ids.add(ev.target_id)
                    notes.append(f"{sc.scene_id}: prepended create for '{ev.target_id}' before '{ev.op}'")

        for o in sc.objects:
            carried_map[o.id] = o
            removed_ids.discard(o.id)
        for ev in sc.events:
            if ev.op in ("create", "draw", "reveal", "trace", "sequence", "parallel"):
                created_ids.add(ev.target_id)
            if ev.op == "remove" and ev.target_id:
                removed_ids.add(ev.target_id)

    # Re-materialize newly linked models + fit bounds & disjoint panel slots across carried objects
    out = materialize_constructions(out)
    out = apply_visual_intelligence_and_fit(out)
    return out, notes


def repair_plan_scenes(
    plan: VideoPlanV3,
    *,
    frame_audit: Any | None = None,
    provider: Any | None = None,
    preview_verify: bool = False,
    look_name: str = "light",
    out_dir: Path | None = None,
    max_scenes_preview: int = 2,
) -> tuple[VideoPlanV3, list[dict[str, Any]]]:
    """Diagnose per-scene visual defects, repair deterministically (+ optional LLM/preview verify), and return audit evidence."""
    orig_narration = {sc.scene_id: sc.narration for sc in plan.scenes}
    before = diagnose_scene_visual_defects(plan, frame_audit=frame_audit)
    repaired, det_notes = repair_scene_deterministically(plan)
    for sc in repaired.scenes:
        if sc.scene_id in orig_narration:
            sc.narration = orig_narration[sc.scene_id]
    after = diagnose_scene_visual_defects(repaired)
    evidence: list[dict[str, Any]] = []

    flagged_ids = sorted(set(before.keys()) | set(after.keys()))
    for sid in flagged_ids:
        b_list = before.get(sid, [])
        a_list = after.get(sid, [])
        evidence.append(
            {
                "scene_id": sid,
                "before_defects": b_list,
                "after_defects": a_list,
                "deterministic_notes": [n for n in det_notes if n.startswith(f"{sid}:")],
            }
        )

    if provider is not None and after:
        from studio.ai.v3_schema import SceneV3

        for sid, res_issues in list(after.items())[:2]:
            sc_idx = next((i for i, s in enumerate(repaired.scenes) if s.scene_id == sid), -1)
            if sc_idx < 0:
                continue
            sc_obj = repaired.scenes[sc_idx]
            prompt = (
                f"Repair ONLY visual objects/events/panel_slots in this single SceneV3 '{sid}' while keeping scene_id and spoken narration IDENTICAL.\n"
                f"Issues to resolve:\n- " + "\n- ".join(res_issues)
                + f"\nAvailable math_models: {[m.model_id for m in repaired.math_models]}\n"
                + f"Scene JSON:\n{sc_obj.model_dump_json(exclude_none=True)}"
            )
            try:
                fixed_sc, _ = provider.generate_structured(prompt, SceneV3, temperature=0.2)
                fixed_sc.scene_id = sid
                fixed_sc.narration = orig_narration.get(sid, sc_obj.narration)
                cand = repaired.model_copy(deep=True)
                cand.scenes[sc_idx] = fixed_sc
                cand = apply_visual_intelligence_and_fit(materialize_constructions(cand))
                repaired = cand
            except Exception:
                pass

    if preview_verify and flagged_ids:
        from studio.preview import render_single_scene_preview

        preview_dir = out_dir or (Path("build") / "repair_previews")
        for sid in flagged_ids[: max(1, max_scenes_preview)]:
            try:
                prev_res = render_single_scene_preview(
                    repaired,
                    sid,
                    out_dir=preview_dir / sid,
                    look_name=look_name,
                    backend="hyperframes",
                    with_tts=False,
                )
                for e in evidence:
                    if e["scene_id"] == sid:
                        e["preview_qa_score"] = prev_res.get("qa_score")
                        e["preview_state"] = prev_res.get("production_state")
                        e["preview_video_path"] = prev_res.get("video_path")
                        e["preview_contact_sheet"] = prev_res.get("contact_sheet")
            except Exception as exc:
                for e in evidence:
                    if e["scene_id"] == sid:
                        e["preview_error"] = str(exc)

    return copy.deepcopy(repaired), evidence
