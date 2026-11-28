"""V4.1 reliability, typed mathematical constraints, visual intelligence, layout separation, QA & repair tests."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio.ai.linked_models import (
    LinkedMathModelSpec,
    evaluate_dynamical_simulation_model,
    evaluate_frequency_response_model,
)
from studio.ai.simulation import (
    build_simulation_sweep_frames,
    compute_driven_frequency_response,
    integrate_ode_system,
)
from studio.ai.v3_schema import AnimationEvent, SceneObject, SceneV3, StructuredClaim, VideoPlanV3
from studio.ai.validate_v3 import validate_v3_plan
from studio.ai.visual_intelligence import apply_visual_intelligence_and_fit, infer_beat_visual_decision
from studio.ai.world_model import materialize_constructions
from studio.audio.tts import find_ffmpeg
from studio.design.spec import default_design_spec
from studio.layout.solver import compute_adaptive_zones, fit_axes_pixel_box
from studio.provenance import (
    build_and_write_manifest,
    finalize_manifest_qa,
    inspect_manifest_verification,
)
from studio.qa.engine import audit_educational_quality
from studio.qa.frames import inspect_video_scene_aware
from studio.registry import get_default_registry
from studio.renderers.hyperframes.compiler import compile_plan_to_html
from studio.repair import diagnose_scene_visual_defects, repair_plan_scenes


class TestV41ReliabilityAndVisualIntelligence(unittest.TestCase):
    def test_01_cash_karp_rk45_vs_rk4_adaptive_and_tolerance_rejection(self) -> None:
        # True 6-stage Cash-Karp RK45 preserves unit circle x'=-y, y'=x with high precision
        res_45 = integrate_ode_system(
            state_vars=["x", "y"],
            rhs_exprs=["-y", "x"],
            initial_state=[1.0, 0.0],
            t_range=[0.0, 6.283185307179586],
            steps=80,
            method="rk45",
            rtol=1e-7,
            atol=1e-9,
        )
        self.assertEqual(res_45.solver, "rk45")
        self.assertEqual(res_45.summary["solver_algorithm"], "cash_karp_rk45")
        self.assertLess(abs(res_45.states[-1][0] - 1.0), 1e-5)

        res_adp = integrate_ode_system(
            state_vars=["x", "y"],
            rhs_exprs=["-y", "x"],
            initial_state=[1.0, 0.0],
            t_range=[0.0, 6.283185307179586],
            steps=80,
            method="rk4_adaptive",
        )
        self.assertEqual(res_adp.solver, "rk4_adaptive")
        self.assertEqual(res_adp.summary["solver_algorithm"], "rk4_step_doubling")

        # Stiff / impossible tolerance must raise instead of silently accepting inaccurate output
        with self.assertRaises(ValueError):
            integrate_ode_system(
                state_vars=["x", "y"],
                rhs_exprs=["-500*x", "-500*y"],
                initial_state=[1.0, 1.0],
                t_range=[0.0, 2.0],
                steps=12,
                method="rk45",
                rtol=1e-18,
                atol=1e-18,
            )

    def test_02_no_silent_fallbacks_or_singularity_fabrications(self) -> None:
        # Missing ODE RHS must raise, not fall back to a harmonic oscillator
        bad_rhs = LinkedMathModelSpec(model_id="lv_bad", kind="dynamical_simulation", state_vars=["x", "y"], rhs_exprs=[], initial_state=[10.0, 5.0])
        with self.assertRaises(ValueError):
            evaluate_dynamical_simulation_model(bad_rhs)

        # Mismatched initial_state dimension must raise, not pad defaults
        bad_ic = LinkedMathModelSpec(model_id="dim_bad", kind="dynamical_simulation", state_vars=["x", "y"], rhs_exprs=["y", "-x"], initial_state=[1.0])
        with self.assertRaises(ValueError):
            evaluate_dynamical_simulation_model(bad_ic)

        # Variable names ["x", "v"] must NOT invent mechanical energy unless mass & stiffness are explicitly declared
        no_physics = evaluate_dynamical_simulation_model(
            LinkedMathModelSpec(model_id="abstract_xy", kind="dynamical_simulation", state_vars=["x", "v"], rhs_exprs=["v", "-x"], initial_state=[1.0, 0.0], steps=40)
        )
        self.assertNotIn("total_energy", no_physics["derived"])

        with_physics = evaluate_dynamical_simulation_model(
            LinkedMathModelSpec(
                model_id="phys_osc",
                kind="dynamical_simulation",
                state_vars=["x", "v"],
                rhs_exprs=["v", "-4.0*x"],
                initial_state=[1.0, 0.0],
                mass=1.0,
                stiffness=4.0,
                damping_coeff=0.0,
                steps=40,
            )
        )
        self.assertIn("total_energy", with_physics["derived"])

        # Acceleration must NOT be copied as restoring_force unless force_expr or mass is explicit
        frames_no_force = build_simulation_sweep_frames(
            integrate_ode_system(state_vars=["x", "v"], rhs_exprs=["v", "-x"], initial_state=[1.0, 0.0], steps=12),
            n_keyframes=6,
        )
        self.assertTrue(all("restoring_force" not in fr for fr in frames_no_force))

        # Undamped resonance (zeta == 0) must flag singularity and NOT invent 1e4 finite amplitude
        undamped = compute_driven_frequency_response(natural_freq=2.0, damping_ratio=0.0, drive_amp=1.0, freq_range=[0.2, 4.0], n_pts=80)
        self.assertTrue(undamped["has_singularity"])
        self.assertIsNone(undamped["peak_gain"])
        ev_undamped = evaluate_frequency_response_model(
            LinkedMathModelSpec(model_id="res0", kind="frequency_response", natural_freq=2.0, damping_ratio=0.0, drive_amp=1.0)
        )
        self.assertTrue(ev_undamped["measurements"]["has_singularity"])

    def test_03_typed_mathematical_constraints_and_hollow_trace_rejection(self) -> None:
        # Correct Lotka-Volterra model + typed positivity & conservation constraints passes; wrong bounds fail!
        lv_model = LinkedMathModelSpec(
            model_id="lv",
            kind="dynamical_simulation",
            state_vars=["x", "y"],
            rhs_exprs=["0.5*x - 0.1*x*y", "0.02*x*y - 0.3*y"],
            initial_state=[22.0, 8.0],
            t_range=[0.0, 20.0],
            steps=160,
            derived_exprs={"conserved_invariant": "0.02*x - 0.3*log(x) + 0.1*y - 0.5*log(y)"},
            units={"x": "prey", "y": "predators", "t": "years"},
            assumptions=["Closed two-species continuous Lotka-Volterra system without carrying-capacity limit"],
        )
        ok_plan = VideoPlanV3(
            topic="Predator Prey",
            title="Lotka-Volterra",
            math_models=[lv_model],
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Positive bounded predator-prey cycle",
                    narration="Prey and predators stay strictly positive and conserve the orbital invariant.",
                    objects=[
                        SceneObject(id="ax", kind="axes", x_range=[0, 20, 5], y_range=[0, 40, 10]),
                        SceneObject(id="prey", kind="data_series", axes_id="ax", model_id="lv", model_output="time_series_x"),
                        SceneObject(id="dot", kind="point", axes_id="ax", model_id="lv", model_output="time_point"),
                    ],
                    events=[
                        AnimationEvent(t=0.0, op="create", target_id="ax"),
                        AnimationEvent(t=0.2, op="draw", target_id="prey"),
                        AnimationEvent(t=0.4, op="create", target_id="dot"),
                        AnimationEvent(t=0.6, op="animate_param", target_id="dot", param_range=[0.0, 1.0]),
                    ],
                    claims=[
                        StructuredClaim(kind="positivity", model_id="lv", state_var="x"),
                        StructuredClaim(kind="state_bounds", model_id="lv", state_var="x", lower_bound=1.0, upper_bound=60.0),
                        StructuredClaim(kind="state_bounds", model_id="lv", state_var="conserved_invariant", lower_bound=-5.0, upper_bound=5.0),
                        StructuredClaim(kind="conservation_law", model_id="lv", expected_value=0.05),
                        StructuredClaim(kind="physical_units", model_id="lv", units={"x": "prey", "y": "predators"}),
                    ],
                )
            ],
        )
        ok_mat = materialize_constructions(ok_plan)
        self.assertEqual(validate_v3_plan(ok_mat, require_claims=True), [])

        # Violating asserted state_bounds OR trace on unlinked point must fail validation
        bad_plan = ok_mat.model_copy(deep=True)
        bad_plan.scenes[0].claims.append(StructuredClaim(kind="state_bounds", model_id="lv", state_var="x", lower_bound=0.0, upper_bound=12.0))
        bad_plan.scenes[0].objects.append(SceneObject(id="orphan", kind="point", axes_id="ax", coords=[10.0, 10.0]))
        bad_plan.scenes[0].events.append(AnimationEvent(t=0.8, op="create", target_id="orphan"))
        bad_plan.scenes[0].events.append(AnimationEvent(t=1.0, op="trace", target_id="orphan"))
        errs = validate_v3_plan(bad_plan, require_claims=True)
        self.assertTrue(any("state_bounds" in e for e in errs), errs)
        self.assertTrue(any("static unlinked point" in e for e in errs), errs)

    def test_04_reusable_compositions_layout_disjoint_and_visual_intelligence(self) -> None:
        reg = get_default_registry()
        comp_names = {s.name for s in reg.all_specs() if s.category == "composition_template"}
        for cname in (
            "physical_system_with_displacement",
            "physical_system_with_phase_space",
            "dual_time_series",
            "parameter_sweep_with_response",
            "vector_field_with_trajectory",
            "state_with_conserved_chart",
        ):
            self.assertIn(cname, comp_names, f"Missing composition {cname}")

        # Left and right panel boxes must never overlap
        zones = compute_adaptive_zones("widescreen_hd", has_header_banner=False, has_equation_strip=False, caption_style="editorial")
        bl = fit_axes_pixel_box(zones, xr=[0.0, 10.0], yr=[0.0, 10.0], equal_aspect=False, composition="side_by_side_comparison", panel_slot="left")
        br = fit_axes_pixel_box(zones, xr=[0.0, 10.0], yr=[0.0, 10.0], equal_aspect=False, composition="side_by_side_comparison", panel_slot="right")
        ix = max(0.0, min(bl.x2, br.x2) - max(bl.x, br.x))
        self.assertEqual(ix, 0.0, f"Left {bl} and Right {br} overlapped by {ix}px!")

        bt = fit_axes_pixel_box(zones, xr=[0.0, 10.0], yr=[0.0, 10.0], equal_aspect=False, composition="stacked_comparison", panel_slot="top")
        bb = fit_axes_pixel_box(zones, xr=[0.0, 10.0], yr=[0.0, 10.0], equal_aspect=False, composition="stacked_comparison", panel_slot="bottom")
        iy = max(0.0, min(bt.y2, bb.y2) - max(bt.y, bb.y))
        self.assertEqual(iy, 0.0, f"Top {bt} and Bottom {bb} overlapped by {iy}px!")

        class _DummyBeat:
            scene_id = "b_phys"
            educational_purpose = "Show how the spring-mass block drives sinusoidal displacement"
            visual_intent = "Animate spring block alongside displacement graph"
            spoken_text = "As the mass pulls on the spring, displacement rises and falls smoothly."
            transition_from_prev = "Moving from setup into live motion"
            latex_refs = ["F = -kx"]
            demo_contract = None

        vdec = infer_beat_visual_decision(_DummyBeat())
        self.assertEqual(vdec.composition, "physical_system_with_displacement")

        # apply_visual_intelligence_and_fit expands clipped axes bounds & assigns disjoint panel slots
        clipped_plan = VideoPlanV3(
            topic="Oscillator",
            title="Clipped Test",
            scenes=[
                SceneV3(
                    scene_id="s_fit",
                    purpose="Two views without overlap",
                    narration="Prey surges past initial axis limits.",
                    objects=[
                        SceneObject(id="ax_t", kind="axes", x_range=[0, 5, 1], y_range=[0, 10, 5]),
                        SceneObject(id="ax_p", kind="axes", x_range=[0, 5, 1], y_range=[0, 10, 5]),
                        SceneObject(id="c1", kind="data_series", axes_id="ax_t", points_list=[[0.0, 4.0], [10.0, 28.0], [20.0, 46.0]]),
                    ],
                )
            ],
        )
        fitted = apply_visual_intelligence_and_fit(clipped_plan)
        ax_t = next(o for o in fitted.scenes[0].objects if o.id == "ax_t")
        ax_p = next(o for o in fitted.scenes[0].objects if o.id == "ax_p")
        self.assertGreaterEqual(ax_t.x_range[1], 20.0)
        self.assertGreaterEqual(ax_t.y_range[1], 46.0)
        self.assertNotEqual(ax_t.panel_slot, ax_p.panel_slot)

    def test_05_qa_threshold_ceil_and_content_aware_frame_inspection(self) -> None:
        # 1/3 scenes animated = 33%, strictly below 65% (V4.0 bug `int(0.65*3)==1` missed this!)
        sparse_plan = VideoPlanV3(
            topic="Pacing check",
            title="Sparse Motion",
            scenes=[
                SceneV3(scene_id="s1", purpose="Moving curve", narration="Wave moves.", objects=[SceneObject(id="ax", kind="axes"), SceneObject(id="d", kind="data_series", axes_id="ax", points_list=[[0, 0], [1, 1], [2, 0]])], events=[AnimationEvent(t=0.0, op="create", target_id="ax"), AnimationEvent(t=0.2, op="draw", target_id="d")]),
                SceneV3(scene_id="s2", purpose="Static 1", narration="Second static.", objects=[SceneObject(id="t1", kind="annotation", text="Static A")], events=[AnimationEvent(t=0.0, op="create", target_id="t1")]),
                SceneV3(scene_id="s3", purpose="Static 2", narration="Third static.", objects=[SceneObject(id="t2", kind="annotation", text="Static B")], events=[AnimationEvent(t=0.0, op="create", target_id="t2")]),
            ],
        )
        rev = audit_educational_quality(topic="Pacing", title="Sparse", plan=sparse_plan, format_name="widescreen_hd", strict_pacing=True)
        self.assertEqual(rev.category_statuses.get("animation_meaningfulness"), "failed", f"Cats={rev.category_statuses}, Metrics={rev.metrics}")

        # Synthetic 1280x720 MP4 with BARE EMPTY AXES (only black axes arrows, no colored curve inside) must be flagged!
        ffmpeg = find_ffmpeg()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            img = Image.new("RGB", (1280, 720), (249, 248, 244))
            draw = ImageDraw.Draw(img)
            # Draw bare dark axes arrows only, nothing colored inside
            draw.line([(200, 520), (1080, 520)], fill=(18, 24, 38), width=3)
            draw.line([(200, 160), (200, 520)], fill=(18, 24, 38), width=3)
            img_p = tdp / "bare.png"
            mp4_p = tdp / "bare.mp4"
            meta_p = tdp / "meta.json"
            img.save(img_p)
            meta_p.write_text(
                json.dumps(
                    {
                        "s1": [
                            {"id": "ax1", "kind": "axes", "box": [180, 140, 450, 400]},
                            {"id": "ax2", "kind": "axes", "box": [420, 140, 450, 400]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            bare_plan = VideoPlanV3(
                topic="Bare Axes",
                title="Bare Axes",
                scenes=[
                    SceneV3(
                        scene_id="s1",
                        purpose="Empty bare axes",
                        narration="Empty stage.",
                        objects=[
                            SceneObject(id="ax1", kind="axes"),
                            SceneObject(id="ax2", kind="axes"),
                        ],
                    )
                ],
            )
            subprocess.run(
                [ffmpeg, "-y", "-loop", "1", "-i", str(img_p), "-t", "2.0", "-r", "60", "-pix_fmt", "yuv420p", str(mp4_p)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            fa = inspect_video_scene_aware(
                mp4_p,
                look_name="light",
                format_name="widescreen_hd",
                plan=bare_plan,
                renderer_meta_path=meta_p,
            )
            self.assertTrue(fa.empty_midpoint_scenes, f"Expected empty axes finding, got: {fa.findings}")
            self.assertTrue(fa.panel_overlap_findings, f"Expected overlapping panel finding, got: {fa.findings}")

    def test_06_qa_report_sha256_manifest_provenance(self) -> None:
        ffmpeg = find_ffmpeg()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            mp4 = tdp / "sample.mp4"
            qa_json = tdp / "qa_review.json"
            qa_json.write_text('{"overall_score": 88.0, "approved": true}', encoding="utf-8")
            subprocess.run(
                [
                    ffmpeg, "-y",
                    "-f", "lavfi", "-i", "color=c=white:s=640x360:r=60:d=1.0",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=1.0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(mp4),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            plan = VideoPlanV3(
                topic="Manifest Proof",
                title="Manifest Proof",
                scenes=[SceneV3(scene_id="s1", purpose="Ok", narration="One clear scene.")],
            )
            man, man_p = build_and_write_manifest(
                slug="proof",
                topic="Manifest Proof",
                title="Manifest Proof",
                mp4_path=mp4,
                plan=plan,
                backend="hyperframes",
                format_name="widescreen_hd",
                look_name="light",
                caption_style="editorial",
                review_path=qa_json,
            )
            self.assertTrue(bool(man.qa_report_sha256))
            rep = inspect_manifest_verification(man_p)
            self.assertTrue(rep["identity_verified"], rep)
            # Tampering with qa_review.json must fail verification
            qa_json.write_text('{"overall_score": 99.0, "approved": true}', encoding="utf-8")
            rep_bad = inspect_manifest_verification(man_p)
            self.assertFalse(rep_bad["identity_verified"])
            self.assertTrue(any("SHA-256 mismatch" in i for i in rep_bad["issues"]), rep_bad)

    def test_07_repair_loop_fixes_lv_equilibrium_overlap_and_placeholders(self) -> None:
        lv_eq = LinkedMathModelSpec(
            model_id="lv_eq",
            kind="dynamical_simulation",
            state_vars=["x", "y"],
            rhs_exprs=["0.5*x - 0.1*x*y", "0.02*x*y - 0.3*y"],
            initial_state=[15.0, 5.0],  # Exact fixed point -> collapsed trajectory!
            t_range=[0.0, 20.0],
            steps=100,
        )
        broken_plan = VideoPlanV3(
            topic="LV Repair",
            title="LV Repair",
            math_models=[lv_eq],
            scenes=[
                SceneV3(
                    scene_id="sc_fix",
                    purpose="Oscillation",
                    narration="Look at how the prey population rises first, then predators follow.",
                    objects=[
                        SceneObject(id="ax1", kind="axes", x_range=[0, 20, 5], y_range=[0, 10, 5]),
                        SceneObject(id="ax2", kind="axes", x_range=[0, 10, 5], y_range=[0, 10, 5]),
                        SceneObject(id="curve1", kind="data_series", axes_id="ax1", model_id="lv_eq", model_output="time_series_x"),
                        SceneObject(id="dot1", kind="point", axes_id="ax1", coords=[15.0, 5.0]),
                        SceneObject(id="lbl1", kind="annotation", text="Peak x = ?"),
                    ],
                    events=[
                        AnimationEvent(t=0.0, op="create", target_id="ax1"),
                        AnimationEvent(t=0.1, op="create", target_id="ax2"),
                        AnimationEvent(t=0.2, op="draw", target_id="curve1"),
                        AnimationEvent(t=0.5, op="trace", target_id="dot1"),
                    ],
                )
            ],
        )
        from studio.ai.world_model import materialize_constructions

        mat_broken = materialize_constructions(broken_plan)
        before_defects = diagnose_scene_visual_defects(mat_broken)
        self.assertIn("sc_fix", before_defects)

        repaired, evidence = repair_plan_scenes(mat_broken, preview_verify=False)
        self.assertTrue(evidence)
        # Locked spoken text preserved verbatim
        self.assertEqual(repaired.scenes[0].narration, broken_plan.scenes[0].narration)
        # No residual defects
        after_defects = diagnose_scene_visual_defects(repaired)
        self.assertNotIn("sc_fix", after_defects, f"Remaining defects: {after_defects}")

        # Disjoint panel slots and KaTeX HTML compilation succeeds cleanly
        html_str, _ = compile_plan_to_html(repaired, default_design_spec("widescreen_hd", "light"))
        self.assertIn("data-hf-role", html_str)
        self.assertIn("data-hf-elem", html_str)


if __name__ == "__main__":
    unittest.main()
