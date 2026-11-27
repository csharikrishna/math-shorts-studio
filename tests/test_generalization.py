"""Comprehensive generalization & architectural tests across V4 adaptive capabilities."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from studio.ai.knowledge import ChapterSpec, KnowledgeNode, KnowledgeTree, LongFormPlan
from studio.ai.math_truth import (
    compose_expressions,
    definite_integral,
    discrete_pmf_convolution,
    fourier_partial_sum_terms,
    markov_chain_trajectory,
    matrix_transform_and_eigen,
    newton_raphson_trajectory,
    rk4_ode_trajectory,
    solve_equation_roots,
)
from studio.ai.v3_schema import (
    AnimationEvent,
    ConstructionOp,
    ConstructionParams,
    SceneObject,
    SceneV3,
    StructuredClaim,
    VideoPlanV3,
)
from studio.ai.validate_v3 import validate_v3_plan, verify_v3_plan_detailed
from studio.ai.world_model import materialize_constructions
from studio.creative import (
    ConceptAnalysis,
    CreativeBrief,
    DIALOGUE_PROFILES,
    NarrationBeatIR,
    ScriptIR,
    StorylineOption,
    detect_banned_cliches,
    export_script_md,
    export_storylines_md,
    script_ir_to_beat_specs,
    validate_script_ir,
)
from studio.design.spec import NAMED_LOOKS, default_design_spec, export_design_md
from studio.qa.engine import audit_educational_quality
from studio.registry import adapt_composition_for_aspect, capabilities_prompt_summary, get_default_registry
from studio.renderers.hyperframes_backend import compile_plan_to_html, validate_safe_relpath
from studio.storage.cache import ArtifactCache, JobStore
from studio.storyboard import export_review_md, export_storyboard_md, storyboard_from_v3_plan


class TestGeneralization(unittest.TestCase):
    def test_general_computational_engines_across_domains(self) -> None:
        # 1. Definite integral & Riemann rectangles on arbitrary trig/poly integrand
        integ = definite_integral("sin(x) + 1", 0.0, math.pi, n=20, method="midpoint")
        self.assertAlmostEqual(float(integ["integral_exact"]), 2.0 + math.pi, places=4)  # type: ignore[arg-type]
        self.assertEqual(len(integ["boxes"]), 20)  # type: ignore[arg-type]

        # 2. RK4 harmonic oscillator preserves energy x^2 + y^2 ~= 1.0
        traj = rk4_ode_trajectory("y", "-x", x0=1.0, y0=0.0, dt=0.05, steps=80)
        self.assertEqual(len(traj), 81)
        _, x_end, y_end = traj[-1]
        self.assertAlmostEqual(x_end * x_end + y_end * y_end, 1.0, places=3)

        # 3. 2x2 matrix eigen decomposition regimes
        self.assertEqual(matrix_transform_and_eigen([[2.0, 1.0], [1.0, 2.0]])["regime"], "unstable_node")
        self.assertEqual(matrix_transform_and_eigen([[1.0, 2.0], [2.0, 1.0]])["regime"], "saddle")
        self.assertEqual(matrix_transform_and_eigen([[-0.3, -1.0], [1.0, -0.3]])["regime"], "stable_spiral")

        # 4. Arbitrary non-uniform discrete PMF convolution (skewed 3-outcome RV convolved 3x)
        xs, ps, mu, sig = discrete_pmf_convolution([0.0, 2.0, 5.0], [0.5, 0.3, 0.2], n_trials=3)
        self.assertAlmostEqual(sum(ps), 1.0, places=6)
        self.assertAlmostEqual(mu, 3.0 * (0.0 * 0.5 + 2.0 * 0.3 + 5.0 * 0.2), places=4)
        self.assertGreater(sig, 0.0)
        self.assertGreater(len(xs), 4)

        # 5. General Fourier inner-product projection for arbitrary periodic waveform f(x)=x
        fs = fourier_partial_sum_terms("x", period=2.0 * math.pi, n_harmonics=4)
        self.assertAlmostEqual(float(fs["a0"]), 0.0, places=3)  # type: ignore[arg-type]
        self.assertIn("sin(", str(fs["expr"]))

        # 6. Newton-Raphson & equation roots & composition
        nr = newton_raphson_trajectory("x**2 - 2", x0=2.0, steps=5)
        self.assertAlmostEqual(nr[-1][0], math.sqrt(2.0), places=5)
        roots = solve_equation_roots("x**2 - 1", "0", -3.0, 3.0)
        self.assertEqual(roots, [-1.0, 1.0])
        self.assertEqual(compose_expressions("x**2 + 1", "x - 1"), "(x - 1)**2 + 1")

        # 7. Markov chain stationary convergence
        mk = markov_chain_trajectory([[0.9, 0.1], [0.2, 0.8]], [1.0, 0.0], steps=25)
        stat = mk["stationary"]  # type: ignore[index]
        self.assertAlmostEqual(stat[0], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(stat[1], 1.0 / 3.0, places=4)

    def test_unseen_multi_domain_constructions_objects_events_and_claims(self) -> None:
        plan = VideoPlanV3(
            topic="Eigenvalues, Phase Space, and Markov Equilibrium",
            title="Dynamical Eigenvalues",
            scenes=[
                SceneV3(
                    scene_id="eigen_intro",
                    purpose="Inspect matrix eigenvalues",
                    composition="main_with_equation_panel",
                    narration="Look <bookmark mark='m1'/> at the two by two matrix stretching along <bookmark mark='m2'/> invariant eigenvectors.",
                    constructions=[
                        ConstructionOp(
                            op="eigen_decomposition_2x2",
                            output_ids=["mat_a", "v1", "v2"],
                            params=ConstructionParams(matrix_2x2=[[2.0, 1.0], [1.0, 2.0]], axes_id="phase_ax"),
                        ),
                        ConstructionOp(
                            op="ode_phase_trajectory",
                            output_ids=["orbit"],
                            params=ConstructionParams(dx_expr="-0.2*x - y", dy_expr="x - 0.2*y", x0=2.0, y0=0.0, dt=0.1, steps=30, axes_id="phase_ax"),
                        ),
                    ],
                    claims=[
                        StructuredClaim(kind="eigenvalue_determinant", object_ids=["mat_a"], matrix_2x2=[[2.0, 1.0], [1.0, 2.0]], expected_value=3.0),
                        StructuredClaim(kind="ode_equilibrium", dx_expr="-0.2*x - y", dy_expr="x - 0.2*y", x_val=0.0, expected_value=0.0),
                    ],
                    objects=[
                        SceneObject(id="phase_ax", kind="axes", x_range=[-3, 3], y_range=[-3, 3], persist=True),
                        SceneObject(id="eq_det", kind="equation", tex="det(A - \\lambda I) = 0", card=True, persist=False),
                    ],
                    events=[
                        AnimationEvent(op="create", target_id="phase_ax"),
                        AnimationEvent(op="reveal", target_id="mat_a", anchor_mark="m1"),
                        AnimationEvent(op="trace", target_id="orbit", anchor_mark="m2"),
                    ],
                ),
                SceneV3(
                    scene_id="markov_view",
                    purpose="Compare network graph & stationary distribution",
                    composition="side_by_side_comparison",
                    narration="Every transition weight moves probability mass toward a fixed <bookmark mark='m3'/> stationary balance.",
                    constructions=[
                        ConstructionOp(
                            op="markov_chain_evolution",
                            output_ids=["stat_bars"],
                            params=ConstructionParams(transition_matrix=[[0.8, 0.2], [0.4, 0.6]], initial_state=[1.0, 0.0], steps=10),
                        ),
                    ],
                    claims=[
                        StructuredClaim(kind="markov_stochastic_invariant", object_ids=["stat_bars"], statement="Row sums equal 1 and pi*P = pi."),
                        StructuredClaim(kind="probability_normalization", object_ids=["stat_bars"]),
                    ],
                    objects=[
                        SceneObject(
                            id="mc_graph",
                            kind="graph_network",
                            points_list=[[-1.5, 0.0], [1.5, 0.0]],
                            node_labels=["A", "B"],
                            edge_pairs=[["0", "1", "0.2"], ["1", "0", "0.4"]],
                            panel_slot="left",
                        ),
                        SceneObject(
                            id="iter_tbl",
                            kind="table",
                            table_headers=["Step", "P(A)", "P(B)"],
                            table_rows=[["0", "1.00", "0.00"], ["10", "0.67", "0.33"]],
                            panel_slot="right",
                        ),
                    ],
                    events=[
                        AnimationEvent(op="sequence", target_id="mc_graph"),
                        AnimationEvent(op="create", target_id="stat_bars", anchor_mark="m3"),
                    ],
                ),
            ],
        )

        mat_plan = materialize_constructions(plan)
        det = verify_v3_plan_detailed(mat_plan, require_claims=True)
        self.assertEqual(det["issues"], [])
        self.assertGreaterEqual(len(det["verified"]), 4)  # type: ignore[arg-type]
        self.assertEqual(det["verification_coverage"]["failed"], 0)  # type: ignore[index]

        # Negative verification test: wrong determinant must fail cleanly
        bad_plan = mat_plan.model_copy(deep=True)
        bad_plan.scenes[0].claims[0].expected_value = 99.0
        bad_issues = validate_v3_plan(bad_plan, require_claims=True)
        self.assertTrue(any("eigenvalue_determinant" in msg for msg in bad_issues))

        # Regression test: malformed list/syntax in scalar claim expr yields clean issue, not uncaught ValueError
        bad_expr_plan = mat_plan.model_copy(deep=True)
        bad_expr_plan.scenes[0].claims.append(
            StructuredClaim(kind="coefficient_correctness", expr="[2, 1]", expected_degree=1, expected_value=2.0)
        )
        bad_expr_issues = validate_v3_plan(bad_expr_plan)
        self.assertTrue(any("coefficient_correctness" in msg for msg in bad_expr_issues))

    def test_creative_dialogue_script_registry_storyboard_and_qa(self) -> None:
        reg = get_default_registry()
        self.assertGreaterEqual(len(reg.list_by_category("math_operation")), 8)
        self.assertGreaterEqual(len(reg.list_by_category("visual_object")), 20)
        summary = capabilities_prompt_summary(backend="hyperframes", aspect="9:16")
        self.assertIn("ode_rk4_trajectory", summary)
        self.assertIn("stacked_comparison", summary)
        self.assertEqual(adapt_composition_for_aspect("side_by_side_comparison", "9:16"), "stacked_comparison")

        # All 4 named looks and aspect specs
        for lk in NAMED_LOOKS:
            ds_w = default_design_spec("widescreen_hd", lk)
            ds_p = default_design_spec("shorts_hd", lk)
            self.assertFalse(ds_w.safe_zone.prefer_stacked_comparison)
            self.assertTrue(ds_p.safe_zone.prefer_stacked_comparison)
            self.assertIn(lk, export_design_md(ds_p))

        # Socratic dialogue & ScriptIR checks
        self.assertGreater(len(detect_banned_cliches("Welcome to a mathematical journey!")), 0)
        self.assertIn("socratic", DIALOGUE_PROFILES)
        concept = ConceptAnalysis(topic="Stability", meaning="Decay vs growth", central_question="When do perturbations die out?", learning_outcome="Classify fixed points")
        brief = CreativeBrief(
            concept=concept,
            storylines=[StorylineOption(option_id="s1", name="Phase Plane", strategy="visual_proof", narrative_device="Contrast orbits", visual_form="Phase portrait", opening_two_seconds="Two dots diverge", beat_arc="Setup -> trace -> eigen", recommended=True)],
            selected_storyline_id="s1",
            selected_strategy="visual_proof",
            narration_mode="socratic",
        )
        self.assertIn("Phase Plane", export_storylines_md(brief))
        script = ScriptIR(
            topic="Stability",
            title="Dynamical Stability",
            narration_mode="socratic",
            beats=[
                NarrationBeatIR(beat_id="b1", scene_id="sc1", speaker="learner", spoken_text="Wait—if two vectors <bookmark mark='k1'/> point mostly right, shouldn't they move together?", educational_purpose="Surface intuition", visual_intent="Two vectors side by side", anchor_marks=["k1"]),
                NarrationBeatIR(beat_id="b2", scene_id="sc2", speaker="educator", spoken_text="Watch the grid stretch along each eigenvalue axis.", educational_purpose="Resolve via visual action", visual_intent="Transform grid"),
            ],
        )
        self.assertEqual(validate_script_ir(script), [])
        specs = script_ir_to_beat_specs(script)
        self.assertEqual([sp.speaker for sp in specs], ["learner", "educator"])
        self.assertIn("Dynamical Stability", export_script_md(script))

        # Storyboard + VideoReviewReport + HTML compiler
        v3p = VideoPlanV3(
            topic="Stability",
            title="Dynamical Stability",
            scenes=[
                SceneV3(scene_id="sc1", purpose="Question", speaker="learner", composition="side_by_side_comparison", narration="Wait—watch <bookmark mark='k1'/> the curve.", objects=[SceneObject(id="ax", kind="axes", x_range=[-2, 2], y_range=[-2, 2], panel_slot="left"), SceneObject(id="fn", kind="function_plot", expr="x**2", axes_id="ax")], events=[AnimationEvent(op="draw", target_id="fn", anchor_mark="k1")]),
            ],
        )
        sb = storyboard_from_v3_plan(v3p, format_name="shorts_hd", script=script)
        self.assertEqual(sb.shots[0].composition, "stacked_comparison")
        self.assertIn("shot_01_sc1", export_storyboard_md(sb))

        html_doc, total_s = compile_plan_to_html(v3p, default_design_spec("shorts_hd", "editorial_paper"))
        self.assertGreater(total_s, 0.5)
        self.assertNotIn("gsap", html_doc.lower())
        self.assertNotIn("cdn.jsdelivr", html_doc.lower())
        self.assertIn("__HF_SEEK", html_doc)

        rep = audit_educational_quality(topic="Stability", title="Dynamical Stability", plan=v3p, format_name="shorts_hd", backend="hyperframes", script=script)
        self.assertIn("Claim Verification Coverage", export_review_md(rep))  # type: ignore[arg-type]

        # Multi-chapter audit must NOT flag duplicate scene_id across independent chapters
        rep_multi = audit_educational_quality(
            topic="Stability",
            title="Dynamical Stability Multi",
            plan=v3p,
            format_name="widescreen_hd",
            backend="manimgx",
            chapter_plans=[v3p, v3p],
        )
        self.assertTrue(rep_multi.approved)  # type: ignore[attr-defined]
        self.assertGreaterEqual(rep_multi.overall_score, 90.0)  # type: ignore[attr-defined]

        from studio.renderers.hyperframes_backend import _current_process_rss_mb, _pretty_math_label

        self.assertEqual(_pretty_math_label(r"\frac{d\mathbf{x}}{dt} = A\mathbf{x}"), "dx/dt = Ax")
        self.assertGreater(_current_process_rss_mb(), 0.0)

    def test_dependency_cache_and_loopback_security(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertIsNone(validate_safe_relpath(root, "/../secret.json"))
            self.assertIsNone(validate_safe_relpath(root, "/%2e%2e/secret.json"))
            self.assertIsNone(validate_safe_relpath(root, "/.env"))
            self.assertIsNone(validate_safe_relpath(root, "/dir\\file"))
            self.assertEqual(validate_safe_relpath(root, "/index.html"), (root.resolve() / "index.html"))

            art = root / "chapter1.mp4"
            art.write_bytes(b"fake_video_bytes_v1")
            cache = ArtifactCache(db_path=root / "jobs.db")
            try:
                up = {"knowledge": "hash_k1", "design": "hash_d1"}
                hit, key, p, reason = cache.lookup("chapter_assembly", "ch1", {"fps": 60}, upstream_keys=up)
                self.assertFalse(hit)
                self.assertIsNone(p)
                self.assertEqual(reason, "miss:not_recorded")

                cache.store_artifact("chapter_assembly", "ch1", key, art, upstream_keys=up)
                hit2, _k2, p2, r2 = cache.lookup("chapter_assembly", "ch1", {"fps": 60}, upstream_keys=up)
                self.assertTrue(hit2)
                self.assertEqual(p2, art)
                self.assertEqual(r2, "hit")

                # Changing upstream knowledge hash invalidates downstream stage
                hit3, _k3, _p3, r3 = cache.lookup("chapter_assembly", "ch1", {"fps": 60}, upstream_keys={"knowledge": "hash_k2", "design": "hash_d1"})
                self.assertFalse(hit3)
                self.assertEqual(r3, "miss:dependency_or_payload_changed")

                # Corrupting cached file on disk triggers checksum mismatch
                art.write_bytes(b"corrupted_bytes")
                hit4, _k4, _p4, r4 = cache.lookup("chapter_assembly", "ch1", {"fps": 60}, upstream_keys=up)
                self.assertFalse(hit4)
                self.assertEqual(r4, "miss:checksum_mismatch")
            finally:
                cache.close()


if __name__ == "__main__":
    unittest.main()
