"""Behavioral regression tests for Math Shorts Studio V3.5."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from studio.ai.math_truth import bayes_posterior, discrete_convolution, gradient_descent_trajectory
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
from studio.audio.pronounce import normalize_for_speech_with_map
from studio.longform import chapter_cache_key
from studio.qa.engine import verify_longform_bundle


class TestV35Regressions(unittest.TestCase):
    def test_gradient_descent_point_and_minimum_rejected(self) -> None:
        plan = VideoPlanV3(
            topic="GD Bad",
            title="GD Bad",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="bad gd",
                    narration="Start at (2, 2) and end at (0, 0).",
                    objects=[
                        SceneObject(id="ax", kind="axes", x_range=[-3, 3], y_range=[-2, 3]),
                        SceneObject(id="fn", kind="function_plot", expr="sin(x)*cos(x)", axes_id="ax"),
                        SceneObject(id="p0", kind="point", coords=[2.0, 2.0], axes_id="ax"),
                        SceneObject(id="pmin", kind="point", coords=[0.0, 0.0], axes_id="ax"),
                    ],
                    claims=[
                        StructuredClaim(kind="point_on_function", object_ids=["p0", "fn"]),
                        StructuredClaim(kind="local_minimum", object_ids=["pmin", "fn"]),
                    ],
                )
            ],
        )
        rep = verify_v3_plan_detailed(plan, require_claims=True)
        self.assertTrue(any("does not lie on" in i for i in rep["issues"]))  # type: ignore[iterable]
        self.assertTrue(any("local_minimum" in i for i in rep["issues"]))  # type: ignore[iterable]

    def test_gradient_descent_trajectory_deterministic(self) -> None:
        traj = gradient_descent_trajectory("x**2", x0=2.0, lr=0.25, steps=4)
        self.assertAlmostEqual(traj[0][0], 2.0)
        self.assertAlmostEqual(traj[0][1], 4.0)
        self.assertAlmostEqual(traj[-1][0], 2.0 * (0.5**4), places=6)

    def test_clt_distribution_stats_and_convolution(self) -> None:
        supp, probs = discrete_convolution([1.0 / 6.0] * 6, n_trials=2, offset_per_trial=1)
        self.assertEqual(supp[0], 2)
        self.assertEqual(supp[-1], 12)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        plan = VideoPlanV3(
            topic="CLT Bad",
            title="CLT Bad",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="bad stats",
                    narration="Two dice sum.",
                    objects=[SceneObject(id="b1", kind="bar_chart", values=probs, bar_labels=[str(s) for s in supp])],
                    claims=[StructuredClaim(kind="distribution_stats", object_ids=["b1"], mu=9.0, sigma=1.0)],
                )
            ],
        )
        issues = validate_v3_plan(plan, require_claims=True)
        self.assertTrue(any("distribution_stats" in i for i in issues))

    def test_bayes_independent_evidence_and_normalization_rejected(self) -> None:
        res = bayes_posterior(0.2, 0.5, 0.5)
        self.assertAlmostEqual(res["posterior"], 0.2, places=6)
        plan = VideoPlanV3(
            topic="Bayes Bad",
            title="Bayes Bad",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="independent",
                    narration="Evidence changes belief.",
                    objects=[SceneObject(id="probs", kind="bar_chart", values=[0.2, 0.5], bar_labels=["A", "B"])],
                    claims=[
                        StructuredClaim(kind="probability_normalization", object_ids=["probs"]),
                        StructuredClaim(kind="conditional_probability", prior=0.2, likelihood=0.5, false_positive=0.5, expected_value=0.2),
                    ],
                )
            ],
        )
        issues = validate_v3_plan(plan, require_claims=True)
        self.assertTrue(any("values sum to" in i for i in issues))
        self.assertTrue(any("independent" in i for i in issues))

    def test_geometric_edge_roles_and_square_area_rejection(self) -> None:
        plan = VideoPlanV3(
            topic="Edges",
            title="Edges",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="edge role check",
                    narration="Build squares on legs and hypotenuse.",
                    constructions=[
                        ConstructionOp(op="right_triangle", output_ids=["tri"], params=ConstructionParams(a=3.0, b=4.0), label="triangle"),
                        ConstructionOp(op="square_on_segment", output_ids=["sq_hyp"], source_id="tri", params=ConstructionParams(edge_role="hypotenuse", outward=True), label="square"),
                    ],
                    claims=[StructuredClaim(kind="area_equality", object_ids=["sq_hyp"], expected_value=16.0)],
                )
            ],
        )
        issues = validate_v3_plan(plan, require_claims=True)
        self.assertTrue(any("area_equality" in i for i in issues))

    def test_incompatible_object_type_and_missing_construction_dep(self) -> None:
        p_incompat = VideoPlanV3(
            topic="Incompat",
            title="Incompat",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="bad type",
                    narration="Test.",
                    objects=[SceneObject(id="eq1", kind="equation", tex="a=b")],
                    claims=[StructuredClaim(kind="right_triangle", object_ids=["eq1"])],
                )
            ],
        )
        self.assertTrue(any("incompatible non-polygon" in i for i in validate_v3_plan(p_incompat)))

        p_missing = VideoPlanV3(
            topic="Missing dep",
            title="Missing dep",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="missing source",
                    narration="Test.",
                    constructions=[ConstructionOp(op="square_on_segment", output_ids=["sq"], source_id="nonexistent")],
                )
            ],
        )
        self.assertTrue(any("missing source_id" in i for i in validate_v3_plan(p_missing)))

    def test_transformed_geometry_evaluated_after_events(self) -> None:
        plan = VideoPlanV3(
            topic="Post-transform",
            title="Post-transform",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="move outside outer",
                    narration="Move square out of container.",
                    objects=[
                        SceneObject(id="outer", kind="polygon", vertices=[[0, 0], [4, 0], [4, 4], [0, 4]], label="square"),
                        SceneObject(id="inner", kind="polygon", vertices=[[0, 0], [2, 0], [2, 2], [0, 2]], label="square"),
                    ],
                    events=[
                        AnimationEvent(op="create", target_id="outer"),
                        AnimationEvent(op="create", target_id="inner"),
                        AnimationEvent(op="move", target_id="inner", shift=[10.0, 0.0]),
                    ],
                    claims=[StructuredClaim(kind="partition_of", object_ids=["inner"], container_id="outer")],
                )
            ],
        )
        issues = validate_v3_plan(plan)
        self.assertTrue(any("outside outer container" in i for i in issues))

    def test_stale_cache_key_invalidation(self) -> None:
        p1 = VideoPlanV3(
            topic="Cache",
            title="Cache",
            scenes=[SceneV3(scene_id="s1", purpose="p", narration="Hello.", objects=[SceneObject(id="eq", kind="equation", tex="x=1")])],
        )
        k1 = chapter_cache_key(p1, format_name="widescreen_hd", tts_engine="kokoro", voice="am_michael", rate="-4%", ui="card", look="cinematic")
        k2 = chapter_cache_key(p1, format_name="widescreen_hd", tts_engine="kokoro", voice="af_heart", rate="-4%", ui="card", look="cinematic")
        self.assertNotEqual(k1, k2)

    def test_missing_subtitles_and_failed_bundle_qa(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            mp4 = d / "out.mp4"
            mp4.write_bytes(b"fake")
            chk = verify_longform_bundle(mp4, d / "ch.json", d / "s.srt", d / "s.vtt", d / "t.txt", min_duration_s=300.0)
            self.assertFalse(chk.passed)
            self.assertIn("Missing or empty", chk.detail)

    def test_narration_token_map_preserves_bookmarks(self) -> None:
        res = normalize_for_speech_with_map("Watch <bookmark mark='sq'/>a^2 plus sin(x) equal 1/2.")
        self.assertEqual(res["bookmarks"], ["sq"])
        tmap = res["token_map"]
        a2_entry = next(e for e in tmap if "a^2" in str(e["display_text"]))  # type: ignore[union-attr]
        self.assertEqual(a2_entry["spoken_words"], ["a", "squared"])  # type: ignore[index]

    def test_pythagorean_dissection_post_transform_passes(self) -> None:
        p = Path("plans/v3_pythagorean_dissection_proof.json")
        if not p.exists():
            self.skipTest("Pythagorean plan missing")
        plan = VideoPlanV3.model_validate_json(p.read_text(encoding="utf-8"))
        self.assertEqual(validate_v3_plan(plan, require_claims=True), [])


if __name__ == "__main__":
    unittest.main()
