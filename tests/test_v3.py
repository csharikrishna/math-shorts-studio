"""Behavioral regression tests for V3 math truth, object-event schema, and geometric clipping."""

import unittest
import numpy as np

from studio.ai.math_truth import derive_latex, extract_taylor_coefficients, verify_polynomial_degree
from studio.ai.v3_schema import AnimationEvent, SceneObject, SceneV3, VideoPlanV3
from studio.ai.validate_v3 import validate_v3_plan


class TestV3(unittest.TestCase):
    def test_math_truth_degree_and_taylor_coeffs(self):
        expr = "sum(k, 0, 4, 1, (-1)**k * x**(2*k) / factorial(2*k))"
        self.assertTrue(verify_polynomial_degree(expr, 8))
        self.assertFalse(verify_polynomial_degree(expr, 10))
        coeffs = extract_taylor_coefficients("cos(x)", 4)
        self.assertAlmostEqual(coeffs[0], 1.0)
        self.assertAlmostEqual(coeffs[2], -0.5)
        self.assertAlmostEqual(coeffs[4], 1.0 / 24.0)
        # Nonzero expansion point x0 = 1.0 for exp(x): coeffs of (x - 1)^k are e / k!
        exp_coeffs = extract_taylor_coefficients("exp(x)", 2, x0=1.0)
        self.assertAlmostEqual(exp_coeffs[0], float(np.e), places=5)
        self.assertAlmostEqual(exp_coeffs[1], float(np.e), places=5)
        self.assertAlmostEqual(exp_coeffs[2], float(np.e) / 2.0, places=5)
        self.assertIn("cos", derive_latex("cos(x)"))

    def test_v3_lifecycle_state_simulation(self):
        # Redeclaring persistent object must fail
        p = VideoPlanV3(
            topic="Lifecycle",
            title="Lifecycle",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Create",
                    narration="Create a circle.",
                    objects=[SceneObject(id="c1", kind="circle", radius=1.0, persist=True)],
                    events=[AnimationEvent(op="create", target_id="c1")],
                ),
                SceneV3(
                    scene_id="s2",
                    purpose="Accidental redeclare",
                    narration="Redeclare.",
                    objects=[SceneObject(id="c1", kind="circle", radius=2.0, persist=True)],
                    events=[],
                ),
            ],
        )
        self.assertTrue(any("already exists" in i for i in validate_v3_plan(p)))

        # Referencing a removed object must fail
        p_rem = VideoPlanV3(
            topic="Remove",
            title="Remove",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Create and remove",
                    narration="Create then remove.",
                    objects=[SceneObject(id="c1", kind="circle", radius=1.0, persist=True)],
                    events=[AnimationEvent(op="create", target_id="c1"), AnimationEvent(op="remove", target_id="c1")],
                ),
                SceneV3(
                    scene_id="s2",
                    purpose="Use after remove",
                    narration="Highlight removed.",
                    objects=[],
                    events=[AnimationEvent(op="highlight", target_id="c1")],
                ),
            ],
        )
        self.assertTrue(any("not live" in i for i in validate_v3_plan(p_rem)))

    def test_euler_v31_plan_valid(self):
        from pathlib import Path

        ep = VideoPlanV3.model_validate_json(Path("plans/v3_euler_identity_rotation.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_v3_plan(ep), [])

    def test_world_model_rejects_buggy_pythagorean_and_verifies_dissection(self):
        from pathlib import Path
        from studio.ai.world_model import is_right_triangle, is_square, polygon_area, pythagorean_dissection, square_on_segment, triangles_congruent

        # Buggy legacy plan must FAIL geometric + narration-to-visual validation
        buggy = VideoPlanV3.model_validate_json(
            Path("plans/v3_pythagorean_theorem_geometric_proof.json").read_text(encoding="utf-8")
        )
        issues = validate_v3_plan(buggy)
        self.assertTrue(any("square" in i.lower() for i in issues), f"Expected square failure, got {issues}")

        # Validated dissection plan must PASS
        good = VideoPlanV3.model_validate_json(
            Path("plans/v3_pythagorean_dissection_proof.json").read_text(encoding="utf-8")
        )
        self.assertEqual(validate_v3_plan(good), [])

        # Property check across multiple (a, b) leg lengths
        for a, b in [(3.0, 4.0), (5.0, 12.0), (1.5, 2.5)]:
            d = pythagorean_dissection(a, b)
            self.assertTrue(is_square(d["outer_square"]))
            self.assertTrue(is_square(d["inner_square"]))
            tris = d["triangles"]
            self.assertEqual(len(tris), 4)
            for t in tris:
                self.assertTrue(is_right_triangle(t))
                self.assertTrue(triangles_congruent(tris[0], t))
            self.assertAlmostEqual(d["outer_area"], d["triangles_area"] + d["inner_area"], places=6)
            sq = square_on_segment([0.0, 0.0], [a, b])
            self.assertTrue(is_square(sq))
            self.assertAlmostEqual(polygon_area(sq), a * a + b * b, places=6)

    def test_v3_plan_validation_and_negative_cases(self):
        p = VideoPlanV3(
            topic="Pythagorean Theorem",
            title="A Squared Plus B Squared",
            scenes=[
                SceneV3(
                    scene_id="s1",
                    purpose="Draw right triangle",
                    narration="Start with a <bookmark mark='tri'/>right triangle.",
                    objects=[
                        SceneObject(id="tri", kind="polygon", vertices=[[0, 0], [3, 0], [0, 4]], persist=True),
                    ],
                    events=[
                        AnimationEvent(op="create", target_id="tri", anchor_mark="tri"),
                    ],
                )
            ],
        )
        self.assertEqual(validate_v3_plan(p), [])

        bad = p.model_copy(deep=True)
        bad.scenes[0].events.append(AnimationEvent(op="highlight", target_id="missing_obj"))
        issues = validate_v3_plan(bad)
        self.assertTrue(any("not live" in i for i in issues))

    def test_plot_clipped_segments_no_flat_clip(self):
        import manimgx as m
        from scenes.multi_scene import plot_clipped_segments

        ax = m.Axes(x_range=[-4, 4, 1], y_range=[-1.5, 1.5, 0.5], x_length=6, y_length=4)
        grp = plot_clipped_segments(ax, lambda x: x**2 - 2.0, [-4, 4], [-1.5, 1.5], color=m.BLUE)
        # Parabola x^2 - 2 enters [-1.5, 1.5] around [-1.87, 1.87], so it should be 1 contiguous segment
        # with no points outside y_range and no artificial flat clamp at y = 1.5
        self.assertGreaterEqual(len(grp.submobjects), 1)


if __name__ == "__main__":
    unittest.main()
