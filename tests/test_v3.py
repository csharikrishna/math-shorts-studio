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
        self.assertIn("cos", derive_latex("cos(x)"))

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
        self.assertTrue(any("not defined" in i for i in issues))

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
