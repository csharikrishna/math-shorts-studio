"""Behavioral tests for V4.2 semantic verification, layout/caption fixes, and K-12 Educational Studio."""

from __future__ import annotations

import json
import unittest
import urllib.request

from studio.ai.linked_models import LinkedMathModelSpec
from studio.ai.v3_schema import AnimationEvent, SceneObject, SceneV3, VideoPlanV3
from studio.ai.visual_intelligence import humanize_visible_label, infer_beat_visual_decision, verify_beat_narration_semantics
from studio.education import (
    CBSE_MATH_INDEX,
    LessonRequest,
    array_multiplication_facts,
    build_lesson_spec,
    calculus_secant_to_tangent,
    exact_fraction_add,
    exact_fraction_equivalent,
    generate_practice_question,
    generate_worked_example,
    get_grade_profile,
    integer_number_line_op,
    place_value_decompose,
    ratio_and_percentage_steps,
    right_triangle_trig,
    solve_linear_equation_steps,
    solve_quadratic_steps,
)
from studio.education.planner import generate_k12_lesson_plan
from studio.education.studio_app import create_studio_server
from studio.design.spec import default_design_spec
from studio.renderers.hyperframes.compiler import compile_plan_to_html
from studio.repair import repair_scene_deterministically


class TestV42AndK12(unittest.TestCase):
    def test_semantic_verification_catches_critical_damping_and_upright_pendulum(self) -> None:
        m = LinkedMathModelSpec(
            model_id="osc",
            kind="dynamical_simulation",
            state_vars=["x", "v"],
            rhs_exprs=["v", "-(k/m)*x - (c/m)*v"],
            params={"m": 1.0, "k": 4.0, "c": 0.2},  # underdamped! c^2=0.04 < 4*1*4=16
            initial_state=[1.0, 0.0],
        )
        p = VideoPlanV3(
            topic="Oscillator and Pendulum",
            math_models=[m],
            scenes=[
                SceneV3(
                    scene_id="s_crit",
                    purpose="Critical damping",
                    narration="With critical damping, the mass returns without oscillating.",
                    objects=[
                        SceneObject(id="ax", kind="axes", x_range=[0.0, 5.0], y_range=[-1.2, 1.2]),
                        SceneObject(id="curve", kind="data_series", axes_id="ax", model_id="osc", model_output="time_series_x", label="phase_orbit"),
                    ],
                ),
                SceneV3(
                    scene_id="s_pend",
                    purpose="Upright equilibrium",
                    narration="At the unstable upright equilibrium, the rod points straight up.",
                    objects=[
                        SceneObject(id="bob", kind="mechanical_body", connector_kind="rod", anchor_coords=[0.0, 1.0], coords=[0.0, -0.8], label="Mystery Annotation"),
                    ],
                ),
            ],
        )
        issues = verify_beat_narration_semantics(p)
        joined = "\n".join(issues)
        self.assertIn("underdamped", joined)
        self.assertIn("hanging below pivot", joined)
        self.assertIn("phase_orbit", joined)
        self.assertIn("Mystery Annotation", joined)

        repaired, _ = repair_scene_deterministically(p)
        labels = [o.label for sc in repaired.scenes for o in sc.objects]
        self.assertIn("Phase Portrait", labels)
        self.assertIn("Key Observation", labels)
        self.assertEqual(humanize_visible_label("time_series_x"), "x(t)")

    def test_potential_landscape_never_picks_bar_chart_and_semantic_qa_flags_it(self) -> None:
        class _DummyBeat:
            scene_id = "s_pot"
            educational_purpose = "Visualize the potential energy landscape V(theta) = 1 - cos(theta)"
            visual_intent = "Show potential well"
            spoken_text = "Here is the potential energy landscape."
            transition_from_prev = ""
            demo_contract = None
            latex_refs = []

        dec = infer_beat_visual_decision(_DummyBeat())
        self.assertNotIn("bar_chart", dec.visual_components)
        self.assertIn("function_plot", dec.visual_components)

        bad_plan = VideoPlanV3(
            topic="Pendulum",
            scenes=[
                SceneV3(
                    scene_id="s_pot",
                    purpose="Potential well",
                    narration="Look at the potential energy landscape.",
                    objects=[SceneObject(id="b1", kind="bar_chart", values=[1.0, 2.0], bar_labels=["K", "U"])],
                )
            ],
        )
        self.assertTrue(any("bar_chart" in msg for msg in verify_beat_narration_semantics(bad_plan)))

    def test_grade_profiles_and_curriculum_index(self) -> None:
        self.assertEqual(get_grade_profile(1).band, "1-2")
        self.assertEqual(get_grade_profile(5).band, "3-5")
        self.assertEqual(get_grade_profile(8).band, "6-8")
        self.assertEqual(get_grade_profile(10).band, "9-10")
        self.assertEqual(get_grade_profile(12).band, "11-12")
        self.assertLess(get_grade_profile(1).max_sentence_words, get_grade_profile(12).max_sentence_words)
        self.assertGreaterEqual(len(CBSE_MATH_INDEX), 12)

        spec = build_lesson_spec(LessonRequest(grade=5, topic="Adding fractions with different denominators"))
        self.assertEqual(spec.curriculum_ref.verification_status, "verified_metadata")

        unseen = build_lesson_spec(LessonRequest(grade=7, board="curriculum_neutral", topic="Modular clock arithmetic"))
        self.assertEqual(unseen.curriculum_ref.verification_status, "curriculum_neutral")

    def test_deterministic_school_math_engine(self) -> None:
        frac = exact_fraction_add(1, 2, 1, 3)
        self.assertEqual(frac["lcd"], 6)
        self.assertEqual(frac["sum_simplified"], "5/6")
        self.assertEqual(exact_fraction_equivalent(2, 3, 4)["equivalent"], "8/12")

        lin = solve_linear_equation_steps(3, 4, 1, 10)
        self.assertEqual(lin["solution"], "3")
        quad = solve_quadratic_steps(1, -5, 6)
        self.assertEqual(set(quad["roots"]), {"2", "3"})
        self.assertEqual(array_multiplication_facts(3, 4)["product"], 12)
        self.assertEqual(place_value_decompose(347)["expanded"], "300 + 40 + 7")
        self.assertEqual(integer_number_line_op(-3, 5)["result"], 2)
        self.assertEqual(ratio_and_percentage_steps(3, 4)["percent"], "75%")
        self.assertAlmostEqual(right_triangle_trig(3.0, 4.0)["hypotenuse"], 5.0)
        calc = calculus_secant_to_tangent("x^2", 1.0)
        self.assertAlmostEqual(calc["exact_slope"], 2.0)

        we = generate_worked_example("adding_fractions")
        self.assertEqual(we.final_answer, "5/6")
        pq = generate_practice_question("adding_fractions", grade=5)
        self.assertEqual(len(pq.choices), 4)
        self.assertEqual(pq.choices[pq.correct_index], pq.correct_answer)

    def test_k12_planner_and_hyperframes_compilation(self) -> None:
        for grade, topic in [
            (1, "Addition using objects"),
            (3, "Multiplication using arrays"),
            (5, "Adding fractions with different denominators"),
            (8, "Solving linear equations"),
            (12, "Derivatives and tangent lines"),
        ]:
            req = LessonRequest(grade=grade, topic=topic, theme="light", format_name="widescreen_hd")
            spec, plan, worked, question = generate_k12_lesson_plan(req, provider=None)
            self.assertEqual(spec.request.grade, grade)
            self.assertGreaterEqual(len(plan.scenes), 3)
            self.assertTrue(worked.verified)
            self.assertTrue(question.verified)
            html_str, dur_s = compile_plan_to_html(plan, default_design_spec("widescreen_hd", "light"))
            self.assertIn("data-hf-elem", html_str)
            self.assertGreater(dur_s, 1.0)

    def test_local_studio_server_roundtrip(self) -> None:
        srv, url = create_studio_server(port=0, use_llm=False)
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                self.assertEqual(r.status, 200)
                self.assertIn(b"K-12 Lesson Creator", r.read())
            req = urllib.request.Request(
                url + "api/generate",
                data=json.dumps({"grade": 5, "topic": "Adding fractions with different denominators", "theme": "light"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
                self.assertEqual(data["worked"]["final_answer"], "5/6")
                self.assertGreaterEqual(len(data["plan"]["scenes"]), 3)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
