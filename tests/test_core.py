"""Baseline unit tests for Math Shorts Studio V2."""

from pathlib import Path
import unittest

from studio.ai.expr import ExprError, compile_expr
from studio.ai.schema import ShortPlan
from studio.ai.validate import validate_plan
from studio.audio.master import cinematic_video_filter
from studio.audio.tts import align_words, parse_script


class TestCore(unittest.TestCase):
    def test_expr_safe_eval(self):
        fn = compile_expr("4/pi*sum(k, 1, 3, 2, sin(k*x)/k)", ("x",))
        val = float(fn(x=1.5707963267948966))
        self.assertTrue(0.8 < val < 1.0)

    def test_expr_rejects_unsafe(self):
        with self.assertRaises(ExprError):
            compile_expr("__import__('os').system('echo hi')", ("x",))

    def test_parse_script_and_align(self):
        clean, tokens, marks = parse_script("Add <bookmark mark='flat'/>*flatten* now.")
        self.assertEqual(clean, "Add flatten now.")
        self.assertEqual(marks, {"flat": 1})
        self.assertTrue(tokens[1].emphasis)
        times = align_words(clean, tokens, [["Add", 0.0, 0.3], ["flatten", 0.3, 0.8]], 1.2)
        self.assertEqual(len(times), 3)
        self.assertTrue(times[0][0] <= times[1][0] <= times[2][0])

    def test_cinematic_video_filter(self):
        self.assertEqual(cinematic_video_filter("none"), "")
        flt = cinematic_video_filter("cinematic")
        self.assertIn("eq=", flt)
        self.assertIn("format=yuv420p", flt)

    def test_validate_saved_plan(self):
        plan_path = Path("plans/why_0_999_equals_1.json")
        plan = ShortPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        problems = validate_plan(plan)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
