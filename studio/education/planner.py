"""AI-teacher instructional planner: bridges LessonSpec + deterministic school math -> verified VideoPlanV3."""

from __future__ import annotations

from typing import Any

from studio.ai.v3_schema import (
    AnimationEvent,
    SceneObject,
    SceneV3,
    StructuredClaim,
    VideoPlanV3,
)
from studio.ai.visual_intelligence import apply_visual_intelligence_and_fit
from studio.education.math_engine import (
    PracticeQuestion,
    WorkedExample,
    generate_practice_question,
    generate_worked_example,
)
from studio.education.profiles import LessonRequest, LessonSpec, build_lesson_spec
from studio.providers.base import LLMProvider
from studio.repair import repair_scene_deterministically


def _infer_math_domain_kind(topic: str, grade: int) -> str:
    t = topic.lower()
    if any(k in t for k in ("fraction", "denominator", "half", "quarter", "equivalent")):
        return "adding_fractions"
    if any(k in t for k in ("equation", "linear", "algebra", "solve for x", "balance")):
        return "linear_equation"
    if any(k in t for k in ("multiply", "multiplication", "array", "times")):
        return "multiplication_array"
    if any(k in t for k in ("integer", "negative", "number line", "addition")):
        return "integers"
    if any(k in t for k in ("derivative", "tangent", "secant", "limit", "calculus", "differentiation")):
        return "calculus"
    if grade <= 2:
        return "integers"
    if grade <= 5:
        return "adding_fractions"
    if grade <= 8:
        return "linear_equation"
    return "calculus"


def build_deterministic_k12_plan(
    spec: LessonSpec,
    worked: WorkedExample,
    question: PracticeQuestion,
) -> VideoPlanV3:
    """Build a clean, grade-adapted, reusable-component VideoPlanV3 grounded in verified school math."""
    req = spec.request
    prof = spec.grade_profile
    domain = _infer_math_domain_kind(req.topic, req.grade)
    aspect = "9:16" if req.format_name == "shorts_hd" else "16:9"
    scenes: list[SceneV3] = []

    # Scene 1: Concrete Hook & Learning Objective
    if domain == "adding_fractions":
        s1_objs = [
            SceneObject(id="hook_frac_a", kind="fraction_bar", persist=False, numerator=1, denominator=2, panel_slot="left", color_role="primary", label="1/2"),
            SceneObject(id="hook_frac_b", kind="fraction_bar", persist=False, numerator=1, denominator=3, panel_slot="right", color_role="secondary", label="1/3"),
        ]
        s1_nar = (
            f"Welcome! Today in Class {req.grade} Mathematics, we are exploring {req.topic}. "
            f"Look at these two bars: one half and one third. <bookmark mark='show_bars'/> "
            f"Their pieces are different sizes, so we cannot just count them together yet."
        )
        s1_evs = [
            AnimationEvent(t=0.2, op="create", target_id="hook_frac_a"),
            AnimationEvent(t=0.6, op="create", target_id="hook_frac_b", bookmark="show_bars"),
            AnimationEvent(t=1.2, op="scale", target_id="hook_frac_a", scale_factor=1.08),
        ]
        s1_comp = "stacked_comparison" if aspect == "9:16" else "side_by_side_comparison"
    elif domain == "linear_equation":
        s1_objs = [
            SceneObject(id="bal_scale", kind="equation_balance", persist=False, lhs_text="3x + 4", rhs_text="x + 10", color_role="primary", label="Balanced Equation"),
        ]
        s1_nar = (
            f"Welcome to Class {req.grade} Mathematics. Today we are solving {req.topic}. "
            f"Think of an equation like a balanced scale: <bookmark mark='show_scale'/> "
            f"whatever we do to one side, we must also do to the other side."
        )
        s1_evs = [
            AnimationEvent(t=0.2, op="create", target_id="bal_scale"),
            AnimationEvent(t=1.0, op="scale", target_id="bal_scale", bookmark="show_scale", scale_factor=1.08),
        ]
        s1_comp = "centered_focus"
    elif domain == "multiplication_array":
        s1_objs = [
            SceneObject(id="arr_main", kind="array_grid", persist=False, rows=3, cols=4, count=12, color_role="primary", label="3 rows of 4 = 12"),
        ]
        s1_nar = (
            f"Hello! In Class {req.grade} Mathematics, let us understand {req.topic}. "
            f"Here we have 3 equal rows with 4 tiles in each row. <bookmark mark='show_array'/> "
            f"Instead of counting one by one, an array lets us multiply 3 times 4 to get 12."
        )
        s1_evs = [
            AnimationEvent(t=0.2, op="create", target_id="arr_main"),
            AnimationEvent(t=1.0, op="scale", target_id="arr_main", bookmark="show_array", scale_factor=1.08),
        ]
        s1_comp = "centered_focus"
    elif domain == "integers":
        s1_objs = [
            SceneObject(id="grp_start", kind="counting_group", persist=False, count=5, group_size=5, panel_slot="left", color_role="primary", label="5 units"),
            SceneObject(id="nl_main", kind="number_line", persist=False, x_range=[-5.0, 5.0], panel_slot="right", color_role="secondary", label="Number Line"),
        ]
        s1_nar = (
            f"Welcome to Class {req.grade} Mathematics! Let us explore {req.topic}. "
            f"We can picture numbers as steps along a number line. <bookmark mark='show_nl'/> "
            f"Moving right adds positive steps, and moving left takes steps in the opposite direction."
        )
        s1_evs = [
            AnimationEvent(t=0.2, op="create", target_id="grp_start"),
            AnimationEvent(t=0.7, op="create", target_id="nl_main", bookmark="show_nl"),
            AnimationEvent(t=1.2, op="scale", target_id="grp_start", scale_factor=1.08),
        ]
        s1_comp = "stacked_comparison" if aspect == "9:16" else "side_by_side_comparison"
    else:
        s1_objs = [
            SceneObject(id="calc_axes", kind="axes", persist=False, x_range=[-1.0, 3.5], y_range=[-1.0, 10.0], x_label="x", y_label="f(x)", color_role="secondary"),
            SceneObject(id="calc_curve", kind="function_plot", persist=False, expr="x**2", axes_id="calc_axes", color_role="primary", label="f(x) = x^2"),
            SceneObject(id="calc_tan", kind="line", persist=False, coords=[0.0, -1.0], end_coords=[2.5, 4.0], axes_id="calc_axes", color_role="highlight", label="Tangent at x = 1"),
            SceneObject(id="calc_pt", kind="point", persist=False, coords=[1.0, 1.0], axes_id="calc_axes", color_role="highlight", label="(1, 1)"),
        ]
        s1_nar = (
            f"In Class {req.grade} Mathematics, let us examine {req.topic}. "
            f"On the curve y equals x squared, <bookmark mark='show_tan'/> "
            f"the tangent line at x equals 1 captures the exact instantaneous slope, which is 2."
        )
        s1_evs = [
            AnimationEvent(t=0.1, op="create", target_id="calc_axes"),
            AnimationEvent(t=0.4, op="draw", target_id="calc_curve"),
            AnimationEvent(t=1.0, op="draw", target_id="calc_tan", bookmark="show_tan"),
            AnimationEvent(t=1.2, op="scale", target_id="calc_tan", scale_factor=1.10),
        ]
        s1_comp = "centered_focus"

    claim_map = {
        "adding_fractions": ("function_equivalence", "1/2 + 1/3", "5/6"),
        "linear_equation": ("function_equivalence", "3*x + 4 - (x + 10)", "2*(x - 3)"),
        "multiplication_array": ("function_equivalence", "3*4", "4 + 4 + 4"),
        "integers": ("function_equivalence", "-3 + 5", "2"),
        "calculus": ("derivative_equality", "x**2", "2*x"),
    }
    ckind, cexpr, cexp = claim_map.get(domain, ("function_equivalence", "1 + 1", "2"))

    scenes.append(
        SceneV3(
            scene_id="s1_intuition",
            purpose=f"Concrete intuition for {req.topic}",
            narration=s1_nar,
            composition=s1_comp,  # type: ignore[arg-type]
            objects=s1_objs,
            events=s1_evs,
            claims=[
                StructuredClaim(
                    kind=ckind,  # type: ignore[arg-type]
                    statement=f"Worked solution for '{worked.problem}' is {worked.final_answer}",
                    expr=cexpr,
                    expected_expr=cexp,
                )
            ],
        )
    )

    # Optional prerequisite & misconception bridge scenes for standard/detailed lessons (>= 3 min)
    if spec.production_mode in ("standard_lesson", "detailed_lesson"):
        prereq_txt = ", ".join(spec.curriculum_ref.prerequisites) or f"Foundations for {req.topic}"
        scenes.append(
            SceneV3(
                scene_id="s1b_prerequisites",
                purpose="Prerequisite check and misconception warning",
                narration=(
                    f"Before we solve a full problem, let us check what we already know: {prereq_txt}. "
                    f"<bookmark mark='prereq_focus'/> Keep in mind a common mistake to avoid: {worked.common_mistake}"
                ),
                composition="centered_focus",
                objects=[
                    SceneObject(
                        id="prereq_card",
                        kind="concept_card",
                        persist=False,
                        label="Prerequisites & Watch-Out",
                        steps_list=[f"Prerequisite: {prereq_txt}", f"Common Mistake: {worked.common_mistake}"],
                        color_role="secondary",
                    )
                ],
                events=[
                    AnimationEvent(t=0.1, op="reveal", target_id="prereq_card"),
                    AnimationEvent(t=2.0, op="scale", target_id="prereq_card", bookmark="prereq_focus", scale_factor=1.10),
                ],
            )
        )

    # Scene 2: Worked Example (Deterministic verified steps)
    if domain == "adding_fractions":
        s2_objs = [
            SceneObject(id="eq_a", kind="fraction_bar", persist=False, numerator=1, denominator=2, subdivisions=3, panel_slot="top", color_role="primary", label="1/2 = 3/6"),
            SceneObject(id="we_card", kind="concept_card", persist=False, label=f"Worked Example: {worked.problem}", steps_list=worked.steps + [f"Answer: {worked.final_answer}"], panel_slot="bottom", color_role="highlight"),
        ]
        s2_evs = [
            AnimationEvent(t=0.1, op="create", target_id="eq_a"),
            AnimationEvent(t=0.9, op="reveal", target_id="we_card", bookmark="steps"),
            AnimationEvent(t=1.4, op="scale", target_id="eq_a", scale_factor=1.08),
        ]
    else:
        s2_objs = [
            SceneObject(id="we_card", kind="concept_card", persist=False, label=f"Worked Example: {worked.problem}", steps_list=worked.steps + [f"Answer: {worked.final_answer}"], color_role="primary"),
        ]
        s2_evs = [
            AnimationEvent(t=0.1, op="reveal", target_id="we_card"),
            AnimationEvent(t=1.0, op="scale", target_id="we_card", bookmark="steps", scale_factor=1.05),
        ]
    s2_nar = (
        f"Let us solve a worked example step by step: {worked.problem} "
        f"<bookmark mark='steps'/> {' '.join(worked.steps)} "
        f"So the exact answer is {worked.final_answer}."
    )
    scenes.append(
        SceneV3(
            scene_id="s2_worked_example",
            purpose="Step-by-step verified worked example",
            narration=s2_nar,
            composition="centered_focus",
            objects=s2_objs,
            events=s2_evs,
        )
    )

    # Scene 3: Check Your Understanding & Recap (think-then-reveal)
    s3_objs = [
        SceneObject(
            id="quiz_card",
            kind="concept_card",
            persist=False,
            panel_slot="top",
            label="Quick Check",
            steps_list=[
                f"Question: {question.question}",
                f"Choices: {', '.join(question.choices)}",
            ],
            color_role="secondary",
        ),
        SceneObject(
            id="ans_card",
            kind="concept_card",
            persist=False,
            panel_slot="bottom",
            label="Verified Answer & Recap",
            steps_list=[
                f"Verified Answer: {question.correct_answer}",
                f"Avoid mistake: {worked.common_mistake}",
            ],
            color_role="primary",
        ),
    ]
    s3_nar = (
        f"Now try this quick check yourself: {question.question} "
        f"<bookmark mark='reveal_ans'/> {question.explanation} "
        f"Keep practicing, and remember: {worked.common_mistake}"
    )
    scenes.append(
        SceneV3(
            scene_id="s3_practice_recap",
            purpose="Practice question, misconception check, and recap",
            narration=s3_nar,
            composition="stacked_comparison" if aspect == "9:16" else "centered_focus",
            objects=s3_objs,
            events=[
                AnimationEvent(t=0.1, op="reveal", target_id="quiz_card"),
                AnimationEvent(t=0.8, op="reveal", target_id="ans_card", bookmark="reveal_ans"),
            ],
        )
    )

    plan = VideoPlanV3(
        topic=req.topic,
        title=f"Class {req.grade}: {req.topic}",
        format=req.format_name,  # type: ignore[arg-type]
        theme="editorial_light" if "light" in req.theme else "cinematic_dark",
        tone=prof.tone_guidance[:80],
        claims=[],
        scenes=scenes,
    )
    plan, _ = repair_scene_deterministically(plan)
    plan = apply_visual_intelligence_and_fit(plan, aspect_label=aspect)
    if aspect == "9:16":
        for sc in plan.scenes:
            if sc.composition == "side_by_side_comparison":
                sc.composition = "stacked_comparison"
    return plan


def generate_k12_lesson_plan(
    req: LessonRequest,
    provider: LLMProvider | None = None,
) -> tuple[LessonSpec, VideoPlanV3, WorkedExample, PracticeQuestion]:
    """Generate a grade-aware, curriculum-aligned K-12 lesson plan + verified example + practice question."""
    spec = build_lesson_spec(req)
    domain = _infer_math_domain_kind(req.topic, req.grade)
    worked = generate_worked_example(domain)
    question = generate_practice_question(domain, grade=req.grade)

    if provider is None:
        return spec, build_deterministic_k12_plan(spec, worked, question), worked, question

    prof = spec.grade_profile
    prompt = (
        f"You are an expert Class {req.grade} ({prof.band} grade band) {req.subject} teacher.\n"
        f"Create a clear, engaging VideoPlanV3 lesson on: {req.topic!r}\n"
        f"Curriculum: {spec.curriculum_ref.board} ({spec.curriculum_ref.chapter_ref or spec.curriculum_ref.source})\n"
        f"Learning objectives: {spec.curriculum_ref.learning_objectives}\n"
        f"Prerequisites: {spec.curriculum_ref.prerequisites}\n"
        f"Common misconceptions to address: {spec.curriculum_ref.common_misconceptions}\n"
        f"Grade constraints:\n"
        f"- Max words per sentence: {prof.max_sentence_words}\n"
        f"- Notation density: {prof.notation_density}\n"
        f"- Tone: {prof.tone_guidance}\n"
        f"- Preferred visual components: {prof.preferred_visuals} (plus fraction_bar, fraction_circle, array_grid, counting_group, equation_balance, concept_card, number_line, axes, function_plot)\n"
        f"- Avoided visuals: {prof.avoided_visuals}\n"
        f"Verified Worked Example (MUST match exact answer {worked.final_answer}):\n"
        f"  Problem: {worked.problem}\n  Steps: {worked.steps}\n  Answer: {worked.final_answer}\n"
        f"Verified Practice Question:\n"
        f"  Question: {question.question}\n  Answer: {question.correct_answer}\n"
        f"Return a valid VideoPlanV3 with 3-5 scenes, bookmarks <bookmark mark='...'/>, and theme={req.theme!r}, format={req.format_name!r}."
    )
    try:
        plan, _ = provider.generate_structured(prompt, VideoPlanV3, temperature=0.35)
        plan, _ = repair_scene_deterministically(plan)
        aspect = "9:16" if req.format_name == "shorts_hd" else "16:9"
        plan = apply_visual_intelligence_and_fit(plan, aspect_label=aspect)
        return spec, plan, worked, question
    except Exception:
        return spec, build_deterministic_k12_plan(spec, worked, question), worked, question
