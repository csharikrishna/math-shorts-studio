"""Grade-aware teaching profiles (Classes 1-12), curriculum metadata, and lesson specifications."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

GradeBand = Literal["1-2", "3-5", "6-8", "9-10", "11-12"]
TeachingStrategy = Literal[
    "concrete_to_abstract",
    "story_based",
    "question_and_answer",
    "visual_proof",
    "problem_and_solution",
    "comparison",
    "misconception_correction",
    "mathematical_discovery",
    "scientific_demonstration",
    "examination_revision",
]


class GradeProfile(BaseModel):
    """Pedagogical, linguistic, pacing, and visual rules for a K-12 grade band."""

    band: GradeBand
    grade_range: tuple[int, int]
    age_range: tuple[int, int]
    max_sentence_words: int
    tts_rate_wpm: int
    pause_between_beats_s: float
    notation_density: Literal["minimal", "gradual", "moderate", "formal", "rigorous"]
    preferred_visuals: list[str]
    avoided_visuals: list[str]
    default_strategy: TeachingStrategy
    tone_guidance: str
    caption_max_chars: int


GRADE_PROFILES: dict[GradeBand, GradeProfile] = {
    "1-2": GradeProfile(
        band="1-2",
        grade_range=(1, 2),
        age_range=(6, 8),
        max_sentence_words=11,
        tts_rate_wpm=125,
        pause_between_beats_s=0.65,
        notation_density="minimal",
        preferred_visuals=["counting_objects", "ten_frame", "place_value_blocks", "number_line", "basic_shapes"],
        avoided_visuals=["vector_field", "matrix", "differential_equation", "complex_plane"],
        default_strategy="concrete_to_abstract",
        tone_guidance="Warm, friendly, curious, non-patronizing; short everyday sentences; large clear objects.",
        caption_max_chars=34,
    ),
    "3-5": GradeProfile(
        band="3-5",
        grade_range=(3, 5),
        age_range=(8, 11),
        max_sentence_words=14,
        tts_rate_wpm=138,
        pause_between_beats_s=0.50,
        notation_density="gradual",
        preferred_visuals=["array_grid", "fraction_bar", "fraction_circle", "number_line", "decimal_grid", "area_diagram"],
        avoided_visuals=["vector_field", "jacobian", "phase_portrait"],
        default_strategy="concrete_to_abstract",
        tone_guidance="Encouraging teacher voice; build from physical picture to rule; step-by-step worked examples.",
        caption_max_chars=40,
    ),
    "6-8": GradeProfile(
        band="6-8",
        grade_range=(6, 8),
        age_range=(11, 14),
        max_sentence_words=17,
        tts_rate_wpm=148,
        pause_between_beats_s=0.40,
        notation_density="moderate",
        preferred_visuals=["number_line", "equation_balance", "algebra_tiles", "ratio_bar", "axes", "polygon", "bar_chart"],
        avoided_visuals=["riemann_tensor", "separatrix"],
        default_strategy="problem_and_solution",
        tone_guidance="Clear, logical, connecting everyday intuition to symbols and algebraic reasoning.",
        caption_max_chars=44,
    ),
    "9-10": GradeProfile(
        band="9-10",
        grade_range=(9, 10),
        age_range=(14, 16),
        max_sentence_words=20,
        tts_rate_wpm=155,
        pause_between_beats_s=0.35,
        notation_density="formal",
        preferred_visuals=["axes", "function_plot", "polygon", "equation", "unit_circle", "table", "worked_example"],
        avoided_visuals=["counting_objects"],
        default_strategy="visual_proof",
        tone_guidance="Structured, insightful, showing both why a formula works and how to apply it cleanly.",
        caption_max_chars=46,
    ),
    "11-12": GradeProfile(
        band="11-12",
        grade_range=(11, 12),
        age_range=(16, 18),
        max_sentence_words=22,
        tts_rate_wpm=160,
        pause_between_beats_s=0.30,
        notation_density="rigorous",
        preferred_visuals=["axes", "function_plot", "parametric_curve", "vector", "matrix", "vector_field", "equation"],
        avoided_visuals=["counting_objects", "ten_frame"],
        default_strategy="mathematical_discovery",
        tone_guidance="Precise, motivating, connecting geometric intuition with formal calculus, vectors, and proofs.",
        caption_max_chars=48,
    ),
}


def grade_to_band(grade: int) -> GradeBand:
    g = max(1, min(12, int(grade)))
    if g <= 2:
        return "1-2"
    if g <= 5:
        return "3-5"
    if g <= 8:
        return "6-8"
    if g <= 10:
        return "9-10"
    return "11-12"


def get_grade_profile(grade: int) -> GradeProfile:
    return GRADE_PROFILES[grade_to_band(grade)]


class CurriculumReference(BaseModel):
    board: str = "curriculum_neutral"
    syllabus_version: str = "NCERT-2023-24"
    grade: int = Field(ge=1, le=12)
    subject: str = "Mathematics"
    chapter: str = ""
    topic: str = ""
    chapter_ref: str = ""
    source: str = "Curriculum-neutral pedagogical specification"
    verification_status: Literal["verified_metadata", "curriculum_neutral", "user_supplied"] = "curriculum_neutral"
    prerequisites: list[str] = Field(default_factory=list)
    learning_objectives: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)


# Verified CBSE/NCERT-aligned reference entries (metadata only; original explanations generated at runtime)
CBSE_MATH_INDEX: dict[tuple[int, str], CurriculumReference] = {
    (1, "addition"): CurriculumReference(
        board="CBSE/NCERT",
        grade=1,
        chapter="Addition",
        topic="Addition using objects",
        chapter_ref="Class 1 Mathematics Ch. 3",
        source="NCERT Class 1 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Counting numbers 1 to 10"],
        learning_objectives=["Combine two small groups of objects and count the total", "Write simple addition equations with + and ="],
        common_misconceptions=["Recounting the same object twice", "Confusing + with taking away"],
    ),
    (2, "place_value"): CurriculumReference(
        board="CBSE/NCERT",
        grade=2,
        chapter="Counting in Tens",
        topic="Place value (tens and ones)",
        chapter_ref="Class 2 Mathematics Ch. 2",
        source="NCERT Class 2 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Counting up to 100", "Grouping by 10"],
        learning_objectives=["Represent two-digit numbers using bundles of tens and single ones", "Compare numbers using tens first"],
        common_misconceptions=["Treating the tens digit as ones (reading 34 as 3 + 4 = 7)"],
    ),
    (3, "multiplication"): CurriculumReference(
        board="CBSE/NCERT",
        grade=3,
        chapter="How Many Times?",
        topic="Multiplication using arrays",
        chapter_ref="Class 3 Mathematics Ch. 9",
        source="NCERT Class 3 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Skip counting", "Repeated addition"],
        learning_objectives=["Model multiplication as equal rows and columns", "See why a x b equals b x a"],
        common_misconceptions=["Adding rows and columns instead of multiplying them"],
    ),
    (4, "equivalent_fractions"): CurriculumReference(
        board="CBSE/NCERT",
        grade=4,
        chapter="Halves and Quarters",
        topic="Equivalent fractions",
        chapter_ref="Class 4 Mathematics Ch. 9",
        source="NCERT Class 4 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Equal parts of a whole", "Unit fractions 1/2 and 1/4"],
        learning_objectives=["Recognize that 1/2, 2/4, and 3/6 cover the exact same amount of a whole"],
        common_misconceptions=["Thinking larger denominator means larger fraction"],
    ),
    (5, "adding_fractions"): CurriculumReference(
        board="CBSE/NCERT",
        grade=5,
        chapter="Parts and Wholes",
        topic="Adding fractions with different denominators",
        chapter_ref="Class 5 Mathematics Ch. 4",
        source="NCERT Class 5 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Meaning of numerator and denominator", "Equivalent fractions"],
        learning_objectives=["Partition unlike fractions into equal-sized parts using a common denominator", "Add numerators accurately while keeping the common denominator"],
        common_misconceptions=["Adding numerators and denominators straight across (1/2 + 1/3 = 2/5)"],
    ),
    (6, "integers"): CurriculumReference(
        board="CBSE/NCERT",
        grade=6,
        chapter="Integers",
        topic="Integers and number lines",
        chapter_ref="Class 6 Mathematics Ch. 6",
        source="NCERT Class 6 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Whole numbers", "Number line ordering"],
        learning_objectives=["Locate positive and negative integers on a number line", "Add and subtract integers using directed steps"],
        common_misconceptions=["Thinking -8 is greater than -3 because 8 > 3"],
    ),
    (7, "ratios_percentages"): CurriculumReference(
        board="CBSE/NCERT",
        grade=7,
        chapter="Comparing Quantities",
        topic="Ratios and percentages",
        chapter_ref="Class 7 Mathematics Ch. 7",
        source="NCERT Class 7 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Fractions and decimals", "Equivalent ratios"],
        learning_objectives=["Convert ratios and fractions into percentages out of 100", "Compare quantities fairly using percent"],
        common_misconceptions=["Forgetting that percent means parts per 100"],
    ),
    (8, "linear_equations"): CurriculumReference(
        board="CBSE/NCERT",
        grade=8,
        chapter="Linear Equations in One Variable",
        topic="Solving linear equations",
        chapter_ref="Class 8 Mathematics Ch. 2",
        source="NCERT Class 8 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Algebraic expressions", "Inverse operations"],
        learning_objectives=["Maintain balance by applying the same operation to both sides", "Solve ax + b = cx + d and verify by substitution"],
        common_misconceptions=["Changing side without changing sign when transposing terms"],
    ),
    (9, "polynomials"): CurriculumReference(
        board="CBSE/NCERT",
        grade=9,
        chapter="Polynomials",
        topic="Zeros of polynomials and geometric meaning",
        chapter_ref="Class 9 Mathematics Ch. 2",
        source="NCERT Class 9 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Algebraic identities", "Cartesian plane"],
        learning_objectives=["Relate zeros of p(x) to x-intercepts of y = p(x)", "Factorize quadratics cleanly"],
        common_misconceptions=["Confusing degree of a polynomial with number of terms"],
    ),
    (10, "trigonometry"): CurriculumReference(
        board="CBSE/NCERT",
        grade=10,
        chapter="Introduction to Trigonometry",
        topic="Trigonometric ratios in right triangles",
        chapter_ref="Class 10 Mathematics Ch. 8",
        source="NCERT Class 10 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Right-angled triangles", "Pythagoras theorem", "Triangle similarity"],
        learning_objectives=["Define sin, cos, and tan from opposite, adjacent, and hypotenuse", "See why ratios depend only on the angle, not triangle size"],
        common_misconceptions=["Mixing up opposite and adjacent sides relative to the chosen acute angle"],
    ),
    (11, "limits"): CurriculumReference(
        board="CBSE/NCERT",
        grade=11,
        chapter="Limits and Derivatives",
        topic="Functions and limits",
        chapter_ref="Class 11 Mathematics Ch. 12",
        source="NCERT Class 11 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Functions and graphs", "Factorization"],
        learning_objectives=["Distinguish f(a) from the limit as x approaches a", "Visualize left-hand and right-hand approach"],
        common_misconceptions=["Assuming a limit cannot exist if f(a) is undefined"],
    ),
    (12, "differentiation"): CurriculumReference(
        board="CBSE/NCERT",
        grade=12,
        chapter="Continuity and Differentiability",
        topic="Derivatives as secant-to-tangent limits",
        chapter_ref="Class 12 Mathematics Ch. 5",
        source="NCERT Class 12 Mathematics Syllabus",
        verification_status="verified_metadata",
        prerequisites=["Limits", "Slope of a line"],
        learning_objectives=["See the derivative f'(x) as the limiting slope of secants as h -> 0", "Connect derivative sign to increasing/decreasing curve behavior"],
        common_misconceptions=["Confusing function value f(x) with instantaneous rate of change f'(x)"],
    ),
}


class LessonRequest(BaseModel):
    grade: int = Field(default=5, ge=1, le=12)
    board: str = "CBSE/NCERT"
    subject: str = "Mathematics"
    chapter: str = ""
    topic: str
    duration_minutes: float = Field(default=1.5, ge=0.5, le=25.0)
    difficulty: Literal["foundational", "standard", "challenge"] = "standard"
    teaching_style: TeachingStrategy = "concrete_to_abstract"
    language: str = "English"
    theme: str = "light"
    format_name: str = "widescreen_hd"
    custom_notes: str = ""


class LessonSpec(BaseModel):
    request: LessonRequest
    grade_profile: GradeProfile
    curriculum_ref: CurriculumReference
    production_mode: Literal["quick_explanation", "standard_lesson", "detailed_lesson", "derived_short"]


def build_lesson_spec(req: LessonRequest) -> LessonSpec:
    prof = get_grade_profile(req.grade)
    t_low = req.topic.lower()
    matched_ref: CurriculumReference | None = None
    for (g_idx, key), ref in CBSE_MATH_INDEX.items():
        if g_idx == req.grade and (key in t_low or ref.topic.lower() in t_low):
            matched_ref = ref.model_copy(deep=True)
            break
    if matched_ref is None:
        matched_ref = CurriculumReference(
            board=req.board or "curriculum_neutral",
            grade=req.grade,
            subject=req.subject,
            chapter=req.chapter or req.topic,
            topic=req.topic,
            source="User-supplied notes" if req.custom_notes.strip() else "Curriculum-neutral pedagogical specification",
            verification_status="user_supplied" if req.custom_notes.strip() else "curriculum_neutral",
            learning_objectives=[f"Understand {req.topic} visually and solve a verified example"],
        )
    mode = "quick_explanation" if req.duration_minutes <= 2.0 else ("standard_lesson" if req.duration_minutes <= 8.5 else "detailed_lesson")
    if req.format_name == "shorts_hd" and req.duration_minutes <= 1.25:
        mode = "derived_short"
    return LessonSpec(
        request=req,
        grade_profile=prof,
        curriculum_ref=matched_ref,
        production_mode=mode,  # type: ignore[arg-type]
    )
