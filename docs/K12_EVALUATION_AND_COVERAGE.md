# Math Shorts Studio — K–12 Evaluation Rubric, Curriculum Matrix & Benchmarks

## 1. Curriculum Coverage Matrix (Verified Metadata + Reusable Components)

| Grade Band | Classes | Reusable Visual Components | Deterministic Math Engine | Curriculum Metadata |
|---|---|---|---|---|
| **1–2** | 1, 2 | `counting_group`, `number_line`, `array_grid`, `concept_card` | Integer steps, `place_value_decompose`, counting/grouping | CBSE/NCERT Ch. 2–3 + `curriculum_neutral` |
| **3–5** | 3, 4, 5 | `array_grid`, `fraction_bar`, `fraction_circle`, `number_line`, `concept_card` | `array_multiplication_facts`, `exact_fraction_add` (`Fraction`), `exact_fraction_equivalent` | CBSE/NCERT Ch. 4, 9 + `curriculum_neutral` |
| **6–8** | 6, 7, 8 | `number_line`, `equation_balance`, `bar_chart`, `axes`, `polygon`, `concept_card` | `integer_number_line_op`, `ratio_and_percentage_steps`, `solve_linear_equation_steps` | CBSE/NCERT Ch. 2, 6, 7 + `curriculum_neutral` |
| **9–10** | 9, 10 | `axes`, `function_plot`, `polygon`, `table`, `equation`, `concept_card` | `solve_quadratic_steps`, `right_triangle_trig` | CBSE/NCERT Ch. 2, 8 + `curriculum_neutral` |
| **11–12** | 11, 12 | `axes`, `function_plot`, `parametric_curve`, `vector`, `matrix`, `vector_field`, `mechanical_body` | `calculus_secant_to_tangent`, Cash–Karp RK45 ODE/spectral/recurrence models | CBSE/NCERT Ch. 5, 12 + `curriculum_neutral` |

## 2. Structured Teacher & Learner Evaluation Rubric

Automated QA (`audit_educational_quality`) evaluates mathematical accuracy, layout/overlap, theme contrast, caption sync, audio loudness, and media streams, **but cannot replace actual human or student learning evaluation**. Each human review records:

1. **Mathematical Accuracy (1–5):** Are all definitions, steps, fractions, and final answers exact?
2. **Age & Grade Suitability (1–5):** Do vocabulary, sentence length, and notation match the student's class?
3. **Narration Clarity & Pacing (1–5):** Does the voice-over sound like a clear teacher with natural pauses?
4. **Visual Usefulness (1–5):** Do diagrams directly illustrate the concept rather than decorating empty axes?
5. **Caption Readability (1–5):** Are captions 1–2 lines, properly punctuated, and free of diagram collisions?
6. **Worked Example & Practice Quality (1–5):** Are steps easy to follow and distractors plausible yet uniquely wrong?

*Status:* Live classroom/student testing has not yet been conducted in this environment (`human_review_required` until evaluated with consenting teachers/learners).

## 3. Hardware & Zero-Paid-API Benchmarks (GTX 1650-class, 16 GB RAM)
- **Primary AI Providers:** Free-tier Gemini (`gemini_api` / `GEMINI_API_KEY`) and optional NVIDIA NIM (`NVIDIA_API_KEY`) / OpenRouter / local Ollama / llama.cpp; deterministic offline fallback requires **$0 API cost**.
- **Voice & Rendering:** Local Kokoro-82M CPU TTS (`48000 Hz`, `-15.1` to `-15.2 LUFS`), local HyperFrames Chromium CDP + offline KaTeX `0.19.1`, local ManimGX + `h264_nvenc` / FFmpeg.

## 4. Known Limitations
- Multi-subject expansion (Physics, Chemistry, Biology, CS) shares the `LessonSpec` and `SceneV3` architecture, while verified syllabus metadata and deterministic domain helpers are currently populated for **Mathematics (Classes 1–12)**.
- Long-form 10–20 minute detailed lessons require an active LLM provider (`gemini` or `nvidia`) for rich multi-chapter narrative variety; offline deterministic mode produces structured 3–4 scene lessons.
- ManimGX (`scenes/v3_scene.py`) and HyperFrames (`studio/renderers/hyperframes/`) both support the 6 K–12 components, with HyperFrames providing the richest SVG/HTML typography and `--studio` preview parity.
