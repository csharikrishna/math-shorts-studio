# Math Shorts Studio — Project Status (V3.4)

## Implemented & Tested
- **Deterministic Mathematical World Model & Construction Registry (`studio/ai/world_model.py`, `studio/ai/v3_schema.py`)**
  - Typed `ConstructionOp` (`right_triangle`, `square_on_segment`, `pythagorean_dissection`, `translate_polygon`, `rotate_polygon`) materialized deterministically into `SceneObject` polygons.
  - Geometric verification helpers: `is_square`, `is_right_triangle`, `triangles_congruent`, `polygons_overlap` (Sutherland-Hodgman convex clip), `polygon_inside`, `verify_partition`, `all_coords_finite`.
  - Structured mathematical claims (`StructuredClaim`: `right_triangle`, `square`, `congruent_triangles`, `area_conservation`, `partition_of`, `nonzero_area`) validated against live polygon vertices in `validate_v3_plan()`.
  - Event ordering and monotonic narration-bookmark order enforcement in `validate_v3_plan()`.
- **Composable V3 Renderer & Adaptive Composition (`scenes/v3_scene.py`)**
  - Composition-aware per-group bounding-box scaling (`_get_geom_frame`) shared across polygons, points, lines, vectors, circles, annotations, and `move` shifts.
  - Selective equation pill cards (`card: bool = False` on `SceneObject`), smooth `transform` across equations, and `ValueTracker` parameter animations.
- **Upgraded Pythagorean Dissection Animation (`plans/v3_pythagorean_dissection_proof.json`)**
  - Uses deterministic `constructions`, `claims`, and narration-anchored `rotate`, `focus`, `highlight`, and `transform` events (`40.15s`, `1080x1920 @ 60fps`, `layout: no problems in 14 samples`).
- **Gemini V3 Creative Director & Provenance (`studio/ai/director.py`)**
  - Retains raw provider JSON (`*_raw.json`) and repair logs (`*_repairs.json`) alongside validated plans.
  - Proven generalization across 4 new Gemini-generated V3 benchmarks:
    - `plans/v3_gradient_descent_optimization.json` -> `output/V3_v3_gradient_descent_optimization_shorts_hd.mp4` (`33.85s`)
    - `plans/v3_bayes_theorem_conditional_probability.json` -> `output/V3_v3_bayes_theorem_conditional_probability_shorts_hd.mp4` (`26.35s`)
    - `plans/v3_central_limit_theorem.json` -> `output/V3_v3_central_limit_theorem_shorts_hd.mp4` (`36.95s`)
    - `plans/v3_constructing_squares_on_a_right_triangle_using_r.json` -> `output/V3_v3_constructing_squares_on_a_right_triangle_using_r_shorts_hd.mp4` (`31.65s`, invokes `ConstructionRegistry` automatically via Gemini)
- **Voice-Over Benchmarking & Math Speech Normalization (`studio/audio/tts.py`, `studio/audio/pronounce.py`, `build/voice_benchmark/benchmark_results.json`)**
  - Benchmarked 5 Kokoro-82M voices (`am_michael`, `am_adam`, `af_heart`, `af_bella`, `bm_george`) across 3 rates (`-6%`, `-4%`, `+0%`) plus `edge-tts` (`en-US-AndrewNeural`) on dev hardware (`17.01 GB` RAM, `7.27 GB` avail, CPU PyTorch 2.15).
  - Warm Kokoro-82M CPU RTF measured at `0.539–0.620` (faster than real time).
  - `normalize_for_speech_with_map()` converts `sin(x)`, `a^2`, `c²`, `1/2` while preserving `<bookmark/>` and `*emphasis*`.
- **Shared Semantic Subtitles (`studio/timeline.py`, `studio/layout/components.py`)**
  - `build_semantic_captions()` groups timed words into grammatical 1–2 line phrases shared across burned-in `CaptionCard`, `.srt`, `.vtt`, and `*_captions.json`.
- **Long-Form Chapter Production Pipeline (`studio/longform.py`, `scripts/build_longform_demo.py`)**
  - Integrates `KnowledgeTree`, `LongFormPlan`, `ContinuityState`, `ChapterSpec`, checkpoint recovery (`checkpoint.json`), FFmpeg concat assembly, chapter timestamps (`chapters.json` / `chapters.txt`), merged semantic `.srt`/`.vtt`, and transcripts.
  - Rendered 13-chapter 5-minute landscape video: `output/LF_math_unity_5min_widescreen_hd.mp4` (`302.07s` / `5m 02s`, `1920x1080 @ 60fps`, `48000 Hz`, `-15.3 LUFS`, `drift=0.05s`).
- **Automated Regression Suite (`tests/test_core.py`, `tests/test_v2_modules.py`, `tests/test_v3.py`, `tests/test_v34.py`)**
  - 29 unit and regression tests passing.

## Partially Implemented
- **Optional Heavy Local TTS / Forced Alignment**
  - `resemble-ai/chatterbox` (~500M Llama) and `QwenLM/Qwen3-TTS` (0.6B–1.7B) were evaluated against dev hardware (CPU-only PyTorch 2.15, 7.27 GB free RAM); not installed in `.venv` due to lack of CUDA PyTorch runtime for 0.5B–1.7B autoregressive TTS. Kokoro-82M (`hexgrad/Kokoro-82M`, Apache-2.0) + `edge-tts` fallback remain active.
  - `faster-whisper` forced-alignment post-check is not installed; word timestamps come directly from Kokoro phoneme/token timings and Edge-TTS `WordBoundary` events via `align_words()`.
- **Automatic Gemini Long-Form Chapter Scripting**
  - `studio/longform.py` orchestrates any `LongFormPlan` + `list[VideoPlanV3]`; automated multi-chapter Gemini generation is currently driven via `create_v3_plan()` per chapter rather than a single monolithic Gemini call.
