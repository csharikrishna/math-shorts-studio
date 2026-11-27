# Math Shorts Studio — Project Status (V3.5)

## Implemented & Tested
- **Extended Mathematical World Model & Computational Registry (`studio/ai/math_truth.py`, `studio/ai/world_model.py`, `studio/ai/v3_schema.py`)**
  - Added symbolic & numerical computation in `math_truth.py`: `eval_function`, `symbolic_derivative`, `eval_derivative`, `is_stationary_point`, `is_local_minimum`, `functions_equivalent`, `gradient_descent_trajectory`, `discrete_convolution`, `distribution_stats`, `bayes_posterior`.
  - Extended `ConstructionKind` & `materialize_constructions()` with: `gradient_descent_path`, `bayes_area_model`, `dice_sum_distribution`, `gaussian_approx`, `taylor_approximation`, `fourier_partial_sum`, plus named `edge_role` (`leg_a`, `leg_b`, `hypotenuse`) on `square_on_segment` with strict dependency checks (no silent fallback on missing `source_id`).
  - Added `is_convex_polygon` and `has_self_intersection` geometry checks.
- **Structured Mathematical Claims & Post-Transform Verification (`studio/ai/validate_v3.py`)**
  - `verify_v3_plan_detailed(plan, require_claims=True)` verifies `point_on_function`, `stationary_point`, `local_minimum`, `derivative_equality`, `function_equivalence`, `polynomial_degree`, `coefficient_correctness`, `probability_normalization`, `conditional_probability`, `area_equality`, `distribution_stats`, `convergence_assumptions`, plus geometric constraints (`right_triangle`, `square`, `congruent_triangles`, `partition_of`, `area_conservation`).
  - Evaluates geometry after `move` and `rotate` events, rejects incompatible object types, and separates `verified`, `failed`, and `unsupported` claims.
- **True Rigid Transformations & Sequential Cue Execution (`scenes/v3_scene.py`, `plans/v3_pythagorean_dissection_proof.json`)**
  - Fixed `V3Scene` cue compilation to group sequential events per target into `m.Succession(...)`, preventing simultaneous `Create` + `Rotate`/`Move` conflicts.
  - Pythagorean dissection duplicates `tri_1` and rigidly rotates/translates `tri_2` (`90°`), `tri_3` (`180°`), `tri_4` (`270°`) into the `(a+b)x(a+b)` corners at narration bookmarks (`43.65s`, `layout: no problems in 14 samples`).
- **Single-Source Semantic Subtitles & Token Mapping (`studio/timeline.py`, `studio/layout/components.py`, `studio/audio/pronounce.py`, `studio/audio/tts.py`)**
  - `normalize_for_speech_with_map()` builds exact token-level mappings between display notation (`a^2`, `sin(x)`, `1/2`) and spoken tokens while preserving `<bookmark/>` positions.
  - `build_semantic_captions()` is the single source of truth shared by burned-in `CaptionCard.set_segments()`, `.srt`, `.vtt`, and `*_captions.json`.
  - Added `--no-edge-fallback` (`allow_edge_fallback`) to make network Edge-TTS fallback explicitly configurable.
- **Voice-Over Benchmarking (`build/voice_benchmark/benchmark_results.json`)**
  - Evaluated 5 Kokoro-82M voices (`am_michael`, `am_adam`, `af_heart`, `af_bella`, `bm_george`) × 3 rates (`-6%`, `-4%`, `+0%`) + `edge-tts` on dev hardware (`17.01 GB` RAM, `7.27 GB` avail, CPU PyTorch `2.15.0`). Selected `am_michael` (`-4%`, warm `RTF=0.552`) as the balanced default.
- **Autonomous Long-Form Pipeline & Content-Addressed Caching (`studio/longform.py`, `studio/pipeline.py`, `studio/qa/engine.py`)**
  - `create_longform_plan()` autonomously generates `KnowledgeTree` -> `LongFormPlan` -> `ContinuityState` -> validated `VideoPlanV3` chapters with duration-aware extension and provider logs.
  - `chapter_cache_key()` hashes chapter plan, math model, TTS engine/voice/rate, format, theme, UI/look, and renderer/world-model source digest into `JobStore` (`studio/storage/cache.py`), with atomic checkpoint updates (`os.replace`) and `verify_longform_bundle()` end-to-end QA.
- **V3.5 Production Experiments (All QA Passed)**
  - **Pythagorean Dissection**: `output/V3_v3_pythagorean_dissection_proof_shorts_hd.mp4` (`43.65s`, `layout: no problems in 14 samples`, `-15.6 LUFS`, `drift=0.05s`)
  - **Euler's Identity**: `output/V3_v3_euler_identity_rotation_shorts_hd.mp4` (`59.65s`, `layout: no problems in 17 samples`, `-15.4 LUFS`, `drift=0.05s`)
  - **Gradient Descent**: `output/V3_v3_gradient_descent_optimization_using_gradient_des_shorts_hd.mp4` (`43.65s`, `layout: no problems in 6 samples`, `-15.4 LUFS`, `drift=0.05s`)
  - **Bayes' Theorem**: `output/V3_v3_bayes_theorem_conditional_probability_using_baye_shorts_hd.mp4` (`75.35s`, `layout: no problems in 12 samples`, `-15.4 LUFS`, `drift=0.05s`)
  - **Central Limit Theorem**: `output/V3_v3_central_limit_theorem_using_dice_sum_distributio_shorts_hd.mp4` (`72.35s`, `layout: no problems in 15 samples`, `-15.4 LUFS`, `drift=0.05s`)
  - **Autonomous 5-Min Long-Form (`--longform --duration 300 --topic "The Geometry of Calculus and Waves"`)**: `output/LF_the_geometry_of_calculus_and_waves_widescreen_hd.mp4` (`316.72s` / `5m 16.7s`, 19 chapters, `1920x1080 @ 60fps`, `48000 Hz`, `-15.5 LUFS`, `TruePeak=-1.8 dBFS`, `drift=0.05s`)
- **Automated Regression Suite (`tests/test_v35.py` + `tests/test_*.py`)**
  - 40 unit and behavioral regression tests passing.

## Partially Implemented / Hardware-Constrained
- **Heavy Local TTS & Forced Alignment**: `resemble-ai/chatterbox` (~500M), `QwenLM/Qwen3-TTS` (0.6B–1.7B), and `faster-whisper` are not installed in `.venv` due to CPU-only PyTorch (`cuda: False`, 7.27 GB available RAM); Kokoro-82M (`hexgrad/Kokoro-82M`, Apache-2.0) + token-level alignment remain the active local baseline.
