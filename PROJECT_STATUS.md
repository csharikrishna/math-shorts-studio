# Math Shorts Studio — Project Status (Adaptive Educational Engine)

## Newly Implemented & Verified
- **Creative Intelligence, Storylines, Dialogue & Script IR (`studio/creative/`, `studio/design/`, `studio/storyboard/`)**
  - Separated pre-render filmmaking into `CreativeDirector` (`ConceptAnalysis`, 3 distinct `StorylineOption` strategies, `CreativeBrief` exported as `STORYLINES.md`) -> renderer-independent `ScriptIR` + revision log (`SCRIPT.md`) -> shot-level `Storyboard` (`STORYBOARD.md`) -> unified QA audit (`VideoReviewReport` exported as `REVIEW.md`).
  - Added 6 pedagogical dialogue profiles (`single_educator`, `conversational_explainer`, `socratic`, `discussion`, `documentary`, `technical_instructor`), anti-cliché/filler detection, speaker-turn merging, and per-speaker voice routing in `build_voice_timeline()` (`studio/audio/tts.py`).
  - Defined explicit visual `DesignSpec` (`DESIGN.md`) with 4 named looks (`midnight_studio`, `obsidian_chalk`, `cartesian_stone`, `editorial_paper`), aspect-aware `SafeZoneSpec` (16:9 vs 9:16), and closed-form Motion Canvas easing/spring specs.
- **Discoverable Component Registry & Generalized Computational Engines (`studio/registry/`, `studio/ai/math_truth.py`, `studio/ai/world_model.py`, `studio/ai/v3_schema.py`, `scenes/v3_scene.py`)**
  - Implemented dynamic `ComponentRegistry` (`capabilities_prompt_summary()`) across `math_operation`, `visual_object`, `animation_mechanism`, `composition_template`, `caption_style`, and `audio_cue`.
  - Generalized `math_truth.py` and `world_model.py` beyond benchmark routines with arbitrary `definite_integral` (Riemann partitions), `rk4_ode_trajectory` (1D/2D ODEs), `matrix_transform_and_eigen` (2x2 determinants, eigenvalues, eigenvectors, dynamical classification), `discrete_pmf_convolution` (arbitrary float support), `fourier_partial_sum_terms` (midpoint trigonometric projection), `newton_raphson_trajectory`, `solve_equation_roots`, `compose_expressions`, and `markov_chain_trajectory`.
  - Extended `VideoPlanV3`, `validate_v3.py`, and `V3Scene` with 7 new `ObjectKind`s (`matrix`, `table`, `data_series`, `graph_network`, `probability_region`, `particle_system`, `waveform`), 10 new `EventOp`s (`reveal`, `morph`, `duplicate`, `group`, `zoom`, `trace`, `iterate_states`, `interpolate_values`, `sequence`, `parallel`), 4 new `ClaimRuleKind`s, automatic multi-equation/annotation slot separation, and persisted `axes_id` inheritance.
- **Dual-Backend Rendering & OSI-Clean HyperFrames Adapter (`studio/renderers/`)**
  - Introduced `RenderBackend` (`ManimGXBackend`, `HyperFramesBackend`, `benchmark_backends()`) with local path-traversal-guarded loopback HTTP serving (`validate_safe_relpath`), CDP `Fetch.enable` offline isolation, inline SVG/HTML5 closed-form seek runtime (`window.__HF_SEEK(t)` with zero GSAP/CDN/eval), DOM layout bounds auditing (`__HF_AUDIT_LAYOUT`), and 6-word active caption windows.
- **Dependency-Aware Caching, Cohesive Long-Form & Derived Portrait Shorts (`studio/storage/cache.py`, `studio/longform.py`, `studio/qa/engine.py`, `studio/pipeline.py`)**
  - Added `ArtifactCache` with multi-stage dependency tracking (`research`, `knowledge`, `script`, `storyboard`, `design`, `math_model`, `tts`, `captions`, `scene_render`, `chapter_assembly`) and SHA-256 tamper verification.
  - Replaced shallow `target/20` long-form slicing with conceptual chapter orchestration (`_ChapterBridge` continuity) + `derive_recomposed_short()` native `9:16` recomposition + `audit_educational_quality()` claim coverage classification (`deterministically_verified`, `numerically_tested`, etc.).
- **Production Verification & Documentation (`docs/OPEN_SOURCE_ADOPTION.md`, `docs/LICENSE_MANIFEST.md`, `docs/BENCHMARK_REPORT.md`)**
  - **9–11 Min Autonomous Long-Form ("Eigenvalues, Phase Space, and Dynamical Stability"):** `output/LF_eigenvalues_phase_space_and_dynamical_stability_widescreen_hd.mp4` (`688.62s` / `11m 28.6s`, 7 chapters, 38 scenes, `-15.2 LUFS`, `48000 Hz`, `drift=0.05s`, `0` chapter layout issues, `REVIEW.md` score `100.0/100`).
  - **Derived Native 9:16 Short:** `output/SHORT_eigenvalues_phase_space_and_dynamical_stability_shorts_hd.mp4` (`58.40s`, `layout: no problems in 6 samples`, `-15.3 LUFS`, `REVIEW.md` score `100.0/100`).
  - **ManimGX vs HyperFrames Benchmark (`58.4s` @ `1080x1920 60fps`):** Baseline `ManimGXBackend` `84.506s` / `HyperFramesBackend` `266.696s` improved via optimization pass (`--preset ultrafast --crf 12` raw intermediate, `-vn` loudnorm/QA, `_probe_encoder` cache, in-memory Kokoro `24->48 kHz` resample, lazy `kokoro_available()`, cross-backend shared voice/SFX cache, `HyperFrames` `__HF_STEP` static-frame skip `203/3498` + single-pass `h264_nvenc`, concurrent long-form scripting & HyperFrames chapters, `widescreen_fast` preset) to `67.928s` cold / `25.312s` warm (`ManimGX`) and `67.098–78.086s` cold / `26.680s` warm (`HyperFrames`).
  - **Test Suite:** 45 unit, generalization, security, and optimization tests passing (`tests/test_generalization.py` + regression suites).

---

# Prior Baseline (V3.5)

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
