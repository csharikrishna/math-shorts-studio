# Math Shorts Studio — Project Status (V4.2 Educational Visual Reasoning & K–12 Educational Studio)

## V4.2 & K–12 Studio Implemented & Verified
- **V4.2 Semantic Verification, Layout/Caption Fixes & Remote Sync (`studio/ai/visual_intelligence.py`, `studio/renderers/hyperframes/`, `studio/repair.py`, `studio/qa/engine.py`, `scenes/v3_scene.py`)**
  - Pushed all 14 local V4.1 commits to `origin/main` cleanly (`6746406..94d0480`).
  - Fixed `bar_chart` oversized single-bar width (`barW` capped cleanly in both HyperFrames & ManimGX), `mechanical_body` coordinate range parity between `el["mech"]` and `sweep_frames_px`, and `runtime.py` inter-cue caption fallback (eliminating full-beat paragraph dumps at cue boundaries).
  - Added `verify_beat_narration_semantics()` and `humanize_visible_label()` checking critical/overdamped non-oscillation consistency, upright pendulum equilibrium orientation, potential landscape vs `bar_chart` substitution, and internal identifier/placeholder leaks (`phase_orbit`, `time_series_x`, `Mystery Annotation`).
- **K–12 Educational Intelligence, Deterministic School Math & Reusable Components (`studio/education/`, `studio/ai/v3_schema.py`, `studio/registry/visuals.py`)**
  - Implemented `GradeProfile` for all 5 grade bands (`1-2`, `3-5`, `6-8`, `9-10`, `11-12`), `CurriculumReference` + verified `CBSE_MATH_INDEX` (Classes 1–12) with `curriculum_neutral` fallback, and `studio/education/math_engine.py` (exact `Fraction` rational arithmetic, linear/quadratic solvers, arrays, place value, integers, ratios/percentages, right-triangle trigonometry, calculus secant-to-tangent limits, verified worked examples & MCQs).
  - Added 6 composable K–12 visual components (`fraction_bar`, `fraction_circle`, `array_grid`, `counting_group`, `equation_balance`, `concept_card`) registered in `studio/registry/visuals.py` (`verify_registry_contracts() == []`) and rendered in HyperFrames SVG.
  - Built the local K–12 Educational Studio web UI (`studio/education/studio_app.py`, `--studio` CLI flag) strictly on `127.0.0.1` with Class 1–12 selector, curriculum/theme/format controls, lesson workspace, and output library.
- **Rendered K–12 Videos & Full Regression Suite (`tests/test_v42_k12.py`)**
  - `k12_c1_addition_light_widescreen_hd.mp4` (`48.40s`, **`100.0/100`**, `publication_approved`), `k12_c3_arrays_short_light_shorts_hd.mp4` (`53.60s`, **`100.0/100`**, `publication_approved`), `k12_c5_fractions_light_widescreen_hd.mp4` (`64.10s`, **`100.0/100`**, `publication_approved`), `k12_c8_linear_eq_dark_widescreen_hd.mp4` (`72.60s`, **`100.0/100`**, `publication_approved`), `k12_c12_derivatives_dark_widescreen_hd.mp4` (`97.60s`, `78.0/100`, `review_required`).
  - **Test Suite:** **106 tests passing** (`unittest discover -s tests -p "test_*.py"`).

---

# Prior Baseline: Math Shorts Studio — Project Status (V4.1 Visual Intelligence, Adaptive Cinematography & Educational Quality)

## V4.1 Implemented & Verified
- **Confirmed Mathematical Reliability Fixes & Typed Constraints (`studio/ai/simulation.py`, `studio/ai/linked_models.py`, `studio/ai/validate_v3.py`, `studio/ai/v3_schema.py`)**
  - Replaced misleading step-doubling label with true Cash–Karp adaptive Runge–Kutta 4(5) (`"rk45"`) vs explicit `"rk4_adaptive"`, enforced error tolerance rejection when `h <= min_h`, flagged undamped resonance singularities (`peak_gain=None`) cleanly instead of inventing finite amplitudes, removed silent damped-oscillator fallbacks and default `initial_state` padding, added explicit mass/force/unit metadata, supported `"energy"` / `"mechanical_energy"` derived aliases, and enforced `verify_v3_plan_detailed()` / `math_truth` checks across 8 typed constraints (`state_bounds`, `positivity`, `conservation_law`, `energy_dissipation`, `periodicity`, `numerical_accuracy`, `parameter_validity`, `physical_units`).
  - Rejects hollow `trace` operations whose target object has no time/parameter curve geometry.
- **Beat-Level Visual Intelligence & Linked Scientific Compositions (`studio/ai/visual_intelligence.py`, `studio/registry/compositions.py`, `studio/repair.py`, `studio/layout/solver.py`, `studio/ai/world_model.py`)**
  - Implemented beat-level `VisualBeatDecision` reasoning (`plan_beat_visuals`) connected directly to layout composition, mathematical bindings, and scene animations across 6 registered linked compositions (`physical_system_displacement_graph`, `physical_system_phase_diagram`, `synchronized_time_series`, `parameter_sweep_response`, `state_conservation_view`, `multi_view_dashboard`).
  - Added general layout and visual repair (`repair_plan_scenes`, disjoint `0.465`/`0.535` horizontal panel splitting via `fit_axes_pixel_box`, flat conserved-invariant `H(x,y)` `is_flat_invariant` y-range expansion, `"sweep"` & `"conserved_invariant"` bindings, carried-panel slot assignment, nonlinear equilibrium IC offsetting in `ode_phase_trajectory` + `math_models`, bare-axes curve injection, static point/body linking, and clean `editorial`/`minimal` phrase emphasis captions).
- **Content-Aware QA, Cross-Renderer Metadata Parity & Manifest SHA Binding (`studio/qa/frames.py`, `studio/qa/engine.py`, `studio/provenance.py`, `studio/renderers/hybrid.py`, `studio/renderers/hyperframes/compiler.py`, `scenes/v3_scene.py`)**
  - Upgraded `frames.py` to inspect interior axes primary-content coverage (catching visually empty coordinate axes despite border tick ink), chapter-scoped `(ch_pref, ax_id)` data-clipping checks, and independent-panel overlaps while exempting parent–child `axes_id` overlays.
  - Fixed `hybrid.py` `seg_{idx:02d}_{backend}` scene-frame metadata merging, `hyperframes/compiler.py` `.hf-eq-card` top-clipping + inherited axis cleanup on `remove`, ManimGX portrait tick readability (`24px`), `math.ceil(0.65 * total_dyn)` dynamic scene coverage threshold in `audit_educational_quality()`, and cryptographic `qa_report_sha256` manifest binding (`inspect_manifest_verification`).
- **Benchmark Improvement & Regression Suite (`tests/test_v41.py`, `docs/BENCHMARK_REPORT.md`)**
  - **Representative Light/Dark Render Suite (`build/v41_exp1/`):** `osc_hf_light` (`100.0/100`), `osc_mgx_dark` (`100.0/100`), `lv_hf_dark` (`94.0/100`), `lv_mgx_light` (`94.0/100`), all `-15.1` to `-15.7 LUFS`, `60 fps`, `48000 Hz`, `drift=0.00 s`.
  - **Harmonic Oscillator Documentary (`editorial_light`, `hybrid`, `16` scenes):** Upgraded from V4.0 `72.0/100` (`review_required`) to **`88.0/100` (`publication_approved`, `approved=True`, `edge_clips=0`, `empty_mids=0`, `overlaps=0`, `data_clips=0`, `production_certified=True`)**; derived `9:16` Short **`100.0/100` (`publication_approved`)**.
  - **Lotka–Volterra Documentary (`midnight_studio`, `hybrid`, `14` scenes):** Upgraded from V4.0 `78.0/100` (`review_required`) to **`100.0/100` (`publication_approved`, `approved=True`, `0` blockers, `0` warnings, `3.45x` warm-cache speedup `69.80 s -> 20.26 s`, `production_certified=True`)**; derived `9:16` Short **`94.0/100` (`publication_approved`)**.
  - **Held-Out Unseen Topic (`"Nonlinear Damped Pendulums, Separatrices, and Phase Portraits"`, `light`, `hybrid`, `13` scenes, clean commit `d2bc406`):** Derived `32.70 s` Short **`100.0/100` (`publication_approved`, `certified=True`)**; `154.02 s` Documentary **`42.0/100` (`rejected`, `11.27x` warm-cache speedup `264.58 s -> 23.48 s`, `math_accuracy=passed`, `-15.1 LUFS`, `drift=0.00 s`)** — strict V4.1 frame QA caught `bar_chart`/`mechanical_body` panel overlaps (`c1__scene_1`, `c3__scene_3`) cleanly.
  - **Test Suite:** **99 tests passing** (`tests/test_v41.py` + `test_v40.py` + `test_v39.py` + `test_v38.py` + `test_v37.py` + `test_generalization.py` + `test_v35.py` + `test_v34.py` + `test_v3.py` + `test_v2_modules.py` + `test_core.py`).

---

# Prior Baseline: Math Shorts Studio — Project Status (V4.0 Simulation-Driven Educational Filmmaking)

## V4.0 Implemented & Verified
- **Kokoro Trailing-Bookmark Caption Integrity & Strict Semantic Highlighting (`studio/audio/tts.py`, `studio/captions/layout.py`)**
  - Fixed trailing punctuation attached after final `<bookmark/>` tags in `normalize_for_speech_with_map()` and `_synthesize_kokoro()` so raw punctuation tokens can never produce inverted `(start > end)` caption timestamps; enforced non-monotonic timestamp rejection across `verify_caption_synchronization()`.
  - Restricted `extract_semantic_emphasis_tokens()` to genuine technical/concept words so everyday prose never receives background highlight pills.
- **Strict Production Certification, Windows Peak RSS Telemetry & Cost Accounting (`studio/provenance.py`, `studio/qa/engine.py`, `studio/longform.py`)**
  - Enforced non-empty font SHAs, FFprobe stream metrics, `git_commit_sha`, `audio_timeline_sha256`, `caption_ir_sha256`, and approved QA gate inside `inspect_manifest_verification(..., require_certification=True)`, fixing the `REVIEW.md` vs pipeline telemetry `tokens.total` aggregation and adding Windows `K32GetProcessMemoryInfo` peak RSS instrumentation.
- **Linked Parameter Synchronization & Endpoint-Exclusive Spectral Representation (`studio/ai/linked_models.py`, `studio/ai/validate_v3.py`, `scenes/v3_scene.py`, `studio/renderers/hyperframes/`, `studio/renderers/hybrid.py`)**
  - Synchronized `set_param`, piecewise `SweepKeyframe` reset/continuity, active `ValueTracker` / model-linked curves, and `ReplacementTransform` equations across ManimGX, HyperFrames, and Hybrid cross-renderer segment handoffs (including persistent `axes` preservation and cross-segment `remove` carry-over).
  - Switched discrete Fourier evaluation to endpoint-exclusive sampling (`endpoint=False`) with exact DFT frequency bin alignment, detected spectral peaks (`2*`, `5*`), and explicit reconstruction vs filtered-spectrum distinction with zero silent NaN/Inf coercion.
- **General Dynamical Simulation Engine & Scientific Visual Components (`studio/ai/simulation.py`, `studio/registry/`, `studio/ai/v3_schema.py`, `studio/ai/math_truth.py`, `scenes/v3_scene.py`, `studio/renderers/hyperframes/components.py`)**
  - Implemented general sandboxed pure-NumPy/SymPy numerical solvers (RK45 adaptive ODE integration with `dx/dy` equilibrium checks, 2nd-order frequency-response curves & resonance thresholds, discrete recurrences, Markov trajectories, trajectories & conserved-quantity diagnostics) with zero arbitrary LLM Python execution.
  - Added certified scientific components (`mechanical_body`, `vector_field`) across both ManimGX and HyperFrames + offline KaTeX, plus coupled multi-view `bar_chart`/`data_series`/`axes` model bindings and general planning/repair rules (`studio/ai/director.py`, `studio/longform.py`).
- **Multi-Scene Dynamic Motion Gate & Cross-Topic Verification (`tests/test_v40.py`, `docs/BENCHMARK_REPORT.md`)**
  - Upgraded `audit_educational_quality()` to require multi-scene dynamic pixel delta and state-changing animation coverage (`animate_param`/`trace`/`move`/`transform`/`morph`) across documentaries, rejecting static-single-scene docs cleanly.
  - **5-Target Fourier V4.0 Re-Benchmark (`ManimGX` dark, `HyperFrames` light & dark, `Hybrid` dark, `9:16` Portrait Short):** All 5 achieved **`100.0/100` (`publication_approved`, `production_certified=True`)**.
  - **Regenerated Harmonic Oscillator (`"Damped Harmonic Oscillators, Resonance Curves, and Phase Portraits"`):** Derived `40.0 s` Short **`100.0/100` (`publication_approved`)**; `273.52 s` Documentary **`72.0/100` (`review_required`, `0` blockers, `animated_scenes=12/16`, `13.6x` warm cache speedup)**.
  - **Held-Out Nonlinear ODE (`"Lotka-Volterra Predator-Prey Oscillations and Phase Orbits"`):** Derived `37.40 s` Short **`94.0/100` (`publication_approved`)**; `185.32 s` Documentary **`78.0/100` (`review_required`, `0` blockers, `animated_scenes=10/14`, `10.3x` warm cache speedup)**.
  - **Test Suite:** **90 tests passing** (`tests/test_v40.py` + `test_v39.py` + `test_v38.py` + `test_v37.py` + `test_generalization.py` + `test_v35.py` + `test_v34.py` + `test_v3.py` + `test_v2_modules.py` + `test_core.py`).

---

# Prior Baseline: Math Shorts Studio — Project Status (V3.9 Mathematical Cinematography & Linked Visualization)

## V3.9 Implemented & Verified
- **Immutable Production Manifests & Provenance (`studio/provenance.py`, `studio/longform.py`, `studio/pipeline.py`, `studio/renderers/`)**
  - Every generated video produces a sibling `.manifest.json` (`ProductionManifest`) cryptographically binding `production_id`, `git_commit_sha`, script revision/SHA-256, `video_plan_sha256`, `math_model_sha256`, backend/theme/caption configuration, font SHA versions, `audio_timeline_sha256`, `caption_ir_sha256`, final MP4 SHA-256, probed streams, QA status, timestamp, and provider provenance (`verify_manifest_matches_mp4()` detects tampered/overwritten MP4s).
  - Resolved V3.8 `SHORT_` vs `V3_` output overwrite collision cleanly (`output_stem=f"SHORT_{slug}"`).
- **Shared Linked Mathematical Data Model & Genuine Complex Winding (`studio/ai/linked_models.py`, `studio/ai/v3_schema.py`, `studio/ai/world_model.py`, `studio/ai/validate_v3.py`, `studio/registry/math_ops.py`)**
  - Added `LinkedMathModelSpec` (`signal_spectral_pipeline`, `optimization_trajectory`, `bayesian_probabilistic`, `dynamical_eigen`, `series_approximation`) driving coupled views from one mathematical source of truth: $z(t)=x(t)e^{-i\omega t}$, trapezoidal complex centroid $C(\omega)$, derived magnitude spectrum, `rfft`/`irfft` spectral filtering & Parseval energy conservation, synchronized parameter `SweepKeyframe`s, and `ClaimLedgerEntry`.
  - Added 4 linked claims (`fourier_winding_centroid`, `spectral_derivation_consistency`, `signal_filtering_equivalence`, `parseval_energy_conservation`), rejects 1D `cos(x)` function plots masquerading as Fourier winding curves, enforces `ClaimLedgerEntry` spans/assumptions, and never credits self-declared `source_supported` tags without independent verification.
- **Mathematical Cinematography, Offline Vendored KaTeX `0.19.1` & Semantic Captions (`assets/vendor/katex/`, `studio/renderers/hyperframes/`, `scenes/v3_scene.py`, `studio/captions/layout.py`, `studio/layout/`)**
  - Vendored pinned offline KaTeX `0.19.1` (`assets/vendor/katex/KATEX_MANIFEST.json`, `verify_vendored_katex() == []`), rendering equations in HyperFrames offline without CDN access, alongside synchronized `sweep_frames_px` curve/centroid/vector playback, axis labels/legends, and camera `viewBox` zoom reframing.
  - Synchronized ManimGX `ValueTracker` sweeps across coupled curves/points/vectors, external axes labels/curve legends, and `extract_semantic_emphasis_tokens()` so ordinary stopwords (`"Every"`, `"a"`, `"by"`, `"written"`) never get background pill boxes while conceptual tokens highlight cleanly.
- **Global Merged `CaptionIR`, Complete Scene-Aware QA & Synthesized Audio Signal Demos (`studio/captions/layout.py`, `studio/qa/`, `studio/audio/demo_signals.py`, `studio/longform.py`)**
  - Fixed `verify_caption_synchronization()` overlap check (`cs < prev_end - 0.02`), merged all chapter `CaptionIR` files via `merge_chapter_caption_irs()` with rebased offsets, removed the silent `24`-frame QA truncation, exported `scene_frame_metadata.json` across ManimGX/HyperFrames/Hybrid, added local synthesized audio waveform demos (`studio/audio/demo_signals.py`), eliminated post-lock `ext_*` bypasses in `studio/longform.py`, added JSON-mode schema fallback in `studio/providers/gemini.py`, and fixed thread-safe `extra_env` in `studio/pipeline.py`.
- **Verification (`tests/test_v39.py`, Benchmarks & Held-Out Topic Evaluation — `docs/BENCHMARK_REPORT.md`)**
  - **5-Target Fourier Linked Benchmarks:** `light` ManimGX, `midnight_studio` ManimGX (`layout: no problems in 14 samples`), `HyperFrames + KaTeX` (`editorial_light`), `Hybrid`, and `9:16` Portrait Short — all **`publication_approved` (`94.0/100`)**, `-15.2` to `-15.6 LUFS`, `60.00 fps`, `drift=0.00 s`, `manifest_ok=True`.
  - **Held-Out Unseen Topic (`"Damped Harmonic Oscillators, Resonance Curves, and Phase Portraits"`, `hybrid` + derived `shorts_hd`):** Generated 7 chapters / 28 scenes (`225.62 s`, `-15.2 LUFS`, `drift=0.00 s`, `124` frames across `28/28` scenes, `79` global captions) and `44.00 s` short autonomously (`manifest_ok=True`); honest QA flagged `c2__cue_0003` Kokoro trailing-period timestamp inversion (`54.0/100` `rejected` on documentary; `78.0/100` `review_required` on short).
  - **Test Suite:** **81 tests passing** (`tests/test_v39.py` + `test_v38.py` + `test_v37.py` + `test_generalization.py` + `test_v35.py` + `test_v34.py` + `test_v3.py` + `test_v2_modules.py` + `test_core.py`).

---

# Prior Baseline: Math Shorts Studio — Project Status (V3.8 Professional Production & Reliability)

## V3.8 Implemented & Verified
- **Authoritative Script-First Generation & Revision Governance (`studio/longform.py`, `studio/creative/script_ir.py`, `studio/creative/dialogue.py`, `studio/creative/director.py`, `studio/providers/gemini.py`)**
  - Ordered pipeline executes `ConceptAnalysis -> CreativeBrief -> Global ScriptIR -> Editorial Review -> Immutable SHA-256 Approved Revision -> Chapter Scene Compilation` with `strict_ids=True` (no positional beat fallback), separate conceptual `anchor_marks`/`anchor_token_offsets`, unpronounced identifier / cliché / boilerplate-stem guards, and sanitized Gemini JSON schema handling for `dict`-typed models.
- **Enforceable Executable Registry & `SceneRuntimeState` Parity (`studio/registry/`, `studio/renderers/scene_state.py`, `studio/renderers/hybrid.py`, `studio/renderers/hyperframes/`, `scenes/v3_scene.py`)**
  - Enforced typed Pydantic input/output contracts, executable `test_fn`, `validate_fn` / `compute_fn`, and backend render bindings across all registered components (`verify_registry_contracts() == []`); uncertified components fail audit and are excluded from planner prompts.
  - Added renderer-independent `SceneRuntimeState` preserving transformed coordinates/rotations/scales/parameters and `removed_ids` across `ManimGX <-> HyperFrames`, keyframed HyperFrames `move`/`rotate`/`scale`/`zoom` transforms, `TransformMatchingTex` equation transitions, `eq_clearance` axis sizing, `_hidden_persisted` curve restoration across multi-axes scenes, and `\infty` / `\omega` Unicode typography fixes.
- **Canonical Single-Source `CaptionIR`, Strict Publication Gates & Scene-Aware Frame QA (`studio/captions/`, `studio/timeline.py`, `studio/audio/tts.py`, `studio/qa/engine.py`, `studio/qa/frames.py`, `studio/ai/validate_v3.py`, `studio/ai/math_truth.py`)**
  - Compiled `caption_ir.json` once from `AudioTimeline` with word-level `TimingProvenance` (`forced_aligned`, `tts_word_boundary`, `interpolated`, `estimated`) consumed identically by SRT, WebVTT, ASS, ManimGX, HyperFrames, and `studio/preview.py`.
  - Separated `ProductionState` (`draft`, `rendered`, `review_required`, `rejected`, `publication_approved`) so unrendered plans or any `unsupported`/`human_review_required` claims or unverified caption sync/motion can never pass publication approval; verified real Markov row-stochastic & stationary `\pi P = \pi` invariants; and added `inspect_video_scene_aware()` sampling opening, scene start/mid/end, bookmarks, backend switches, and motion deltas with JSON sidecar reports.
- **12-Layer Dependency Caching & Local Scene Preview Workbench (`studio/storage/cache.py`, `studio/preview.py`)**
  - Implemented `compute_dependency_fingerprints()` + `validate_sidecar_bundle()`, plus an offline HTML/loopback (`127.0.0.1`) scene preview workbench (`python -m studio.preview --source ... [--scene ...] [--serve]`) supporting single-scene isolation via `SceneRuntimeState` (`10.19 s` vs full multi-minute render).
- **V3.8 Unseen Documentary, Portrait Short & Regression Verification (`docs/BASELINE_V38_AUDIT.md`, `docs/BENCHMARK_REPORT.md`)**
  - **Unseen Documentary (`"Fourier Analysis, Spectral Decomposition, and Signal Filtering"`, `light`, `editorial`):** `460.52 s` (`8` chapters, `53` scenes), `-15.2 LUFS`, `drift=0.00 s`, `0` dead-air silences, `0` empty midpoints/edge clips, `REVIEW.md` **`publication_approved` (`100.0/100`, `det_verified=52`, `unsupported=0`, `failed=0`)**.
  - **Derived Native `9:16` Portrait Short (`shorts_hd`, `light`, `kinetic`):** `41.90 s` (`5` scenes from climax `c3`), `-15.1 LUFS`, `drift=0.00 s`, `0` empty midpoints/edge clips, `REVIEW.md` **`publication_approved` (`94.0/100`)**.
  - **Dual-Theme/Backend Samples:** `midnight_studio` HyperFrames (`73.10 s`, `-15.3 LUFS`, `0` empty mids) and `light` Hybrid (`53.30 s`, `-15.3 LUFS`, `0` empty mids).
  - **Test Suite:** **64 tests passing** (`tests/test_v38.py` + `test_v37.py` + `test_generalization.py` + `test_v35.py` + `test_v34.py` + `test_v3.py` + `test_v2_modules.py` + `test_core.py`).

---

# Prior Baseline: Math Shorts Studio — Project Status (V3.7 Professional Filmmaking & Visual QA)

## V3.7 Implemented & Verified
- **Audit, Pinned Offline Typography & Unified WCAG Design (`docs/ARCHITECTURE_AUDIT.md`, `studio/design/fonts.py`, `studio/config.py`, `studio/design/spec.py`, `assets/fonts/`)**
  - Pinned SIL OFL-1.1 binaries (`Inter-Variable.ttf`, `SourceSans3-Variable.ttf`, `STIXTwoText-Variable.ttf`, `STIX2Math.otf`) with SHA-256 manifest verification (`verify_pinned_fonts()`).
  - Unified 6 `THEMES` / `NAMED_LOOKS` (`light`, `dark`, `midnight_studio`, `obsidian_chalk`, `cartesian_stone`, `editorial_paper`), enforced WCAG 2.1 contrast (`wcag_contrast_ratio`, `validate_look_contrast() == []` across all 6), and removed silent dark-theme coercion so `light` renders genuinely light frames across ManimGX and HyperFrames.
- **Unified `CaptionIR`, Adaptive Layout Solver & Scene Polishing (`studio/captions/`, `studio/layout/solver.py`, `studio/layout/components.py`, `scenes/v3_scene.py`)**
  - Added 6 cohesive `CaptionStyle` presets (`editorial`, `kinetic`, `minimal`, `dialogue`, `accessible`, `off`), grammatical & pixel-width phrase chunking without orphan conjunction tails, and `export_ass()`.
  - Built shared pixel-space `compute_adaptive_zones`, `fit_axes_pixel_box`, and `resolve_non_overlapping_boxes`, retiring superseded apparatus before obstacle computation, clipping `data_series`/`parametric_curve` runs inside `axes` bounds, separating side-by-side `table`/`bar_chart`/`matrix`/`polygon` slots cleanly, and fixing persistent-axes + future-equation reservation.
- **Locked `ScriptIR`, Dialogue Isolation & Calibrated Pacing (`studio/creative/`, `studio/longform.py`, `studio/pipeline.py`, `studio/ai/director.py`)**
  - Connected `CreativeDirector` (`STORYLINES.md`, `SCRIPT.md`), `lock_script_to_plan()`, measured `AudioTimeline` storyboard alignment, non-overwriting `resolve_speaker_voices()`, concept-depth `_allocate_chapter_budgets()`, and `KOKORO_EFFECTIVE_WPS = 2.08` (`[1.85, 2.28]` word budget).
- **Executable Registry, Modular HyperFrames & Scene-Level `HybridRenderBackend` (`studio/registry/`, `studio/renderers/hyperframes/`, `studio/renderers/hybrid.py`)**
  - Bound executable schemas/self-tests onto `ComponentRegistry` (`verify_registry_contracts() == []`), refactored HyperFrames into a clean modular package (`11` focused modules with offline `@font-face` and Unicode math labels), and implemented `HybridRenderBackend` (`partition_plan_by_backend` + FFmpeg normalized `concat` filter mux onto one canonical `AudioTimeline`).
- **Stability Math Truth, Honest 10-Category QA & Dependency-Verified Caching (`studio/ai/math_truth.py`, `studio/ai/validate_v3.py`, `studio/qa/engine.py`, `studio/storage/cache.py`)**
  - Added `verify_eigenvalue_stability_regime()` (word-boundary checked), row-stochastic Markov diagnostics, pre-render rejection of bare empty `axes` and non-geometric `rotate`, decoded-frame contact sheets (`verify_decoded_frames()`), honest 10-category `audit_educational_quality()` (blocks fake `100/100`), and `ArtifactCache` SHA-256 integrity checks across look/caption keys.
- **V3.7 Production Renders, Visual Inspection & Suite Verification (`docs/BENCHMARK_REPORT.md`)**
  - **Eigenvalues Documentary (`--theme light --caption-style editorial`):** `585.42 s` (`+8.4%` vs `540s`), `7/7` chapters `layout: no problems`, `-15.3 LUFS`, `TruePeak=-1.8 dBFS`, `48000 Hz`, `60.00 fps`, `drift=0.00 s`, `REVIEW.md` **`97.5/100`** (`approved=True`, `edge_clips=0`).
  - **Derived Native Portrait Short (`--derive-short`):** `55.00 s` (`0.0%` vs `55s`), `1080x1920 @ 60.00 fps`, `layout: no problems in 8 samples`, `-15.3 LUFS`, `drift=0.00 s`, `REVIEW.md` **`94.0/100`**.
  - **Unseen Documentary: Markov Chains (`--theme dark --duration 510`):** `490.92 s` (`-3.7%` vs `510s`), `6/6` chapters `layout: no problems`, `-15.3 LUFS`, `TruePeak=-1.7 dBFS`, `48000 Hz`, `60.00 fps`, `drift=0.00 s`, `REVIEW.md` **`90.0/100`** (`edge_clips=0`).
  - **Warm-Cache Theme/Caption Switch (`16.20 s` 1080p60):** `25.524 s` cold -> **`6.864 s`** warm (`3.72x` speedup).
  - **Test Suite:** **55 tests passing** (`tests/test_v37.py` + `tests/test_generalization.py` + regression suites).

---

# Prior Baseline (Adaptive Educational Engine)

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
