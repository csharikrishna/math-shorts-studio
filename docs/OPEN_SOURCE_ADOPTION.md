# Open-Source Repository Research & Adoption Report

> External research workspace: `c:\Users\cshar\Downloads\math_shorts_research` (kept strictly outside `math-shorts-studio`; untrusted text treated solely as architectural reference).

## Inspected Repositories & Exact Commit SHAs

| Repository | Commit SHA | License | Research Role |
| --- | --- | --- | --- |
| `heygen-com/hyperframes` | `3c85fbcde6eff6218fff2542ddd7473e75eea617` | Apache-2.0 (`LICENSE`; third-party notices in `CREDITS.md`) | Primary adaptation target: creative workflow, storyboard spec, design system, deterministic browser seek/capture architecture, lint gates |
| `souzatharsis/podcastfy` | `053b54a0917e35edb2808049499292361ccaae7b` | Apache-2.0 (`LICENSE`, `pyproject.toml`) | Multi-speaker natural dialogue prompting, long-form contextual continuity, alternating speaker reconciliation |
| `motion-canvas/motion-canvas` | `7b91435c301d530351dcf5ebb91dd139c002e405` | MIT (`LICENSE`) | Reference architecture for declarative closed-form easing, spring physics, flow combinators (`all`/`sequence`), scene lifecycle states |
| `3b1b/manim` | `fafa083a4fb274bba9cabde0b6e2f50ba6da0622` | MIT (`LICENSE.md`) | Reference architecture for `SampleSpace`, `Matrix`, `CoordinateSystem`, `TransformMatchingParts`, ODE vector fields |
| `3b1b/videos` | `31f8ffe785e860477a6051fde983c02c7d165bc2` | CC-BY-NC-SA-4.0 (`LICENSE.txt`) | **Educational reference only** (noncommercial license; zero source code or assets copied into `math-shorts-studio`) |

---

## Feature-by-Feature Adoption Matrix

### 1. Filmmaking Workflow: Storylines, Spec, Storyboard & Visual Review
- **Source & Commit:** `heygen-com/hyperframes` (`3c85fbcde6eff6218fff2542ddd7473e75eea617`)
- **Exact Source Files:**
  - `skills/hyperframes/SKILL.md`
  - `skills/hyperframes/references/story.md`
  - `skills/hyperframes/references/spec.md`
  - `skills/hyperframes/references/storyboard.md`, `storyboard-format.md`
  - `skills/hyperframes/references/review.md`
  - `skills/hyperframes/references/types/explainer.md`, `types/data.md`, `types/vertical.md`
  - `packages/core/src/storyboard/types.ts`, `parseStoryboard.ts`
- **Existing Upstream Behavior:** Separates video creation into explicit stages before animation: identifying one thesis + concrete worked example (`story.md`), generating three distinct storylines (`STORYLINES.md`), locking quantitative design/typography/banned-default constraints (`SPEC.md`), parsing frame-by-frame storyboards (`STORYBOARD.md` -> `StoryboardManifest` with `outline`/`built`/`animated` status and `reads`/`sound`/`poster` metadata), and visual strip/keyframe inspection (`REVIEW.md`).
- **Our Corresponding Capability:** `studio/creative/` (`ConceptAnalysis`, `ExplanationStrategy`, `StorylineOption`, `EditorialReviewReport`), `studio/design/spec.py` (`DesignSpec` + `DESIGN.md`), and `studio/storyboard/model.py` (`Storyboard`, `StoryboardShot` + `STORYBOARD.md` + `REVIEW.md`), adapted specifically for mathematics/science pedagogy rather than SaaS product launches.
- **Integration Approach:** Reimplemented natively in Python + Pydantic (`VideoPlanV3`, `ScriptIR`, `DesignSpec`, `Storyboard`) so artifacts are human-readable (`STORYLINES.md`, `SCRIPT.md`, `STORYBOARD.md`, `DESIGN.md`, `REVIEW.md`) and machine-validated.
- **Dependencies:** Pure Python + Pydantic (no Node/Mintlify required for authoring).
- **License Requirements:** Apache-2.0 pattern attribution; independent Python implementation.
- **Performance Implications:** Consolidates concept understanding + strategy + storyline into 1 structured Gemini request to stay within Gemini free-tier RPD/RPM budgets.
- **Status:** **Adapted & Reimplemented**

### 2. Design Tokens, Anti-Slop Rules & Frame Presets
- **Source & Commit:** `heygen-com/hyperframes` (`3c85fbcde6eff6218fff2542ddd7473e75eea617`)
- **Exact Source Files:**
  - `DESIGN.md`
  - `skills/hyperframes/references/look.md`
  - `skills/hyperframes/references/spec.md` ("BANNED by default")
  - `skills/hyperframes/assets/frame-presets/cartesian/FRAME.md`
  - `skills/hyperframes/assets/frame-presets/code-editorial/FRAME.md`
  - `skills/hyperframes/assets/frame-presets/cartesian/caption-skin.html`
- **Existing Upstream Behavior:** Replaces generic styling with explicit numerical typography ramps (`cqw` responsive sizing), hairline elevation (`1px` subtle borders over heavy drop shadows/glow), scarce single-accent rationing, and explicit banned defaults (no permanent dark glow boxes, no 3-card rows, no idle decorative particles).
- **Our Corresponding Capability:** `studio/design/spec.py` (`DesignSpec`, `NAMED_LOOKS` including `midnight_studio`, `obsidian_chalk`, `cartesian_stone`, `editorial_paper`) with responsive 16:9 landscape vs. 9:16 portrait safe-zone math (`80%` title-safe / `90%` action-safe / platform caption floor).
- **Integration Approach:** Pure Python data models driving CSS custom properties in our HTML composition compiler AND theme/sizing tokens in `SafeLayout` / `V3Scene`.
- **Dependencies:** System fonts / local clean stacks (`Inter`, `Segoe UI`, sans/mono math stacks); zero external Google Fonts CDN fetch during deterministic rendering.
- **License Requirements:** Apache-2.0 reference; original look specs. Note upstream preset fonts like Geist/Cormorant use SIL OFL 1.1 if bundled later.
- **Performance Implications:** Eliminates font-fetch network latency and prevents dark gradient banding in H.264.
- **Status:** **Adapted & Reimplemented**

### 3. Deterministic HTML Composition Runtime, Seek Protocol & Network Isolation
- **Source & Commit:** `heygen-com/hyperframes` (`3c85fbcde6eff6218fff2542ddd7473e75eea617`)
- **Exact Source Files:**
  - `docs/concepts/compositions.mdx`, `determinism.mdx`, `frame-adapters.mdx`
  - `packages/core/src/runtime/state.ts`, `clock.ts`, `applyVariableBindings.ts`
  - `packages/producer/src/services/fileServer.ts` (`VIRTUAL_TIME_SHIM`, `isPathInside`)
  - `packages/engine/src/services/screenshotService.ts`, `browserManager.ts`, `publicNetwork/captureProxy.ts`
  - `packages/lint/src/rules/composition.ts`, `captions.ts`, `core.ts`, `gsap.ts`
- **Existing Upstream Behavior:** Pinned browser capture advances deterministically frame-by-frame (`t = frame / fps`) over a paused timeline (`window.__hf.seek(t)`), seeds PRNG (`mulberry32`), freezes wall-clock `Date`/`performance.now`, isolates local file serving with `isPathInside`, blocks private/external network fetches during render (`captureProxy.ts`), and statically lints composition contracts (`clip`, no `visibility` tweening on `.clip`, caption hard-kill, text overflow bounds).
- **Our Corresponding Capability:** `studio/renderers/hyperframes_backend.py` + `studio/renderers/base.py` common `RenderBackend` contract (`ManimGXBackend`, `HyperFramesBackend`, `select_backend_for_scene`).
- **Integration Approach:** Instead of executing arbitrary LLM-generated JavaScript or depending on external GSAP CDN (`gsap Standard License` per `CREDITS.md` is not OSI open-source), we compile validated Pydantic `SceneV3` / component specs into trusted SVG/HTML5 compositions driven by an inline closed-form seek engine (`window.__HF_SEEK(t)`) + virtual-time shim (`window.performance.now`) + local loopback HTTP server (`127.0.0.1` with `validate_safe_relpath` traversal guard) + local Chrome CDP (`websockets`) with CDP `Fetch.enable` network isolation blocking non-loopback requests.
- **Dependencies:** Local Chrome (`chrome.exe`), Python stdlib `http.server`, `websockets` (BSD-3-Clause, already in `.venv`), `ffmpeg`.
- **License Requirements:** Apache-2.0 architecture adaptation; deliberately **rejected** GSAP proprietary script dependency in favor of pure MIT/original closed-form easing/springs so the repository stays 100% OSI-clean and zero-CDN.
- **Performance Implications:** Fast for typography, overlays, charts, matrices, tables, and chapter cards; compared against ManimGX in `docs/BENCHMARK_REPORT.md` / backend benchmarks for wall time, RSS RAM, VRAM, and output bytes.
- **Status:** **Reimplemented (Core Adapter & Sandbox); Rejected (GSAP proprietary runtime & HeyGen paid media APIs)**

### 4. Registry-Driven Discoverable Components & Chart Story Primitives
- **Source & Commit:** `heygen-com/hyperframes` (`3c85fbcde6eff6218fff2542ddd7473e75eea617`)
- **Exact Source Files:**
  - `packages/core/schemas/registry-item.json`
  - `registry/registry.json`, `registry/components/CATALOG.md`
  - `registry/components/chart-story/chart-story.html`, `animated-bar-chart/animated-bar-chart.html`
  - `registry/blocks/bar-chart-race/bar-chart-race.html`, `registry/examples/nyt-graph/compositions/nyt-chart.html`
- **Existing Upstream Behavior:** Structured JSON-schema metadata per distributable component (`name`, `type`, `description`, `dimensions`, `tags`) with `data-composition-variables`, `what`/`use_when`/`avoid_when`/`variables` agent-discoverable catalog (`CATALOG.md`), and elastic container scaling.
- **Our Corresponding Capability:** `studio/registry/` (`ComponentRegistry`, `ComponentSpec`, `math_ops.py`, `visuals.py`, `animations.py`, `compositions.py`) covering mathematical operations, visual objects, animation ops, and multi-aspect compositions — dynamically synthesizing LLM capability prompts from live registrations.
- **Integration Approach:** Adapted metadata & discoverability contract into Pydantic/Python `ComponentSpec` with typed schemas, aspect ratio support, backend compatibility (`manimgx` vs `hyperframes`), mathematical validation requirements, and resource bounds.
- **Dependencies:** Pure Python + Pydantic.
- **License Requirements:** Apache-2.0 schema pattern reference.
- **Performance Implications:** Removes brittle monolithic prompt prose; lets the director discover valid primitives cleanly per target aspect ratio and backend.
- **Status:** **Adapted & Reimplemented**

### 5. Natural Conversational & Socratic Scripting, Anti-Filler Cleaning & Long-Form Continuity
- **Source & Commit:** `souzatharsis/podcastfy` (`053b54a0917e35edb2808049499292361ccaae7b`)
- **Exact Source Files:**
  - `podcastfy/conversation_config.yaml`, `config.yaml`
  - `podcastfy/utils/config_conversation.py` (`ConversationConfig`)
  - `podcastfy/content_generator.py` (`LongFormContentGenerator`, `ContentCleanerMixin._fix_alternating_tags`, `StandardContentStrategy`, `LongFormContentStrategy`)
  - `podcastfy/client.py` (`process_content`), `podcastfy/text_to_speech.py`
- **Existing Upstream Behavior:** Parameterizes speaker roles, dialogue styles, engagement techniques (`rhetorical questions`, `analogies`), chunks long content while chaining prior `chat_context` across parts with speaker-continuity instructions, strips stage directions/scratchpad markup (`_clean_scratchpad`), merges adjacent same-speaker turns (`_fix_alternating_tags`), and alternates multi-speaker TTS synthesis.
- **Our Corresponding Capability:** `studio/creative/dialogue.py` + `studio/creative/script_ir.py` + multi-speaker `build_voice_timeline()` in `studio/audio/tts.py`, plus `GlobalNarrativePlan` chapter continuity in `studio/longform.py`.
- **Integration Approach:** Adapted configurable conversation profiles (`single_educator`, `conversational_explainer`, `socratic`, `discussion`, `documentary`, `technical_instructor`), speaker turn normalization/merging, and cross-chapter `ContinuityState` into our existing free-tier `LLMProvider` + local Kokoro dual-voice pipeline (`am_michael` + `af_bella`), **explicitly removing** Podcastfy's forced podcast intro greetings ("Welcome to...") and conversational fluff ("Yes, and...") via strict pedagogical anti-filler checks. **Rejected** Podcastfy's mandatory `langchain`, `openai`, `elevenlabs`, and `pydub` dependencies.
- **Dependencies:** Existing `google-genai`, Pydantic, Kokoro-82M / optional `edge-tts`.
- **License Requirements:** Apache-2.0 (`LICENSE`). Compatible cleanly; no third-party code verbatim-copied.
- **Performance Implications:** Preserves single `LLMProvider` abstraction; multi-speaker Kokoro reuses loaded pipelines (`_KOKORO_PIPELINES`) so memory overhead stays bounded (<200 MB extra for second speaker voice tensor).
- **Status:** **Adapted (Prompting, Dialogue Config, Long-Form Continuity, Dual-Voice Synthesis); Rejected (LangChain/OpenAI/ElevenLabs wrappers)**

### 6. Declarative Closed-Form Tweening, Springs, Flow Combinators & Scene State Lifecycle
- **Source & Commit:** `motion-canvas/motion-canvas` (`7b91435c301d530351dcf5ebb91dd139c002e405`)
- **Exact Source Files:**
  - `packages/core/src/tweening/timingFunctions.ts`, `spring.ts`, `interpolationFunctions.ts`, `tween.ts`
  - `packages/core/src/flow/all.ts`, `sequence.ts`
  - `packages/core/src/scenes/SceneState.ts`, `GeneratorScene.ts`, `packages/core/src/app/Renderer.ts`
  - `packages/2d/src/lib/components/Line.ts`, `Latex.ts`
- **Existing Upstream Behavior:** Uses pure mathematical easing functions (`easeInOutCubic`, `easeOutExpo`, `easeOutQuart`), physically damped harmonic spring integration (`F = -kx - cv`), deep property/string interpolation, generator-based `all(...)` / `sequence(delay, ...)` orchestration, and deterministic `SceneState` lifecycle transitions (`Initial -> AfterTransitionIn -> CanTransitionOut -> Finished`). Uses Vite/browser runtime.
- **Our Corresponding Capability:**
  1. Closed-form timing functions (`easeOutExpo`, `easeInOutCubic`) + damped harmonic `springEval` directly in `studio/renderers/hyperframes_backend.py` so deterministic seeking evaluates motion curves directly from time `t` without frame-history state.
  2. Explicit `parallel` / `sequence` staggered animation groups (`m.LaggedStart` / `m.AnimationGroup`) in `AnimationEvent` & `scenes/v3_scene.py`.
- **Integration Approach:** Pattern & mathematical formulas reimplemented natively in JS/Python (`studio/renderers/hyperframes_backend.py`, `scenes/v3_scene.py`); Motion Canvas retained as an architectural reference rather than adding a Vite + Canvas2D Node bundler runtime.
- **Dependencies:** Pure math (`math` / JS `Math`), zero Vite/Node runtime dependency.
- **License Requirements:** MIT (`LICENSE`). Easing formulas match standard `easings.net` definitions.
- **Performance Implications:** Closed-form spring/easing evaluation runs in O(1) time per frame regardless of seek direction (random access / chunked rendering compatible).
- **Status:** **Reimplemented (Easing/Spring curves, Sequence/Parallel combinators); Deferred (Full Vite TS scene runtime)**

### 7. Mathematical Mobjects, Matching Transforms, Sample Space & ODE/Field Primitives
- **Source & Commit:** `3b1b/manim` (`fafa083a4fb274bba9cabde0b6e2f50ba6da0622`) and `3b1b/videos` (`31f8ffe785e860477a6051fde983c02c7d165bc2`)
- **Exact Source Files:**
  - `manim`: `manimlib/scene/scene.py`, `manimlib/animation/animation.py`, `manimlib/animation/transform_matching_parts.py`, `manimlib/mobject/coordinate_systems.py`, `functions.py`, `matrix.py`, `probability.py`, `number_line.py`, `vector_field.py`
  - `videos` (**educational reference only**): `_2015/pythagorean_proof.py`, `_2019/bayes/part1.py`, `_2023/clt/dice_sims.py`
- **Existing Upstream Behavior:**
  - `manim`: `TransformMatchingParts`/`Strings` decomposes TeX/VMobject subparts and matches shapes via `SequenceMatcher`; `SampleSpace` subdivides rectangles vertically and horizontally to represent prior & conditional probabilities physically by area; `Matrix` constructs grid-aligned brackets + elements; `vector_field.py` integrates ODE trajectories (`solve_ivp`) for streamlines. Note `manimgx` already provides native Vulkan GPU rendering for Manim mobjects (`MathTex`, `TransformMatchingShapes`, `Axes`, `VectorField`, etc.).
  - `videos`: Demonstrates pedagogical pacing — introducing one simple example before generalizing (`dice_sims.py`), semantic color-to-symbol consistency across formula & diagram (`part1.py`), and geometric construction on shared vertices (`pythagorean_proof.py`).
- **Our Corresponding Capability:**
  - Re-usable visual components in `scenes/v3_scene.py` + `studio/ai/v3_schema.py`: `matrix`, `table`, `graph_network`, `probability_region`, `waveform`, `particle_system`, `data_series`, `morph` / `duplicate` / `iterate_states` / `zoom`.
  - Generalized math constructions in `studio/ai/world_model.py`, `studio/ai/math_truth.py`, & `studio/registry/math_ops.py`: `discrete_pmf_convolution`, `riemann_integral`, `ode_phase_trajectory` (RK4), `linear_transform_2x2` & `eigen_decomposition_2x2`, `markov_chain_evolution`, `newton_root_path`, `taylor_approximation` & `fourier_general_synthesis`.
- **Integration Approach:** Mathematical algorithms & visual composition patterns cleanly implemented from scratch atop `manimgx` + NumPy/SymPy. **Zero code or assets copied from `3b1b/videos`** (per CC-BY-NC-SA-4.0 boundary).
- **Dependencies:** `manimgx`, `sympy`, `numpy` (already installed).
- **License Requirements:** `3b1b/manim` is MIT; `3b1b/videos` is CC-BY-NC-SA-4.0 (strictly non-copied educational study only).
- **Performance Implications:** Native ManimGX Vulkan shaders render mathematical vector geometry on GTX-class GPUs far faster than CPU Cairo (`3b1b/manim` PyOpenGL/ManimCE), while pure NumPy RK4/iteration computes trajectories offline prior to render.
- **Status:** **Adapted & Reimplemented (`manim` visual patterns atop `manimgx`); Educational Reference Only (`3b1b/videos`)**
