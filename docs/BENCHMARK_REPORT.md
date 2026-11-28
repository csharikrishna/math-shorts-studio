# Cross-Backend & Long-Form Benchmark Report

All measurements below were collected empirically on local dev hardware using zero mandatory paid APIs or proprietary runtimes.

## 1. End-to-End Autonomous Unseen Long-Form Documentary

- **Topic:** `"Eigenvalues, Phase Space, and Dynamical Stability"` (genuinely unseen topic outside `plans/`)
- **Pipeline:** `create_longform_plan(target_duration_s=540.0)` -> `build_longform_video(format_name="widescreen_hd", backend="manimgx", tts_engine="kokoro")`
- **Final Output:** `output/LF_eigenvalues_phase_space_and_dynamical_stability_widescreen_hd.mp4` (`42,546,988 bytes`)
- **Duration & Chapters:** **688.62 s (`11m 28.62s`)** across **7 substantive conceptual chapters** (`CH1` 105.6s, `CH2` 90.3s, `CH3` 102.5s, `CH4` 90.5s, `CH5` 92.2s, `CH6` 85.3s, `CH7` 122.2s; `38` total scenes across `10` distinct `ObjectKind`s and `5` compositions)
- **Visual & Media QA:**
  - Every chapter (`CH1..CH7`) passed ManimGX strict layout QA with **`0` layout problems**
  - Integrated loudness: **`-15.2 LUFS`**, True Peak: **`-2.2 dBFS`** (`passed`)
  - Streams: **`1920x1080 @ 60.00 fps`** (`h264_nvenc`), **`48000 Hz stereo AAC`**, **`v_dur=688.57s` / `a_dur=688.62s` (`drift=0.05s`)** (`passed`)
  - Claim verification coverage (`REVIEW.md`): **`deterministically_verified=18`**, **`numerically_tested=11`**, **`failed=0`**, **`unsupported=0`**, **`Overall Score: 100.0/100`** (`Approved: True`)

## 2. Recomposed Native 9:16 Short Derivation

- **Derived Output:** `output/SHORT_eigenvalues_phase_space_and_dynamical_stability_shorts_hd.mp4` (`3,983,993 bytes`) via `derive_recomposed_short()`
- **Source Chapter:** `CH1` (word-budget trimmed to core arc and natively adapted via `adapt_composition_for_aspect(..., "9:16")`)
- **Measured QA:**
  - Duration: **`58.40 s`** (`v_dur=58.35s`, `a_dur=58.40s`, `drift=0.05s`)
  - Layout QA: **`layout: no problems in 6 samples`**
  - Loudness: **`-15.3 LUFS`**, True Peak: **`-2.3 dBFS`** (`passed`)
  - Stream: **`1080x1920 @ 60.00 fps`**, **`48000 Hz stereo AAC`**
  - Review Score (`REVIEW.md`): **`100.0/100`** (`blockers=0`, `warnings=0`, `deterministically_verified=3`, `numerically_tested=1`)

## 3. ManimGX vs. HyperFrames Comparison Benchmark

Both backends rendered the exact same recomposed 3-beat `shorts_hd` (`1080x1920 @ 60 fps`) `VideoPlanV3` JSON using local Kokoro-82M (`rate="-4%"`) and NVENC mastering:

| Metric | `ManimGXBackend` (wgpu GPU) | `HyperFramesBackend` (Offline Chrome CDP) |
| --- | --- | --- |
| Output Duration | `58.4 s` (`v_dur=58.35s`, `a_dur=58.40s`) | `58.4 s` (`v_dur=58.35s`, `a_dur=58.40s`) |
| Wall-Clock Total Time | **`84.506 s`** (`25.0 s` ManimGX GPU phase) | `266.696 s` (`3500` deterministic CDP seek/JPEG frames) |
| Master File Size | `3,929,407` bytes (`3.7 MB`) | `3,479,919` bytes (`3.3 MB`) |
| Process Peak RSS (parent orchestrator w/ Kokoro loaded) | `1798.98 MB` | `1798.98 MB` |
| Audio Sample Rate & Loudness | `48000 Hz` / `-15.3 LUFS` (`-2.7 dBFS`) | `48000 Hz` / `-15.2 LUFS` (`-2.8 dBFS`) |
| Layout Audit Issues | **`[]`** (`no problems in 6 samples`) | **`[]`** (`__HF_AUDIT_LAYOUT` clean) |
| Best Architectural Fit | Dense geometric morphs, LaTeX math, custom Vulkan curves & vector fields (~3.2x faster wall time) | Web/UI overlays, HTML tables, CSS/SVG typography, DOM layout audit |

## 4. Render-Time Optimization Pass (Verified Before vs. After on Same 58.4s `shorts_hd` Short)

Implemented pipeline, encoder, CDP capture, and TTS/cache optimizations while maintaining identical output QA (`48000 Hz` stereo AAC, `-15.3` to `-15.8 LUFS`, `<= -2.5 dBFS` peak, `0.00 s` A/V drift, zero layout defects):

- **ManimGX intermediate preset & NVENC probe caching (`studio/pipeline.py`, `studio/audio/master.py`):** Intermediate `raw_mp4` now encodes with `--preset ultrafast --crf 12` before final NVENC (`-preset p2`) mastering, NVENC availability probe is cached (`_probe_encoder`), loudnorm Pass 1 and `verify_audio_loudness` use `-vn` (skipping video decode), and `widescreen_fast` (`1280x720 @ 30fps`) was added to `studio/config.py`.
- **HyperFrames static-frame skip + single-pass direct NVENC (`studio/renderers/hyperframes_backend.py`):** Combined `window.__HF_STEP(t, doAudit)` skips unchanged DOM/SVG frames (`203/3498` unique CDP captures instead of `3498`) and pipes unique JPEGs + voice/BGM directly into single-pass `h264_nvenc` mastering without an intermediate `libx264` file.
- **Lazy Kokoro probe, in-memory resampling, shared caches & long-form concurrency (`studio/audio/tts.py`, `studio/audio/sfx.py`, `studio/longform.py`):** `kokoro_available()` uses `importlib.util.find_spec("kokoro")` (eliminating a `12.85 s` PyTorch cold import on cache hits), Kokoro resamples `24 -> 48 kHz` and trims silence in memory (zero per-beat FFmpeg spawns), shared cross-backend `voice_segments`/`sfx` caches omit unused `pitch` from Kokoro keys, `edge-tts` runs concurrently, and `longform.py` scripts chapters + renders HyperFrames chapters in bounded thread pools.

| Benchmark Configuration (`58.4s` `shorts_hd`) | Before | After | Speedup | QA Status |
| --- | --- | --- | --- | --- |
| `ManimGX` intermediate GPU render phase | `25.0 s` (`22.03 s` medium) | **`10.4 – 10.7 s`** (`ultrafast`) | **~2.4x** | `no problems in 6 samples` |
| `ManimGX` cold end-to-end (incl. Kokoro CPU) | `84.506 s` | **`67.928 s`** | **1.24x** | `-15.3 LUFS`, `drift=0.00s` |
| `ManimGX` warm shared TTS cache (cross-backend) | `84.506 s` | **`25.312 s`** | **3.34x** | `-15.3 LUFS`, `drift=0.00s` |
| `HyperFrames` cold end-to-end (incl. Kokoro CPU) | `266.696 s` | **`67.098 – 78.086 s`** | **~3.4x – 4.0x** | `-15.8 LUFS`, `drift=0.00s` |
| `HyperFrames` warm shared TTS cache | `266.696 s` | **`26.680 s`** | **10.0x** | `-15.8 LUFS`, `drift=0.00s` |

## 5. V3.7 Production Verification, Light/Dark & Unseen Documentary, Hybrid Mux & Warm-Cache Switch Benchmarks

Measured after the V3.7 upgrades (pinned offline OFL fonts, WCAG-validated unified `light`/`dark` looks, `CaptionIR`, adaptive layout solver, `CreativeDirector` locked script, `HybridRenderBackend`, honest 10-category QA, and `ArtifactCache` dependency integrity):

| Deliverable / Configuration | Output / Scope | Duration & Target | Media & Loudness QA | Decoded-Frame / Review QA |
| --- | --- | --- | --- | --- |
| **Eigenvalues Documentary (`light`, `editorial`, `manimgx`)** | `output/LF_eigenvalues_phase_space_and_dynamical_stability_widescreen_hd.mp4` (`7` chapters, `26` scenes) | `585.42 s` vs `540.0 s` (`+8.4%`) | `1920x1080 @ 60.00 fps`, `48000 Hz`, `-15.3 LUFS`, `-1.8 dBFS`, `drift=0.00 s` | `layout: no problems` (`CH1..CH7`), `edge_clips=0`, `median_stage_w=40.5%`, `REVIEW.md`: **`97.5/100`** (`approved=True`, `det_verified=23`, `num_tested=18`, `unsupported=1`, `failed=0`) |
| **Derived Native Portrait Short (`light`, `kinetic`, `manimgx`)** | `output/SHORT_eigenvalues_phase_space_and_dynamical_stability_shorts_hd.mp4` (`3` scenes) | `55.00 s` vs `55.0 s` (`0.0%`) | `1080x1920 @ 60.00 fps`, `48000 Hz`, `-15.3 LUFS`, `-1.2 dBFS`, `drift=0.00 s` | `layout: no problems in 8 samples`, `edge_clips=0`, `median_stage_w=76.5%`, `REVIEW.md`: **`94.0/100`** (`approved=True`) |
| **Unseen Documentary: Markov Chains (`dark`, `editorial`, `manimgx`)** | `output/LF_markov_chains_random_walks_and_stationary_equili_widescreen_hd.mp4` (`6` chapters, `22` scenes, `12` `ObjectKind`s) | `490.92 s` vs `510.0 s` (`-3.7%`) | `1920x1080 @ 60.00 fps`, `48000 Hz`, `-15.3 LUFS`, `-1.7 dBFS`, `drift=0.00 s` | `layout: no problems` (`c1..c6`), `edge_clips=0`, `median_stage_w=61.4%`, `REVIEW.md`: **`90.0/100`** (`approved=True`, `det_verified=9`, `unsupported=11`, `failed=0`) |
| **Scene-Level `HybridRenderBackend` Mux Smoke (`hyb_smoke`)** | `manimgx(1) -> hyperframes(1)` onto single `AudioTimeline` (`1280x720 @ 30 fps`) | `7.60 s` | `30.00 fps`, `48000 Hz`, `-15.4 LUFS`, `-2.9 dBFS`, `drift=0.00 s` | `FRAMES: passed` (`edge_clips=0`, `finds=[]`) |

### Warm-Cache Theme & Caption Switch Experiment (`eig_sample`, `16.20 s` `1920x1080 @ 60 fps`)
- **Pass 1 (`--theme light --caption-style editorial`, cold Kokoro TTS + GPU + NVENC):** **`25.524 s`** (`-15.4 LUFS`, `-2.5 dBFS`, `drift=0.00 s`, `layout: no problems`).
- **Pass 2 (`--theme dark --caption-style minimal`, warm Kokoro voice-segment reuse + GPU + NVENC):** **`6.864 s`** (**`3.72x` speedup**; `-15.4 LUFS`, `-2.5 dBFS`, `drift=0.00 s`, `layout: no problems`).
- **Dependency Cache Verification (`ArtifactCache`):** `tts` stage lookup remained `(True, "hit")`, while `captions` / `chapter_assembly` keys cleanly returned `(False, "miss:dependency_or_payload_changed")` on theme/caption mutation.

---

## 6. V3.8 Unseen-Topic Documentary, Portrait Short, Cross-Theme/Backend Samples & Scene Preview Benchmarks

Measured after V3.8 implementation on local Windows hardware (**NVIDIA GeForce GTX 1650 `4096 MiB` (`h264_nvenc`), Local Kokoro-82M CPU TTS, free Gemini API**) on an unseen topic with zero hardcoded topic logic (`"Fourier Analysis, Spectral Decomposition, and Signal Filtering"`):

| Deliverable / Configuration | Output Artifact | Duration & Pacing | Media, Loudness & Silence QA | Scene-Aware Frame & Honest QA Review |
| --- | --- | --- | --- | --- |
| **Unseen Documentary (`light`, `editorial`, `manimgx`)** | `output/LF_fourier_analysis_spectral_decomposition_and_sign_widescreen_hd.mp4` (`18.55 MB`, `8` chapters, `53` scenes, `16` object kinds, `7` compositions) | `460.52 s` (`7m 40.5s` vs `500.0 s`, `-7.9%`) | `1920x1080 @ 60.00 fps`, `48000 Hz`, `-15.2 LUFS`, `-1.5 dBFS`, `drift=0.00 s`, `0` silences `>1.2 s` | `24` scene-aware frames: `median_stage_w=68.8%`, `edge_clips=0`, `empty_mids=0`; `REVIEW.md`: **`publication_approved` (`100.0/100`)**, `det_verified=52`, `source_supported=1`, `unsupported=0`, `failed=0`, `cues=12` (`tts_word_boundary=86`) |
| **Derived Native `9:16` Portrait Short (`light`, `kinetic`, `manimgx`)** | `output/SHORT_fourier_analysis_spectral_decomposition_and_sign_shorts_hd.mp4` (`2.3 MB`, `5` scenes from climax `c3` via `SceneRuntimeState`) | `41.90 s` (`-19.0%` vs `51.7 s`) | `1080x1920 @ 60.00 fps`, `48000 Hz`, `-15.1 LUFS`, `-2.8 dBFS`, `drift=0.00 s` | `17` scene-aware frames: `median_stage_w=72.2%`, `edge_clips=0`, `empty_mids=0`; `REVIEW.md`: **`publication_approved` (`94.0/100`)**, `det_verified=5`, `unsupported=0`, `cues=19` (`tts_word_boundary=106`) |
| **Dual-Theme `HyperFrames` Sample (`midnight_studio`, `editorial`, `c3`)** | `output/SAMPLE_fourier_analysis_spectral_decomposition_and_sign_midnight_hf_widescreen_hd.mp4` (`3.09 MB`) | `73.10 s` | `1920x1080 @ 60.00 fps`, `48000 Hz`, `-15.3 LUFS`, `-2.8 dBFS`, `drift=0.00 s`, `0` silences `>1.2 s` | `frames=11`, `median_stage_w=45.5%`, `edge_clips=0`, `empty_mids=0` (`passed`) |
| **Dual-Backend `Hybrid` Sample (`light`, `editorial`, `c2`)** | `output/SAMPLE_fourier_analysis_spectral_decomposition_and_sign_light_hybrid_widescreen_hd.mp4` (`2.16 MB`) | `53.30 s` | `1920x1080 @ 60.00 fps`, `48000 Hz`, `-15.3 LUFS`, `-2.4 dBFS`, `drift=0.00 s`, `0` silences `>1.2 s` | `frames=9`, `median_stage_w=36.8%`, `edge_clips=0`, `empty_mids=0` (`passed`) |
| **Isolated Single-Scene Preview (`studio/preview.py`, `midnight_studio`, `hyperframes`)** | `preview_single/scene_2_widescreen_hd_midnight_studio_editorial.mp4` + `preview_workbench/index.html` | Rendered in **`10.19 s`** (`3` carried `SceneRuntimeState` objects) | `1920x1080 @ 60.00 fps` | `REVIEW.md`: **`publication_approved` (`100.0/100`)** |

### Strict Publication-Gate Verification on V3.7 Markov Chains Baseline
- Re-auditing the retained V3.7 `Markov Chains` plan (`c1..c6`) under V3.8 `audit_educational_quality()` honestly downgraded its stale V3.7 approval (`90.0/100, approved=True`) to **`review_required` (`13.0/100, approved=False`)** because `unsupported=10` and `human_review_required=5`.

---

## 7. V3.9 Cross-Renderer Linked Benchmark Suite & Held-Out Unseen Topic Evaluation

Measured after V3.9 implementation on local Windows hardware (**NVIDIA GeForce GTX 1650, `h264_nvenc`, Local Kokoro-82M CPU TTS, offline Chromium CDP + vendored KaTeX `0.19.1`, free Gemini `gemini-3.5-flash-lite`**).

### A. 5-Target Fourier Linked-Visualization Benchmark (`SignalSpectralPipeline`)

All 5 targets share the identical approved `ScriptIR`, `LinkedMathModelSpec(kind="signal_spectral_pipeline", expr="0.85*cos(2*pi*2*t) + 0.55*cos(2*pi*5*t)")`, trapezoidal complex winding integral $z(t)=x(t)e^{-i\omega t}$, consistent derived magnitude spectrum & low-pass `rfft`/`irfft` filtering, and locally synthesized sound demonstration (`studio/audio/demo_signals.py`):

| Benchmark Target | Backend / Theme / Caption | Duration & Size | Render Time / Peak RSS | Media, Loudness & QA Status | Manifest Verified |
| --- | --- | --- | --- | --- | --- |
| `v39_bench_light_manimgx_widescreen_hd.mp4` | `manimgx` / `light` / `editorial` | `46.90 s` (`3,261,178 B`) | `39.16 s` / `282.4 MB` | `60.00 fps`, `48000 Hz`, `-15.2 LUFS`, `-2.7 dBFS`, `drift=0.00s`; **`publication_approved` (`94.0/100`)** | `True` |
| `v39_bench_dark_manimgx_widescreen_hd.mp4` | `manimgx` / `midnight_studio` / `editorial` | `46.90 s` (`3,141,724 B`) | `36.40 s` / `1025.96 MB` | `60.00 fps`, `48000 Hz`, `-15.2 LUFS`, `-2.7 dBFS`, `drift=0.00s`, `layout: no problems in 14 samples`; **`publication_approved` (`94.0/100`)** | `True` |
| `v39_bench_hf_katex_widescreen_hd.mp4` | `hyperframes` / `editorial_light` / `editorial` | `42.70 s` (`3,045,974 B`) | `40.78 s` / `1030.02 MB` | `60.00 fps`, `48000 Hz`, `-15.6 LUFS`, `-2.9 dBFS`, `drift=0.00s`, offline KaTeX `0.19.1`; **`publication_approved` (`94.0/100`)** | `True` |
| `v39_bench_hybrid_widescreen_hd.mp4` | `hybrid` / `midnight_studio` / `editorial` | `42.50 s` (`2,546,052 B`) | `50.97 s` / `1032.01 MB` | `60.00 fps`, `48000 Hz`, `-15.2 LUFS`, `-2.8 dBFS`, `drift=0.00s`; **`publication_approved` (`94.0/100`)** | `True` |
| `v39_bench_portrait_shorts_hd.mp4` | `manimgx` / `light` / `kinetic` (`9:16`) | `46.90 s` (`4,043,909 B`) | `41.48 s` / `1032.01 MB` | `1080x1920 @ 60.00 fps`, `48000 Hz`, `-15.2 LUFS`, `-2.7 dBFS`, `drift=0.00s`; **`publication_approved` (`94.0/100`)** | `True` |

### B. Frozen-Code Held-Out Unseen Topic Documentary & Portrait Short

Topic generated autonomously **without any topic-specific core code**: `"Damped Harmonic Oscillators, Resonance Curves, and Phase Portraits"`.

- **Long-Form Documentary (`hybrid`, `editorial_light`, `editorial`):** `output/LF_damped_harmonic_oscillators_resonance_curves_and_widescreen_hd.mp4` (`10,165,835 B`, `225.62 s` vs `240.0 s` target, `7` chapters, `28` scenes, `10` object kinds, `6` compositions, `0` layout errors across ManimGX chapters, `124` decoded QA frames across `28/28` scenes, `median_stage_w=68.8%`, `edge_clips=0`, `empty_mids=0`, `1920x1080 @ 60.00 fps`, `-15.2 LUFS`, `-2.2 dBFS`, `drift=0.00 s`, merged `CaptionIR`: `79` cues / `517` words, `deterministically_verified=23`, `numerically_tested=19`, `unsupported=0`, `manifest_ok=True`).
  - **Honest QA Result (`REVIEW.md`):** **`54.0/100` (`production_state="rejected"`, `approved=False`)** — honest QA caught a trailing punctuation token timing inversion in `c2__cue_0003` (`('.', 31.633, 31.253)`) and empty `motion_ops` warning on Gemini-compiled chapters instead of rubber-stamping `100/100`.
- **Derived Native `9:16` Portrait Short (`manimgx`, `light`, `kinetic`):** `output/SHORT_damped_harmonic_oscillators_resonance_curves_and_shorts_hd.mp4` (`2,738,573 B`, `44.00 s`, `1080x1920 @ 60.00 fps`, `-15.1 LUFS`, `-2.5 dBFS`, `drift=0.00 s`, `17` QA frames covering `4/4` scenes, `median_stage_w=72.2%`, `cues=21`, `tts_word_boundary=102`, `manifest_ok=True`).
  - **Honest QA Result (`REVIEW.md`):** **`78.0/100` (`production_state="review_required"`, `approved=False`)** — flagged `motion_ops=[]` and `-12.2%` pacing deviation cleanly.

### C. Single-Prompt & Hybrid Throughput Optimizations (`V3.9 Perf Pass`)

- **Gemini Schema & Prompt Deduplication (`studio/providers/gemini.py`, `studio/ai/director.py`, `studio/longform.py`):** Pre-cached JSON-mode-only schemas (`_JSON_MODE_ONLY_SCHEMAS` + `_COMPACT_SCHEMA_CACHE`) so `VideoPlanV3` calls skip doomed constrained-decoding roundtrips (`400 too many states`), switched `429` errors immediately to fallback models without sleeping `4 s`, and removed duplicate `system_prompt` / `ch_system` prefixes from user messages.
- **Hybrid `video_only` Segments & GPU NVENC Concat (`studio/renderers/hyperframes/assembly.py`, `studio/renderers/hyperframes/capture.py`, `studio/renderers/hybrid.py`, `studio/pipeline.py`, `studio/longform.py`):** Added `video_only=True` segment rendering (skipping discarded SFX/BGM/loudnorm Pass 1 + manifest writes), loopback Chrome readiness polling (`0.02s` steps), `-an` + `h264_nvenc` concat, thread-safe `_V3_SCENE_ENV_LOCK` around scene imports, and 2-worker `hybrid` chapter parallelism (`STUDIO_HYB_WORKERS`).
- **Measured Hybrid Benchmark Gain (`46.90 s` 1080p60 Fourier linked render):** Dropped from **`50.97 s`** (`1.20x` real-time on `42.50 s` video) to **`38.81 s`** (`0.83x` real-time on `46.90 s` video — **`~1.44x` faster per video second**; `loud=passed`, `stream=passed`, `manifest=True`).

---

## 8. V4.0 Simulation-Driven Educational Filmmaking & Held-Out Verification

Measured after V4.0 implementation on local Windows hardware (**NVIDIA GeForce GTX 1650 `4096 MiB` (`h264_nvenc`), Local Kokoro-82M CPU TTS, offline Chromium CDP + vendored KaTeX `0.19.1`, free Gemini API**). All final artifacts are manifest-verified (`identity_verified=True`, `stream_and_assets_complete=True`).

### A. 5-Target Fourier V4.0 Linked-Visualization Re-Benchmark (`s3_centroid` 2.0 Hz Sync & Endpoint-Exclusive DFT Peaks)

| Output Artifact | Backend / Theme / Caption | Duration & Size | Wall-Clock / Peak RSS | QA Outcome & Manifest Certification |
| --- | --- | --- | --- | --- |
| `output/V40_fourier_hf_light_widescreen_hd.mp4` | `hyperframes` / `editorial_light` / `editorial` | `41.90 s` (`2,335,268 B`) | `42.002 s` / `1311.66 MB` | **`100.0/100`** (`publication_approved`, `certified=True`) |
| `output/V40_fourier_hf_dark_widescreen_hd.mp4` | `hyperframes` / `midnight_studio` / `editorial` | `41.90 s` (`2,432,579 B`) | `36.976 s` (`36.969 s` warm) / `1317.09 MB` | **`100.0/100`** (`publication_approved`, `certified=True`) |
| `output/V40_fourier_mgx_dark_widescreen_hd.mp4` | `manimgx` / `midnight_studio` / `editorial` | `46.60 s` (`2,305,214 B`) | `33.573 s` / `1321.43 MB` | **`100.0/100`** (`publication_approved`, `certified=True`) |
| `output/V40_fourier_hyb_dark_widescreen_hd.mp4` | `hybrid` / `midnight_studio` / `editorial` | `41.90 s` (`2,237,635 B`) | `76.016 s` / `1315.90 MB` | **`100.0/100`** (`publication_approved`, `certified=True`) |
| `output/V40_fourier_short_portrait_shorts_hd.mp4` | `manimgx` / `midnight_studio` / `kinetic` (`9:16`) | `46.60 s` (`3,224,210 B`) | `35.530 s` / `1323.95 MB` | **`100.0/100`** (`publication_approved`, `certified=True`) |

### B. Regenerated Harmonic Oscillator Documentary & Derived Portrait Short (Simulation Engine + Scientific Components)

Topic: `"Damped Harmonic Oscillators, Resonance Curves, and Phase Portraits"` (`editorial_light`, `7` chapters, `16` scenes, zero topic-specific code).

| Deliverable | Configuration & Output | Duration & Size | Cold vs. Warm Wall Time (Speedup) | Media, Motion & Honest QA Status |
| --- | --- | --- | --- | --- |
| **Derived Portrait Short** | `manimgx` / `editorial_light` / `kinetic` (`SHORT_damped_harmonic_oscillators_resonance_curves_and_shorts_hd.mp4`) | `40.00 s` (`2,325,325 B`) | `21.326 s` | `1080x1920 @ 60 fps`, `edge_clips=0`, `empty_mids=0`; **`100.0/100`** (`publication_approved`, `certified=True`) |
| **Full Documentary** | `hybrid` / `editorial_light` / `editorial` (`LF_damped_harmonic_oscillators_resonance_curves_and_widescreen_hd.mp4`) | `273.52 s` (`9,853,627 B`, `7` ch, `16` sc) | `369.572 s` cold -> **`27.155 s`** warm (**`13.6x`**) / `4272.87 MB` peak RSS | `-15.1 LUFS`, `-2.2 dBFS`, `drift=0.00 s`, `585` TTS words (`94` cues), `animated_scenes=12/16`, `det_verified=14`, `num_tested=15`, `unsupported=0`, `failed=0`, `edge_clips=0`, `empty_mids=0`; **`72.0/100`** (`review_required` — `3` warnings: `c3__scene_2` low pixel delta + `+14.0%` duration vs `240s` target; improved from V3.9 `54.0/100 rejected`) |

### C. Held-Out Unseen Nonlinear ODE Topic (Lotka-Volterra Predator-Prey) on Frozen V4.0 Core

Topic generated completely autonomously with zero core edits for the topic: `"Lotka-Volterra Predator-Prey Oscillations and Phase Orbits"` (`midnight_studio`, `5` chapters, `14` scenes, `target_duration_s=180.0`, `provider_requests=15`, `tokens=50,150`).

| Deliverable | Configuration & Output | Duration & Size | Cold vs. Warm Wall Time (Speedup) | Media, Motion & Honest QA Status |
| --- | --- | --- | --- | --- |
| **Derived Portrait Short** | `manimgx` / `midnight_studio` / `kinetic` (`SHORT_lotka_volterra_predator_prey_oscillations_and_ph_shorts_hd.mp4`) | `37.40 s` (`2,071,459 B`, `3` sc) | `50.203 s` (`4.903 s` QA) / `1924.12 MB` | `1080x1920 @ 60 fps`, `-15.1 LUFS`, `-2.5 dBFS`, `drift=0.00 s`, `15` frames (`edge_clips=0`, `empty_mids=0`), `84` TTS words (`15` cues), `det_verified=3`, `num_tested=2`; **`94.0/100`** (`publication_approved`, `certified=True`) |
| **Full Documentary** | `hybrid` / `midnight_studio` / `editorial` (`LF_lotka_volterra_predator_prey_oscillations_and_ph_widescreen_hd.mp4`) | `185.32 s` vs `180.0 s` (`+2.9%`, `6,571,762 B`, `14` sc) | `269.622 s` cold (`108.27s` TTS, `134.91s` render, `6.30s` asm, `19.77s` QA) -> **`26.155 s`** warm (**`10.3x`**) / `4349.17 MB` | `1920x1080 @ 60 fps`, `-15.1 LUFS`, `-1.9 dBFS`, `drift=0.00 s`, `49` frames (`14/14` scenes, `edge_clips=0`, `empty_mids=0`, `median_stage_w=70.1%`), `411` TTS words (`61` cues), `animated_scenes=10/14`, `det_verified=10`, `num_tested=9`, `unsupported=0`, `failed=0`; **`78.0/100`** (`review_required` — `c3__scene_2` low pixel delta warning) |

---

## 9. V4.1 Visual Intelligence, Linked Compositions & Reliability Benchmarks

All V4.1 full documentaries and derived portrait Shorts below were built on clean commit `77709f3865e48c631d02705a3f8c2abb328abe3c` (`git_dirty=False`, `qa_report_sha256` verified) on the same local Windows workstation (`h264_nvenc`, Kokoro-82M CPU TTS, offline Chromium CDP + vendored KaTeX `0.19.1`, `production_certified=True`).

### A. Representative Multi-Component Light/Dark Experiments (`build/v41_exp1/`)

| Artifact | Backend / Theme / Duration | File Size | Loudness & Sync | QA Outcome & Visual Inspection |
| --- | --- | --- | --- | --- |
| `osc_hf_light.mp4` | `hyperframes` / `light` / `14.90 s` | `935,860 B` | `-15.7 LUFS`, `-2.8 dBFS`, `drift=0.00 s` | **`100.0/100`** (`publication_approved`, `0` blockers, `0` warnings, `edge_clips=0`, `empty_mids=0`, `overlaps=0`, `data_clips=0`) |
| `osc_mgx_dark.mp4` | `manimgx` / `midnight_studio` / `14.90 s` | `829,335 B` | `-15.2 LUFS`, `-2.2 dBFS`, `drift=0.00 s` | **`100.0/100`** (`publication_approved`, `0` blockers, `0` warnings) |
| `lv_hf_dark.mp4` | `hyperframes` / `midnight_studio` / `18.50 s` | `956,218 B` | `-15.6 LUFS`, `-2.8 dBFS`, `drift=0.00 s` | **`94.0/100`** (`publication_approved`, closed orbit + vector field + flat `H(x,y)` invariant verified visually) |
| `lv_mgx_light.mp4` | `manimgx` / `light` / `18.60 s` | `1,041,370 B` | `-15.1 LUFS`, `-2.8 dBFS`, `drift=0.00 s` | **`94.0/100`** (`publication_approved`, `layout: no problems in 4 samples`) |

### B. V4.1 Regenerated Harmonic Oscillator & Lotka–Volterra Documentaries + Shorts (Before vs. After vs. V4.0 Baseline)

| Deliverable | Configuration & Output | Duration & Size | Wall Time (Cold -> Warm) | Content-Aware QA & Verified Improvement |
| --- | --- | --- | --- | --- |
| **Harmonic Oscillator Documentary** | `hybrid` / `editorial_light` / `editorial` (`LF_damped_harmonic_oscillators_resonance_curves_and_widescreen_hd.mp4`) | `272.12 s` (`9,924,566 B`, `7` ch, `16` sc) | `113.008 s` | `1920x1080 @ 60.00 fps`, `48000 Hz`, `-15.1 LUFS`, `-2.2 dBFS`, `drift=0.00 s`, `51` frames (`16/16` sc), `edge_clips=0`, `empty_mids=0`, `overlaps=0`, `data_clips=0`, `94` cues (`585` words), `det_verified=14`, `num_tested=15`; **`88.0/100`** (`publication_approved`, up from V4.0 **`72.0/100`**; honest `+13.4%` duration advisory & `2` motion warnings retained) |
| **Harmonic Oscillator Short** | `manimgx` / `editorial_light` / `editorial` (`SHORT_damped_harmonic_oscillators_resonance_curves_and_shorts_hd.mp4`) | `40.00 s` (`1,650,975 B`) | `15.553 s` | `1080x1920 @ 60.00 fps`, `48000 Hz`, `-15.2 LUFS`, `-2.5 dBFS`, `drift=0.00 s`; **`100.0/100`** (`publication_approved`, `certified=True`) |
| **Lotka–Volterra Documentary** | `hybrid` / `midnight_studio` / `editorial` (`LF_lotka_volterra_predator_prey_oscillations_and_ph_widescreen_hd.mp4`) | `185.32 s` (`6,818,663 B`, `5` ch, `14` sc) | `69.801 s` cold -> **`20.257 s`** warm (**`3.45x`**) | `1920x1080 @ 60.00 fps`, `48000 Hz`, `-15.1 LUFS`, `-1.9 dBFS`, `drift=0.00 s`, `49` frames (`14/14` sc), `edge_clips=0`, `empty_mids=0`, `overlaps=0`, `data_clips=0`, `dynamic_gaps=0/8`, `61` cues (`411` words), `det_verified=10`, `num_tested=9`; **`100.0/100`** (`publication_approved`, `0` blockers, `0` warnings, up from V4.0 **`78.0/100`**; fixed `[15,5]` empty-orbit & `[83s]` panel-overlap failures visually verified on contact sheet) |
| **Lotka–Volterra Short** | `manimgx` / `midnight_studio` / `editorial` (`SHORT_lotka_volterra_predator_prey_oscillations_and_ph_shorts_hd.mp4`) | `37.40 s` (`1,534,156 B`) | `17.764 s` | `1080x1920 @ 60.00 fps`, `48000 Hz`, `-15.2 LUFS`, `-2.5 dBFS`, `drift=0.00 s`; **`94.0/100`** (`publication_approved`, `certified=True`) |
