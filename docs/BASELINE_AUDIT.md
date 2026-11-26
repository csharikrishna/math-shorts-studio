# Baseline Audit — Math Shorts Studio (V1)

**Audited Repository:** `https://github.com/csharikrishna/math-shorts-studio` (branch `main`, commit `e5a1cde`)  
**Reference Video:** `output/AI_explain_simply_step_by_step_how_adding_smooth_si_shorts_hd.mp4`  
**Audit Date:** 2026-11-26

---

## 1. Repository Architecture & Execution Flow

The V1 pipeline (`studio/pipeline.py`) executes 5 sequential stages:
1. **`[0/5]` Gemini Creative Director (`studio/ai/director.py`)**:
   - Single-provider (`google-genai` only), lite-first fallback chain (`gemini-3.5-flash-lite` → `3.1-flash-lite` → `3.5-flash` → `3.8-flash`).
   - Runs Feasibility gate → `ShortPlan` JSON compiler → up to 3 static validation/repair rounds (`studio/ai/validate.py`).
2. **`[1/5]` TTS & Alignment (`studio/audio/tts.py`)**:
   - Synthesizes each beat independently via `kokoro` (`hexgrad/Kokoro-82M`, default `am_michael`) or `edge-tts`.
   - Trims leading/trailing silence in NumPy (`-45 dBFS` threshold) and aligns word timestamps (`align_words`) onto `ScriptToken`s, with character-position interpolation fallback.
3. **`[2/5]` Procedural Audio (`studio/audio/sfx.py`, `studio/audio/bgm.py`)**:
   - Generates stereo WAV SFX (`whoosh_soft`, `pop_marimba`, `tick_soft`, `chime_reveal`, `chord_epiphany`, `riser_soft`) and ambient pad + arpeggio BGM scaled to `timeline.total_duration + 3.0s`.
4. **`[3/5]`–`[4/5]` GPU Rendering & Layout QA (`scenes/ai_short.py`, `manimgx`)**:
   - Routes `ShortPlan.template` to `AIShort` (`graph`, `parametric`, `sequence`), `AIShort3D` (`surface`), or `AIShortPolytope` (`polytope4d`).
   - Renders via `manimgx` (Rust/wgpu Vulkan on `NVIDIA GeForce GTX 1650`) and checks `layout: no problems`.
5. **`[5/5]` FFmpeg Mastering (`studio/audio/master.py`)**:
   - Sidechain-ducks BGM under voiceover (`sidechaincompress`), widens stereo, applies `alimiter=limit=0.92`, and optionally applies YUV-native `eq` color grading encoded with `h264_nvenc` (or `libx264` fallback).

---

## 2. Reference Video Measurements (`AI_explain_simply_step_by_step_how_adding_smooth_si_shorts_hd.mp4`)

| Metric | Measured Value | Target / Standard | Status |
|---|---|---|---|
| **Duration** | `60.31 s` (9 beats) | `60 s` preset | ✅ Pass |
| **Resolution & FPS** | `1080 × 1920 @ 60 fps` | `1080 × 1920 @ 60 fps` | ✅ Pass |
| **Video Bitrate / Size** | `2135 kb/s`, `16.6 MB` (`h264 High`) | `< 25 MB` | ✅ Pass |
| **Audio Stream** | `48000 Hz`, stereo, `aac 163 kb/s` | `48000 Hz` stereo | ✅ Pass |
| **Integrated Loudness** | **`-24.0 LUFS`** | **`-16 to -14 LUFS`** | ❌ **Too quiet (−8 to −10 dB)** |
| **Loudness Range (LRA)** | `4.6 LU` | `4–8 LU` | ✅ Pass |
| **True Peak** | **`-4.3 dBFS`** | `≤ -1.0 dBTP` | ⚠️ Headroom unused |
| **Layout QA** | `no problems in 19 samples` | `no problems` | ✅ Pass |

---

## 3. Root-Cause Findings

### A. Audio & Synchronization
1. **Low Integrated Loudness (`-24.0 LUFS`)**: `studio/audio/master.py` mixes `sfx`, `voice_mix`, and `bgm_ducked` through `amix=normalize=0,alimiter=limit=0.92` with **no loudness normalization** (`loudnorm`). Social platforms normalize to `-14 LUFS`, leaving V1 ~10 dB quieter than peer educational videos.
2. **No Independent ASR Verification**: `tts.py` trusts `kokoro` token timestamps / `edge-tts` `WordBoundary` events without independent verification or pronunciation normalization for math symbols (`S_3`, `sin(3x)`, `f'(x)`).

### B. Visual Design & Hierarchy
1. **Permanent Static Header (`HeaderBanner`)**: `FOURIER ANALYSIS / Smooth Waves, Sharp Edges` remains locked at the top for all 60.3s, consuming ~23% of vertical real estate even when no longer needed.
2. **Distracting Live Decimal Counter (`StatBadge`)**: The sweeping tangent tracer + constantly ticking `f(x)` / `|ε|` readout draws eye focus away from the actual wave shape changes.
3. **Single-Template Lock**: `ShortPlan.template` forces the entire video into a single `graph` view. It cannot show component sine waves ($k=1, 3, 5$) separately before summing, nor transition to a frequency spectrum or real-world signal.
4. **Caption Font & Chunking**: While `Segoe UI` and `neon` dock improved readability over V1's serif `New Computer Modern`, `CaptionCard` still uses fixed 4-word chunks rather than semantic phrase segmentation, and has no WebVTT/transcript export.

### C. Mathematical & Educational Rigor
1. **Harmonic Order vs. Term Count**: Beats jump from `S_5` (`3 waves`) to `S_{15}` (`8 waves`) without explaining *why* only odd harmonics appear.
2. **Gibbs Phenomenon Precision**: Beat 7 states `Gibbs Phenomenon: ≈ 9%` without clarifying that it is **≈ 8.949% of the jump height** ($\approx 1.179$ overshoot on a $\pm 1$ square wave) and persists regardless of how many terms are added.
3. **No SymPy/SciPy Verification**: `studio/ai/validate.py` only checks AST syntax and numeric finiteness, not symbolic identities, derivatives, or approximation error bounds.

### D. Architecture & Extensibility
1. **Tightly Coupled Gemini**: `studio/ai/director.py` directly calls `google.genai.Client`, with no provider abstraction for OpenRouter (free models), Ollama, or `llama.cpp`.
2. **No Job Persistence or Incremental Scene Caching**: A crash at Beat 8 requires re-running the entire render; no SQLite job state or content-addressed scene cache exists.
3. **No Automated Tests**: Zero unit, integration, or regression tests in the repository.

---

## 4. Prioritized Improvement Plan
1. **Phase 1**: Fix two-pass `loudnorm` (`-14 LUFS`, `-1.0 dBTP`) in `master.py`, clean up visual clutter (optional/auto-hiding header, less distracting tracer), and add `pytest` baseline tests.
2. **Phase 2**: Introduce `studio/providers/` (`LLMProvider` interface, Gemini, OpenRouter free-only, Ollama, `llama.cpp`, strict free-only guard).
3. **Phase 3**: Math pronunciation normalizer, `TTSProvider` abstraction, unified `MasterTimeline`, WebVTT + transcript export.
4. **Phase 4–5**: Composable multi-scene `AnimationPlan`, knowledge-tree reasoning, long-form chaptered 16:9 pipeline, and long-to-short repurposing.
5. **Phase 6–8**: SQLite job persistence, content-addressed caching, SymPy + OpenCV QA, and V1-vs-V2 Fourier + generalization benchmarks.
