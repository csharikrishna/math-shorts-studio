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
