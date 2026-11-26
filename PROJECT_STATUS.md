# Project Status — AI Educational Video Studio V2.1

**Branch:** `main`  
**Last Updated:** 2026-11-26

## Component Status

| Component | Status | Notes / Verification |
|---|---|---|
| **V2/V2.1 Plan Routing & Auto-Director** | ✅ **Implemented and tested** | `create_v2_plan()` & schema-aware `run_plan()` tested end-to-end. |
| **Geometric Viewport Clipping (`plot_clipped_segments`)** | ✅ **Implemented and tested** | Replaced `np.clip()` with discontinuity/bound-aware segment splitting. |
| **Composable Bar Charts (`bar_chart`, `log_scale`)** | ✅ **Implemented and tested** | Honest log-scale `1/(2k)!` bars verified in Taylor V2.1 (`1`, `1/2!`, `1/4!`, `1/6!`, `1/8!`). |
| **Two-Pass `loudnorm` + `48000 Hz` CFR + Media QA** | ✅ **Implemented and tested** | Enforced `-ar 48000`, `-fps_mode cfr`, `I=-15.4 LUFS`, `TruePeak=-2.4 dBFS`. |
| **AST-Safe SymPy QA (`safe_sympy_from_expr`)** | ✅ **Implemented and tested** | Rejects arbitrary code; 15/15 unit & regression tests passing. |
| **Semantic Captions (`MAX_WORDS=6`)** | ✅ **Implemented and tested** | Wider phrase-aware chunks with `Segoe UI Bold` & `.srt`/`.vtt` export. |
| **Provider Abstraction (`Gemini`, `OpenRouter :free`, `Ollama`, `llama.cpp`)** | ⚠️ **Partially implemented** | Gemini tested live; OpenRouter/Ollama/llama.cpp unit-tested with mocks. |
| **Long-Form 16:9 & Scene-Level Caching** | ⏳ **Implemented but untested** | `KnowledgeTree`, `LongFormPlan`, `JobStore` exist; 5-min chaptered render pending. |

## Verified Rendered Benchmarks
- **Taylor `cos(x)` V2.1 (`V2_v2_taylor_series_approximation_of_cos_x_shorts_hd.mp4`):** `62.73s`, `1080×1920@60fps`, `5.2 MB`, `25.1s` GPU render, `layout: no problems in 16 samples`, **`-15.4 LUFS`**, **`-2.4 dBFS` TruePeak**, **`48000 Hz`**.
- **Fourier V2 (`V2_fourier_v2_benchmark_shorts_hd.mp4`):** `67.62s`, `1080×1920@60fps`, `5.9 MB`, `37.5s` GPU render, `-14.2 LUFS`.
