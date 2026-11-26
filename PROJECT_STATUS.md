# Project Status — AI Educational Video Studio V2

**Branch:** `main`  
**Last Updated:** 2026-11-26

## Component Status

| Component | Status | Notes / Verification |
|---|---|---|
| **V2 Plan Routing (`run_plan`, `--engine v2`)** | ✅ **Implemented and tested** | Auto-routes `VideoPlan` V2 to `MultiSceneVideo` and legacy `ShortPlan` to `AIShort`; tested end-to-end. |
| **Automatic V2 Director (`create_v2_plan`)** | ✅ **Implemented and tested** | Generated & rendered `v2_taylor_series_approximation_of_cos_x.json` via Gemini (`3.9s`, `508+1216` tokens). |
| **Two-Pass `loudnorm` + TruePeak QA** | ✅ **Implemented and tested** | `I=-15.5 LUFS`, `TruePeak=-2.1 dBFS <= -1.0` verified on `V2_v2_taylor_series_approximation_of_cos_x_shorts_hd.mp4`. |
| **Multi-Scene Renderer (`MultiSceneVideo`)** | ✅ **Implemented and tested** | Distinct `Create` / `LaggedStart(GrowFromEdge)` / Gibbs `1.17898` line; `layout: no problems`. |
| **AST-Safe SymPy QA (`safe_sympy_from_expr`)** | ✅ **Implemented and tested** | Rejects arbitrary code; 13/13 unit tests passing. |
| **Provider Abstraction (`Gemini`, `OpenRouter :free`, `Ollama`, `llama.cpp`)** | ⚠️ **Partially implemented** | Gemini tested live; OpenRouter/Ollama/llama.cpp unit-tested with mocks. |
| **Pronunciation Normalizer (`pronounce.py`)** | ✅ **Implemented and tested** | Wired into `tts.py` before Kokoro/Edge synthesis. |
| **Long-Form 16:9 & Scene-Level Caching** | ⏳ **Implemented but untested** | `KnowledgeTree`, `LongFormPlan`, `JobStore` schemas/store exist; full 5-min chaptered render pending. |
| **Semantic Phrase Subtitles (replacing 4-word chunks)** | ⏳ **Pending** | `CaptionCard` still uses 4-word phrase-aware chunks; `.srt`/`.vtt` use full narration. |

## Verified Rendered Benchmarks
- **Fourier V2 (`V2_fourier_v2_benchmark_shorts_hd.mp4`):** `67.62s`, `1080×1920@60fps`, `5.9 MB`, `37.5s` GPU render, `-14.2 LUFS`.
- **Taylor `cos(x)` V2 (`V2_v2_taylor_series_approximation_of_cos_x_shorts_hd.mp4`):** `35.94s`, `1080×1920@60fps`, `3.0 MB`, `15.9s` GPU render, `-15.5 LUFS`, `-2.1 dBFS` TruePeak.
