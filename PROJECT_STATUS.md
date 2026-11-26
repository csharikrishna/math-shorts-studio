# Project Status — AI Educational Video Studio V2

**Branch:** `v2-studio`  
**Last Updated:** 2026-11-26

## Phase Progress

| Phase | Name | Status | Key Deliverables |
|---|---|---|---|
| **Phase 0** | Investigation & Baseline | ✅ **Done** | `docs/BASELINE_AUDIT.md`, `docs/TECHNOLOGY_SURVEY.md`, `docs/ARCHITECTURE.md` |
| **Phase 1** | Existing Quality Improvements | 🔄 **In Progress** | Two-pass `loudnorm` (`-14 LUFS`, `-1.0 dBTP`), `pytest` suite, visual polish |
| **Phase 2** | AI Provider Architecture | ⏳ Pending | `studio/providers/` (Gemini, OpenRouter `:free`, Ollama, `llama.cpp`) |
| **Phase 3** | Voice & Timeline Architecture | ⏳ Pending | Pronunciation normalizer, `TTSProvider`, `MasterTimeline`, WebVTT + transcript |
| **Phase 4** | Multi-Scene Animation Engine | ⏳ Pending | Composable multi-scene plans, reusable visual primitives |
| **Phase 5** | Long-Form & Repurposing | ⏳ Pending | Knowledge tree, 16:9 chapters, long-to-short repurposing |
| **Phase 6** | Performance & Persistence | ⏳ Pending | SQLite job store, content-addressed cache, checkpoints |
| **Phase 7** | Quality Intelligence | ⏳ Pending | SymPy math QA, PIL/NumPy visual QA, audio QA, bounded repair |
| **Phase 8** | Experimental Validation | ⏳ Pending | Fourier V1-vs-V2 benchmark, generalization suite |

## Baseline Metrics (V1)
- Reference video (`AI_explain_simply_step_by_step_how_adding_smooth_si_shorts_hd.mp4`): `60.31s`, `1080×1920@60fps`, `16.6 MB`, Integrated Loudness `-24.0 LUFS`, True Peak `-4.3 dBFS`.
