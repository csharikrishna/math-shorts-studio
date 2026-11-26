# System Architecture — AI Educational Video Studio V2

## High-Level Modular Pipeline

```
User Input (topic / prompt / doc / mode / audience)
  → Knowledge Tree & Reasoning (studio/ai/knowledge.py)
  → Creative Director & Multi-Scene Planner (studio/ai/director.py, studio/providers/)
  → Math & Static Validation (studio/qa/math_qa.py, studio/ai/validate.py)
  → Pronunciation Normalization & TTS (studio/audio/pronounce.py, studio/audio/tts.py)
  → Master Timeline Compilation (studio/timeline.py)
  → Composable Multi-Scene Rendering (scenes/ai_short.py, studio/layout/)
  → Audio Engineering (two-pass loudnorm -14 LUFS, sidechaincompress) (studio/audio/master.py)
  → Subtitle Export (SRT, WebVTT, word-timestamp JSON, transcript)
  → Automated Quality Evaluation & Bounded Repair (studio/qa/)
  → SQLite Job Persistence & Content-Addressed Cache (studio/storage/)
```

## Core Principles
1. **Free-Only AI Policy:** Gemini free tier is primary; OpenRouter strictly enforces `:free` models; Ollama & `llama.cpp` run locally.
2. **Local Media & Math:** Kokoro-82M TTS, SymPy verification, ManimGX Vulkan GPU rendering, and FFmpeg NVENC/x264 run 100% locally.
3. **Timeline-First Sync:** Narration audio is synthesized and measured *before* animation timing is compiled—zero hardcoded `wait()` calls.
4. **Safe Declarative Plans:** LLMs output Pydantic-validated JSON plans; no unrestricted `eval()` or `exec()`.
