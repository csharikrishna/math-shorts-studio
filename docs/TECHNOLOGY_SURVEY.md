# Technology Survey & License Manifest (V2)

**Date:** 2026-11-26  
**Constraint:** Zero mandatory API expenditure; commercially compatible licenses for monetized educational content.

---

## 1. Mathematical Animation & Rendering

| Repository / Tool | License | Role in V2 | Decision |
|---|---|---|---|
| **[academa-labs/manimgx](https://github.com/academa-labs/manimgx)** | Open Source (PyPI `manimgx>=0.1.1`) | Primary GPU-accelerated (Rust/wgpu Vulkan) rendering engine + layout QA | **RETAIN (Primary)** — renders 1080×1920@60fps ~5–10× faster than CPU Cairo Manim. |
| **[ManimCommunity/manim](https://github.com/ManimCommunity/manim)** | MIT | Secondary fallback backend for primitives not yet in ManimGX | **OPTIONAL FALLBACK** — evaluate only if a specific 3D/shader primitive is missing. |
| **[3b1b/manim](https://github.com/3b1b/manim)** & **[3b1b/videos](https://github.com/3b1b/videos)** | MIT (engine) / Custom (video scenes) | Educational storytelling & visual intuition reference | **REFERENCE ONLY** — do not copy non-commercial video scene code. |
| **[ManimCommunity/manim-voiceover](https://github.com/ManimCommunity/manim-voiceover)** | MIT | Bookmark-driven `<bookmark mark='x'/>` sync pattern | **PATTERN INTEGRATED** — lightweight native implementation already in `studio/scene.py`. |
| **Math-To-Manim / AiManim / faceless-shorts** | Various | Prompting & multi-scene composition patterns | **INSPIRATION ONLY** — avoid unrestricted LLM `exec()` of generated Python. |

---

## 2. AI / LLM Providers (Free-Tier & Local)

| Provider | Cost / Policy | Role in V2 | Decision |
|---|---|---|---|
| **Google Gemini (`google-genai`)** | Free tier (`GEMINI_API_KEY`) | Primary cloud LLM for concept, knowledge tree, and structured JSON planning | **RETAIN (Default)** |
| **OpenRouter** | Free models (`:free` suffix only) | Secondary cloud fallback; enforced by strict free-only policy | **INTEGRATE** |
| **[ollama/ollama](https://github.com/ollama/ollama)** | MIT | Local LLM inference (`localhost:11434`) | **INTEGRATE (Adapter)** |
| **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** | MIT | Local OpenAI-compatible GGUF server | **INTEGRATE (OpenAI-compat)** |

---

## 3. Text-to-Speech (Local, Free, Commercial-Friendly)

| Model / Repo | License | Size / Hardware | Decision |
|---|---|---|---|
| **[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** (`kokoro>=0.9`) | **Apache 2.0** | 82M params; runs fast on CPU & 4GB GTX 1650 | **RETAIN (Primary Local TTS)** — commercial-friendly, natural prosody, token timestamps. |
| **edge-tts** | LGPL-3.0 (wrapper) | Cloud (free, no key) | **RETAIN (Fallback)** — fast fallback when Kokoro is unavailable. |
| **[ResembleAI/chatterbox](https://github.com/resemble-ai/chatterbox)** | MIT | ~500M+ params | **DEFER / OPTIONAL** — heavier VRAM footprint on 4GB GTX 1650; benchmarked as optional adapter. |
| **[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** (`1.7B`) | Apache 2.0 | 1.7B params (~3.5+ GB VRAM) | **DEFER / OPTIONAL** — tight on 4GB VRAM alongside Vulkan rendering; support via `TTSProvider` interface. |

---

## 4. Speech Recognition & Forced Alignment

| Tool / Model | License | Role in V2 | Decision |
|---|---|---|---|
| **[SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)** (`small.en`) | MIT | Independent ASR verification of rendered audio & alignment confidence | **RECOMMENDED OPTIONAL** (CTranslate2, runs on CPU/4GB GPU). |
| **Native `WordBoundary` + Kokoro `start_ts`/`end_ts` + `align_words`** | Apache 2.0 / MIT | Primary zero-overhead word alignment | **RETAIN (Primary)** — deterministic, zero extra VRAM. |

---

## 5. Math, Media & Validation

| Library | License | Role | Status |
|---|---|---|---|
| **SymPy** | BSD-3-Clause | Symbolic derivative, integral, identity, and LaTeX verification | **Installed** — integrate into `studio/qa/math_qa.py`. |
| **NumPy** | BSD-3-Clause | Numerical sampling, DSP audio, 4D geometry | **Installed**. |
| **Pillow (PIL)** | HPND | Frame/storyboard inspection & thumbnail/image checks | **Installed**. |
| **FFmpeg 7.1 (`h264_nvenc` / `libx264`)** | GPL/LGPL | Two-pass `loudnorm`, `sidechaincompress`, NVENC encoding | **Installed**. |
| **SciPy / OpenCV (`cv2`)** | BSD / Apache 2.0 | Optional advanced numerical & frame-diff/contrast QA | **Optional** (graceful fallback to NumPy/PIL when absent). |
