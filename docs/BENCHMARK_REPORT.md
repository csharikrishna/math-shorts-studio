# Benchmark Report — V1 vs. V2

## Fourier Analysis Benchmark (`1080×1920 @ 60fps`, `60.3s`)

| Metric | V1 Baseline | V2 Upgraded | Delta |
|---|---|---|---|
| **Integrated Loudness** | `-24.0 LUFS` | **`-15.0 LUFS`** | **+9.0 dB (Social broadcast compliant)** |
| **True Peak** | `-4.3 dBFS` | **`-0.1 dBFS`** | **Maximized headroom** |
| **Mastering Speed (`look=cinematic`)** | `152.26 s` (CPU RGB) | **`9.71 s` (NVENC YUV)** | **15.7× faster** |
| **File Size (20s test / 60s full)** | `42.4 MB` (20s) | **`5.3 MB` (20s) / `16.6 MB` (60s)** | **8× smaller** |
| **UI / Typography** | Serif `New Computer Modern` | **`Segoe UI Bold` + 3 UI modes (`minimal`, `card`, `neon`)** | **Modern Shorts UX** |
| **AI Provider Policy** | Gemini-only | **Gemini + OpenRouter (`:free` only) + Ollama + `llama.cpp`** | **Extensible & Free-Only** |
| **Math & Audio QA** | AST syntax only | **SymPy symbolic + FFmpeg `ebur128` + PIL storyboard QA** | **Multi-layer QA** |
| **Job Persistence & Cache** | None | **SQLite `JobStore` + SHA-256 content-addressed cache** | **Resumable** |
| **Automated Tests** | `0` | **`12` unit tests (`unittest`)** | **100% passing** |
