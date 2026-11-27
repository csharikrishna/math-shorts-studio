# Software, Model, Voice & Reference License Manifest

All runtime dependencies, models, voices, tools, and researched repositories used by **Math Shorts Studio** require **zero mandatory paid subscriptions, royalties, or API keys** (aside from user-supplied optional free-tier Gemini API keys).

## 1. Python Runtime Dependencies (`.venv`)

| Package | Version | License (Verified Metadata) | Role |
| --- | --- | --- | --- |
| `manimgx` | `0.1.1` | `MIT AND GPL-2.0-or-later AND LGPL-2.1-or-later AND Apache-2.0 AND BSD-2-Clause` | Local Vulkan GPU math animation renderer |
| `manimgx-fonts` / `manimgx-fonts-cjk` | `2026.9.26.post1` | Bundled with `manimgx` | Local font assets for ManimGX |
| `numpy` | `2.5.5` | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | Numerical & array computation |
| `sympy` | `1.14.0` | `BSD` | Symbolic verification & calculus/algebra |
| `pydantic` | `2.14.4` | `MIT` | Typed schema & JSON validation |
| `google-genai` | `3.6.1` | `Apache-2.0` | Free-tier Gemini API client |
| `pillow` | `13.0.0` | `MIT-CMU` | Image QA & frame verification |
| `kokoro` | `0.9.4` | `Apache License` (Apache-2.0) | Primary local neural TTS inference |
| `en-core-web-sm` | `3.8.0` | `MIT` | Local English tokenization (`misaki` dependency) |
| `soundfile` | `0.14.0` | `BSD 3-Clause License` | Audio I/O for local synthesis |
| `edge-tts` | `7.3.1` | `LGPLv3` (PyPI Classifier) | Optional opt-in network TTS fallback |
| `websockets` | `17.4` | `BSD-3-Clause` | Headless Chrome DevTools Protocol client |
| `httpx` | `0.28.1` | `BSD-3-Clause` | Local HTTP / OpenAI-compatible client |
| `torch` | `2.15.0` | `Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause` | CPU inference backend for Kokoro |
| `spacy` | `3.8.18` | `MIT` | NLP pipeline dependency of `kokoro` |

## 2. Local Models, Voices & System Binaries

| Component | Source / Identifier | License | Paid / Cloud Requirement |
| --- | --- | --- | --- |
| Kokoro-82M (`am_michael`, `af_bella`, `bm_george`, `am_adam`, `af_heart`) | `hexgrad/Kokoro-82M` (cached locally after initial Hugging Face pull) | Apache-2.0 | None (100% offline CPU synthesis after cache) |
| FFmpeg / FFprobe | Local Python Scripts install | GPL / LGPL (system tool invocation) | None (local CLI subprocess) |
| Google Chrome / Edge | System browser binary (`chrome.exe` / `msedge.exe`) | Chromium BSD + Google Chrome ToS (executed unmodified as external browser binary) | None (offline loopback capture; external hosts blocked) |
| Typography | Local pinned SIL OFL-1.1 fonts in `assets/fonts/` (`Inter`, `Source Sans 3`, `STIX Two Text`, `STIX Two Math`) | SIL OFL-1.1 | None (100% offline bundled fonts; zero CDN scraping) |
| BGM & SFX | Procedural NumPy synthesis (`studio/audio/bgm.py`, `sfx.py`) | Project original code | None |

## 3. Researched Upstream Repositories (External Workspace Only)

| Upstream Repository | Commit SHA | License | Policy in This Repository |
| --- | --- | --- | --- |
| `heygen-com/hyperframes` | `3c85fbcde6eff6218fff2542ddd7473e75eea617` | Apache-2.0 (see `CREDITS.md` for GSAP Standard License note) | Pattern adaptation & independent Python/JS reimplementation. Proprietary GSAP & paid HeyGen services **excluded**. |
| `souzatharsis/podcastfy` | `053b54a0917e35edb2808049499292361ccaae7b` | Apache-2.0 | Prompting & dialogue-turn pattern adaptation. Paid OpenAI/ElevenLabs wrappers **excluded**. |
| `motion-canvas/motion-canvas` | `7b91435c301d530351dcf5ebb91dd139c002e405` | MIT | Easing, spring physics & flow combinator math reimplemented cleanly. |
| `3b1b/manim` | `fafa083a4fb274bba9cabde0b6e2f50ba6da0622` | MIT | Mathematical mobject & transform-matching architecture adapted over `manimgx`. |
| `3b1b/videos` | `31f8ffe785e860477a6051fde983c02c7d165bc2` | CC-BY-NC-SA-4.0 | **Educational study reference only.** Zero code, scripts, or assets copied into `math-shorts-studio`. |

## 4. Pinned Offline Typography Assets (`assets/fonts/`)

Verified via `studio.design.fonts.verify_pinned_fonts()` against `assets/fonts/FONT_MANIFEST.json`:

| Asset | Family / Role | License | Size (Bytes) | SHA-256 |
| --- | --- | --- | --- | --- |
| `Inter-Variable.ttf` (+ `Inter-OFL.txt`) | `Inter` (Display / UI) | SIL OFL-1.1 | `876,576` | `29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031` |
| `SourceSans3-Variable.ttf` (+ `SourceSans3-OFL.txt`) | `Source Sans 3` (Body / Captions) | SIL OFL-1.1 | `646,340` | `042fe2cc0b933e328410d7acbd0aa6a1873dca5aef81875f4bc214b08825c7b9` |
| `STIXTwoText-Variable.ttf` (+ `STIXTwo-OFL.txt`) | `STIX Two Text` (Editorial Serif) | SIL OFL-1.1 | `418,956` | `7962b8b7811e6a896c9a91a0bccbb5241047770eb24d4997c5cb5fe21d5c0df2` |
| `STIX2Math.otf` (+ `STIXTwo-OFL.txt`) | `STIX Two Math` (Mathematical Symbols) | SIL OFL-1.1 | `808,220` | `56cc42241cccc86d6d570134aa918fe8ac714f4b340d9916282fcc53c592d546` |

## 5. Pinned Offline Mathematical Typesetting Assets (`assets/vendor/katex/`)

Verified via `studio.renderers.hyperframes.typography.verify_vendored_katex()` against `assets/vendor/katex/KATEX_MANIFEST.json`:

| Asset | Version / Role | License | SHA-256 / Count |
| --- | --- | --- | --- |
| `katex.min.js` | KaTeX `0.19.1` offline typesetting runtime | MIT | `fa140498b8cfc8302cbe5e7cb1bdbc1f952bc38b3fa258ede09e73edbf9a74bc` |
| `katex.min.css` | KaTeX `0.19.1` CSS stylesheet | MIT | `00e0b094171b07f5842207f5469b4feac2b2f7bce1aeec943c08da4d697f4dcf` |
| `fonts/*.woff2` | Offline KaTeX math/serif font glyphs | MIT / OFL | `20` bundled `.woff2` files (zero remote CDN calls) |
| `LICENSE` | Bundled MIT license text | MIT | Recorded in `KATEX_MANIFEST.json` |
