# Math Shorts Studio

An autonomous math Shorts engine: **topic → Gemini creative director → validated beat plan → ManimGX GPU render → word-synced kinetic captions → procedural stereo sound design → cinematic FFmpeg master**, in 9:16 or 16:9.

## Quick start

```bash
uv sync                                   # core install
uv sync --extra kokoro                    # optional local Kokoro-82M voice (torch + spaCy model)

# Fully autonomous: a field or a topic -> mastered Short
uv run python -m studio.pipeline --topic "Why 0.999... equals 1" --format shorts_hd --ai
uv run python -m studio.pipeline --topic "Chaos Theory" --look cinematic

# An explicit concept
uv run python -m studio.pipeline --prompt "Fourier series sharpening into a square wave"

# Re-render a saved plan (no Gemini call), or a hand-written scene
uv run python -m studio.pipeline --plan plans/zero_point_nine.json --look cinematic
uv run python -m studio.pipeline scenes/taylor_series.py --format widescreen_hd
```

Outputs: `output/<Name>_<format>.mp4`. Each build also leaves `build/<Name>_<format>/` with the raw render, storyboard PNGs, an `.srt` subtitle file, `timeline.json` and the audio stems.

The Gemini key is read from `GEMINI_API_KEY` (or `gemini_api`) in the environment or in `.env`. It is never logged.

## CLI options

| Option | Meaning |
|---|---|
| `--topic` | A field ("Number Theory") gets a concept slate scored for surprise × visual fit; a specific topic goes straight to scripting. |
| `--prompt` | An explicit concept to script and stage. |
| `--plan PATH` | Render a saved `ShortPlan` JSON. AI runs save theirs to `plans/`. |
| `--model` | Gemini model. Default: `$GEMINI_MODEL`, then `gemini-3.5-flash-lite` → `3.1-flash-lite` → `3.5-flash` → `3.8-flash`. Lite models come first because they answer in seconds while the flagship models are often overloaded. Gemini 2.x/1.5 return 404 for new API keys. |
| `--format` | `shorts_hd` (1080×1920@60), `shorts_fast` (720×1280@30), `widescreen_hd` (1920×1080@60), `widescreen_fast` (1280×720@30). |
| `--theme` | `light`, `dark`, `midnight_studio`, `obsidian_chalk`, `cartesian_stone`, `editorial_paper` (`--look-theme` alias; WCAG AA validated). |
| `--caption-style` | `editorial`, `kinetic`, `minimal`, `dialogue`, `accessible`, `off` (unified `CaptionIR` across ASS/SRT/VTT, ManimGX, and HyperFrames). |
| `--backend` | `manimgx` (default GPU engine), `hyperframes` (offline Chromium CDP SVG/DOM renderer), or `hybrid` (scene-level automatic router). |
| `--longform --duration N` | Generate cohesive chaptered documentary (`N` seconds) with continuity bridges, narrative artifacts (`STORYLINES.md`, `SCRIPT.md`, `STORYBOARD.md`, `REVIEW.md`), and contact-sheet visual QA. |
| `--derive-short` | Recompose a standalone native 9:16 portrait Short cleanly from `--longform`. |
| `--tts` / `--no-edge-fallback` | `kokoro` (default offline 82M neural TTS) or `edge`; `--no-edge-fallback` disables network fallback strictly. |
| `--voice`, `--rate`, `--pitch` | Voice ID and prosody, e.g. `--rate="-4%" --pitch="-2Hz"`. |
| `--look` | `none`, `cinematic`, `warm`, `cool`: fast YUV-native color grade (GPU `h264_nvenc` accelerated when available). |
| `--lut PATH` | A `.cube` LUT applied before the look. |
| `--ui` | Legacy UI alias mapped into `--caption-style` when `--caption-style` is omitted (`minimal` / `card` / `neon`). |
| `--output-stem` | Override output MP4 stem name cleanly. |
| `--allow-layout-warnings` | Do not fail on ManimGX layout QA problems (off by default). |

## How it works

### 1. Gemini creative director (`studio/ai/`)
- `director.py` runs four stages:
  - **Concept slate** (fields only): several ideas, the best picked by surprise × visual fit.
  - **Honesty gate** (`--prompt` and specific topics): Gemini judges whether a template can draw the real object. If it can't, the run stops with the reason and the closest honest alternative instead of faking it with unrelated visuals.
  - **Plan compiler.**
  - **Validate-and-repair loop** of up to 3 rounds.

  Every request has a 60 s timeout. Transient 429/5xx errors and timeouts are retried with backoff, then the next model is tried, so a busy endpoint can't hang the run.
- `schema.py` holds the Pydantic `ShortPlan`, which is both Gemini's `response_schema` and the template's only input. Each beat has a narration (≤ 25 spoken words, 8–12 beats for 45–90 s explanations), caption, formula, color role, SFX cue and `visual_intent`.
- `validate.py` checks a plan statically before any TTS or GPU time is spent:
  - every expression is compiled and sampled over its plot window, including per-beat zoom windows
  - every formula is test-typeset and its braces are balanced
  - word and label budgets are enforced
  - every bookmark referenced by a cue exists in the narration

  It also applies storytelling rules: the hook must pose a question rather than state the payoff, emphasis must be 1–2 words, and a converging sequence must zoom if its terms would pile up on the limit line. Each problem is phrased as a fix instruction, which goes back to Gemini. These rules, plus a worked example in the prompt, let fast lite models write strong scripts.
- `expr.py` is a safe math language. Gemini writes `sin(x) + x**3/6`, `1 - 10**(-n)` or `4/pi*sum(k, 1, 19, 2, sin(k*x)/k)`. It is parsed with `ast`, checked against a whitelist and evaluated with NumPy; nothing an LLM writes is passed to `eval`.

Gemini never writes executable code: it writes a plan, and `scenes/ai_short.py` renders it.

### 2. Template scenes (`scenes/ai_short.py`)

| Template | Shows | Live readout |
|---|---|---|
| `graph` | y = f(x) morphing per beat, optional reference curve + error ribbon | f(x), f′(x) or \|f − ref\| at a sweeping tangent tracer |
| `parametric` | (X(t), Y(t)) with a moving point | t, speed, radius |
| `sequence` | a(n) as dots converging on a limit line | aₙ (up to 8 decimals), Δaₙ, \|aₙ − L\| (scientific notation) |
| `surface` (3D) | z = f(x, y) with an orbiting camera, gradient probe and x/y slices | f, \|∇f\| |
| `polytope4d` (3D) | a true tesseract / 16-cell / 5-cell rotated in a 4D plane and perspective-projected to 3D; inner/outer cells and connectors highlighted per beat | rotation angle θ, spin rate ω |

Graph and sequence beats can **zoom** (`visual_intent.y_range`). The view is an animated affine map, so curves, dots, the reference line and the tracer all dive together, with a camera punch-in.

### 3. Beat engine (`studio/scene.py`)
- **No dead air:** `play_beat()` measures each beat against the live scene clock, so beats end exactly on their audio and rounding never drifts. The first beat absorbs the voice lead-in, so the hook visual is on screen at frame 0.
- **Bookmark cues:** narration may contain manim-voiceover-style `<bookmark mark='x'/>` tags; `cues={"x": ...}` fires animations exactly when that word is spoken.
- **Beat tails** play `hold` micro-animations while updaters (tracers, readouts, drifting grid) keep the stage moving. A riser SFX is timed to peak on the next beat.

### 4. Voice and captions (`studio/audio/tts.py`, `studio/layout/components.py`)
- **Word timings:** edge-tts `WordBoundary` events (or Kokoro token `start_ts`/`end_ts`) are captured. Silence is trimmed in NumPy so the offsets stay exact.
- **Alignment:** word timings are aligned onto the script's own tokens, with punctuation kept. Unmatched tokens are interpolated by character position.
- **Captions:** `CaptionCard` shows phrase-aware chunks of up to 4 words, breaking at punctuation.
  - In the default minimal style there is no box. Captions are big bold words with a soft dark outline (MoneyPrinterTurbo-style), legible over grids and curves.
  - Spoken words are white and upcoming ones muted. The active word lights up yellow with a small pop; `*emphasis*` words light up cyan with a damped-spring bounce. Every pop is capped so it never reaches neighbouring words.
  - `--ui card` restores the boxed caption card with a highlight pill.
- **Subtitles:** each beat's caption is also written to the `.srt` file.

### 5. Sound and mastering (`studio/audio/`)
- **`sfx.py`:** procedural stereo whoosh, marimba pop, tick, chime, epiphany chord and riser, with constant-power panning and a decorrelated stereo room.
- **`bgm.py`:** chorus-modulated pad progression, a ping-pong pluck arpeggio and a filtered air bed, sized to the timeline.
- **`master.py`:** sidechain ducking of the music under the voice, stereo widening of SFX and music, a limiter, and an optional grade/LUT, vignette and grain.

## Quality rules
- Never use `self.wait(<number>)`: declare `BEATS` and drive everything through `play_beat`.
- Position from `SafeLayout` (`stage_center`, `stage_w`, `stage_h`, `min_text_h`) only; every scene must render in `shorts_hd` and `widescreen_hd`.
- Every 2D render must pass `layout: no problems`, the pipeline's default.
- Read the storyboard PNGs in `build/` before calling a render done.

## Credits & Acknowledgments
This project stands on the shoulders of awesome open-source work:
- **[academa-labs/manimgx](https://github.com/academa-labs/manimgx)** — Rust/wgpu GPU-accelerated Manim engine powering all scene rendering and layout diagnostics.
- **[3b1b/manim](https://github.com/3b1b/manim)** & **[ManimCommunity/manim](https://github.com/ManimCommunity/manim)** — the mathematical animation API and storytelling inspiration.
- **[ManimCommunity/manim-voiceover](https://github.com/ManimCommunity/manim-voiceover)** — inspiration for `<bookmark mark='...'/>` audio-visual cue synchronization.
- **[hexgrad/kokoro](https://github.com/hexgrad/kokoro)** — Kokoro-82M local neural TTS engine.
- **[FujiwaraChoki/MoneyPrinterTurbo](https://github.com/FujiwaraChoki/MoneyPrinterTurbo)** — inspiration for word-level Shorts caption alignment.
