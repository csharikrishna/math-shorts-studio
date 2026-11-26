# CLAUDE.md — Math Shorts Studio

An autonomous math Shorts engine: Gemini writes a validated `ShortPlan`, ManimGX renders it on the GPU, and edge-tts or Kokoro voices it with word-synced kinetic captions. Procedural NumPy stereo DSP supplies the SFX and music, and FFmpeg masters the result (sidechain ducking plus an optional cinematic look). Hand-written scenes use the same beat engine.

## Project Structure

- `studio/config.py` — `FORMATS` (`shorts_hd` 1080×1920@60, `shorts_fast` 720×1280@30, `widescreen_hd` 1920×1080@60), safe zones, `THEMES`.
- `studio/scene.py` — `StoryScene` / `Story3DScene` with `play_beat(beat_id, *anims, hold=, cues=, sfx=, ...)`.
  - Durations come from the live scene clock, so there is no drift.
  - The first beat absorbs the voice lead-in, so the hook shows at frame 0.
  - `cues` fire at narration bookmarks; `hold` fills the beat tail; risers are timed to the next beat.
- `studio/layout/components.py`:
  - `SafeLayout`, `HeaderBanner`, `fit_mobject`, `wrap_text`
  - UI style (`--ui`, env `STUDIO_UI`): `minimal` (default, box-free) or `card` (boxed).
  - `CaptionCard`: word-synced phrase chunks. Minimal: bare outlined words, the active word lit yellow with a small pop, cyan springy `*emphasis*`. Card: the active word on a pill inside a caption card.
  - `StatBadge`: live readout. Minimal: one centered `label · prefix number` line, or two centered rows on narrow frames. Card: an opaque pill. Optional formatter such as `adaptive_tex` / `precise_tex`.
  - `glow()`: layered-stroke halo; ManimGX bloom only affects material surfaces
  - `ambient_grid()`: drifting `NumberPlane`
  - `scene_time()`
- `studio/audio/tts.py`:
  - `parse_script` handles `*emphasis*` and `<bookmark mark='x'/>`.
  - Synthesis is edge-tts with `WordBoundary` events, or Kokoro (`--tts kokoro`) with token timestamps and a whole-run fallback to edge-tts.
  - Silence is trimmed in NumPy, and `align_words` maps word timings onto script tokens.
  - Writes `timeline.json` (with `words` and `marks`).
- `studio/audio/sfx.py` (incl. `riser_soft`, versioned cache), `bgm.py`, `master.py` (`look=none|cinematic|warm|cool`, `lut_path`).
- `studio/ai/`:
  - `director.py`: Gemini concept slate, honesty gate (`assess`), plan compiler, and validate→repair loop. Lite-first model chain with a 60 s request timeout and retry/backoff on 429/5xx/timeouts.
  - `schema.py`: the Pydantic `ShortPlan`.
  - `validate.py`: static plan QA plus storytelling rules (question hook, 1–2-word emphasis, zoom when terms crowd).
  - `expr.py`: AST-whitelisted NumPy math language, with no `eval`.
  - `env.py`: reads the key from `GEMINI_API_KEY` / `gemini_api` in env or `.env`.
- `scenes/ai_short.py` — plan-driven templates: `AIShort` (graph / parametric / sequence, with per-beat zoom), `AIShort3D` (surface) and `AIShortPolytope` (polytope4d).
- `studio/geometry4d.py` — regular 4D polytopes, R^4 plane rotations, perspective projection to R^3.
- `scenes/` hand-written: `taylor_series.py`, `fourier_square_wave.py`, `hessian_saddle_3d.py`.
- `plans/` — saved `ShortPlan` JSON. `zero_point_nine.json` is the offline sample and the template's default.
- `output/` — mastered `.mp4`. `build/<Name>_<format>/` — raw render, storyboards, `.srt`, timeline, stems.

## Commands

```powershell
uv run python -m studio.pipeline --topic "Why 0.999... equals 1" --format shorts_hd --ai   # autonomous
uv run python -m studio.pipeline --prompt "<concept>" --look cinematic                        # explicit concept
uv run python -m studio.pipeline --plan plans/zero_point_nine.json                             # saved plan, no Gemini
uv run python -m studio.pipeline scenes/taylor_series.py --format widescreen_hd                # hand-written scene
uv sync --extra kokoro                                                                          # optional local voice
```

`uv` lives at `%USERPROFILE%\.local\bin\uv.exe`. `.venv\Scripts\python.exe -m studio.pipeline ...` works too.

## Engineering Rules

1. **No hardcoded timings.** Never `self.wait(<number>)`. Declare `BEATS` and drive everything via `play_beat`. Use bookmarks and `cues` for word-exact accents.
2. **Aspect-ratio agnostic.** Position only from `SafeLayout` (`stage_center`, `stage_w`, `stage_h`, `min_text_h`). Every scene must pass `shorts_hd` and `widescreen_hd`.
3. **Strict visual QA.** Every 2D render must report `layout: no problems`. Always read `build/<Name>_<format>/*storyboard*.png`. 3D scenes are not auto-checked, so inspect them by eye.
4. **QA quirks to design around:**
   - Text crossed by a shape is flagged unless the shape is a `NumberPlane`/grid, sits under an opaque (≥ 0.9 fill) backdrop, or is a `BackgroundRectangle`. In the default box-free UI, readouts and captions therefore live in empty space below the plot. Keep tracers, curves and arrowheads out of that zone (`ai_short.py` places the readout under the lowest label or arrowhead). Card-style pills are `BackgroundRectangle`s.
   - Text under 20 px (at 1080p-normalized height) is flagged.
   - `MathTex` silently auto-closes unbalanced braces, so `validate.py` checks them explicitly.
5. **AI safety and honesty.**
   - Never `eval` model output.
   - Never let a template stand in for a concept it can't draw; the `assess` gate rejects such prompts.
   - To support a new kind of visual, add a real template, extending `schema.py`, `validate.py`, `ai_short.py`, the pipeline's scene routing and the director's `STYLE_GUIDE` together.
6. **Secrets.** The Gemini key lives in `.env`. Never print or log it, and never commit `.env`.
7. **Explore `manimgx` internals** (`.venv\Lib\site-packages\manimgx\`) before reinventing primitives: `add_subcaption`, `NumberPlane`, `ValueTracker`, `always_redraw`, rate functions, `.animate(rate_func=...)`.
