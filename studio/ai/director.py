"""Gemini creative director: topic -> viral concept -> validated ShortPlan (beats + visuals).

Pipeline:
  1. Concept slate (fields like "Calculus" only): Gemini pitches several counter-intuitive
     ideas that fit the renderable templates; the best is picked by surprise x visual fit.
  2. Plan compiler: Gemini writes the full beat sheet as strict JSON (response_schema = ShortPlan).
  3. Validator/repair loop: every expression, formula, word budget and bookmark is checked
     statically (studio/ai/validate.py); problems go back to Gemini until the plan is clean.

The scene template (scenes/ai_short.py) renders the plan, so Gemini never writes executable code.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from pydantic import BaseModel

from studio.ai.env import PROJECT_ROOT, gemini_api_key
from studio.ai.schema import Concept, ConceptSlate, Feasibility, ShortPlan
from studio.ai.validate import validate_plan

DEFAULT_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash")
TRANSIENT_CODES = {429, 500, 502, 503, 504}  # rate limits / overload: retry with backoff
RETRIES_PER_MODEL = 2
BACKOFF_BASE_S = 4.0
REQUEST_TIMEOUT_S = 60.0  # lite models answer a full ShortPlan in seconds; beyond this treat the call as hung
MAX_REPAIRS = 3
PLANS_DIR = PROJECT_ROOT / "plans"

STYLE_GUIDE = """
You are the creative director and script compiler for a 3Blue1Brown-calibre channel of 45-90 second
vertical (9:16) math Shorts. Voice: curious, warm, conversational, never hypey, like a smart friend
explaining something for the first time. Every Short:
- Opens with a 3-second COUNTER-INTUITIVE hook (a question or claim that sounds wrong or surprising);
  the hook visual is on screen at frame 0, so beat 1 must already show a meaningful object.
- Then builds intuition step by step — EXPLAIN EACH STEP FULLY. Don't rush. Each beat narration
  is up to 25 spoken words of clear, friendly English (say "x squared", never LaTeX symbols).
  The viewer should feel they genuinely understand, not just watched a magic trick.
- Uses 8 to 12 beats: hook → setup → 3-6 explanation steps → zoom/reveal → payoff + summary.
- Wraps 1-2 key words per beat in *asterisks* (caption emphasis). Optionally place
  <bookmark mark='name'/> immediately before a word and add a matching cue to time an accent there.

RENDERABLE TEMPLATES (pick exactly one; the plan is rendered by a fixed engine):
- graph: y = f(x) on axes. Each beat's visual_intent.expr is a new f(x) (morphs from the previous).
  Optional reference_expr = fixed target curve (e.g. sin(x)); show_error_area shades the gap;
  readout: value=f(x), slope=f'(x), error=|f-ref| at a moving tracer.
- ZOOM (graph + sequence): visual_intent.y_range = [min, max] animates the vertical window from that beat on.
  Use it to dive into tiny differences (e.g. zoom 100x into the gap between 0.999 and 1, then again);
  at least 3 sequence terms / most of a graph must stay inside the zoomed window.
- parametric: expr "X(t); Y(t)" (semicolon!), t_range [t0, t1]. readout value=t, slope=speed, error=radius.
- sequence: expr a(n) (variable n), terms n=1..n_max (<=40) shown as dots; reference_expr is a limit
  line as a function of x (e.g. "1"). x_range covers n (e.g. [0, n_max+1]).
  readout value=a_n, slope=a_n - a_(n-1), error=|a_n - L|.
- surface: z = f(x, y) in 3D with an orbiting camera; slice_axis "x"/"y" draws f(x,0) / f(0,y).
  Needs z_range. readout value=f, slope=|grad f|.
- polytope4d: a regular 4D polytope (polytope: "tesseract" | "16-cell" | "5-cell") truly rotated in R^4 and
  perspective-projected to 3D (its "shadow"); expr must be null; x_range/y_range unused (use [-1, 1]).
  Per beat: rotation_plane "xw"/"yw"/"zw" turns it THROUGH the 4th dimension (the shadow turns inside out),
  "xy"/"xz"/"yz" are ordinary 3D spins; rotation_speed 0-0.5 turns/s (0 holds still);
  highlight "inner"/"outer"/"connectors"/"none". At rest the tesseract shadow is a small cube inside a big cube,
  joined at every vertex by 8 connector edges. readout value=rotation angle (deg), slope=spin rate (deg/s).

HONESTY: draw the real object. Never stand an unrelated function or surface in for a concept and label it as
that concept; if no template can show it faithfully, the concept is rejected upstream.

MATH EXPRESSION GRAMMAR (NumPy-evaluated, strictly whitelisted):
- variables: x (graph), t (parametric), n (sequence), x and y (surface)
- operators + - * / ** %, comparisons < <= > >= (as 0/1), conditional "a if cond else b"
- constants pi, e, tau, phi
- functions sin cos tan arcsin arccos arctan sinh cosh tanh exp log log10 sqrt abs sign floor ceil
  minimum maximum mod heaviside factorial gamma
- finite sums: sum(k, start, stop, step, expr_in_k) with inclusive stop, e.g.
  Fourier square wave "4/pi*sum(k, 1, 19, 2, sin(k*x)/k)", Taylor sine
  "sum(k, 0, 5, 1, (-1)**k * x**(2*k+1) / factorial(2*k+1))"
- choose x_range / y_range so every expr stays inside the window on most of the domain.

LAYOUT RULES (the frame is narrow and tall):
- title <= 28 chars, tag <= 24 chars, badge_label <= 14 chars, caption <= 48 chars.
- formula: one short LaTeX line (no $), readable on a phone: prefer <= 40 LaTeX characters,
  at most one \\frac or one \\sum. Standard amsmath commands only.
- x_labels: at most 3 landmark ticks (e.g. -\\pi, \\pi).
- sfx_cue: whoosh_soft for the hook, pop_marimba/tick_soft for steps, chime_reveal for reveals,
  chord_epiphany only for the payoff.
- color_role: progress primary/secondary -> tertiary -> success, payoff = highlight.

STORY CRAFT (what separates a great Short from a list of facts):
- The hook formula POSES the question (\\overset{?}{=}, a "?" or an unfinished claim); never state the payoff early.
- Every beat changes exactly one visible thing, and the narration names AND EXPLAINS what just changed.
- After each change, the narration gives the viewer time to absorb it with a full clear sentence.
- When values crowd together (convergence, tiny errors), ZOOM with y_range so the viewer SEES the gap shrink.
- Emphasis: 1-2 single key words per beat (*cliff*, *zero*), never whole phrases.
- Payoff: resolve the hook in one surprising, plain sentence. Then add a brief summary beat.

WORKED EXAMPLE (format and quality reference; do not copy its topic):
{"topic": "Fourier square wave", "tag": "Fourier Analysis", "title": "Smooth Waves, Sharp Edges",
 "template": "graph", "x_range": [-3.6, 3.6], "y_range": [-1.6, 1.6], "reference_expr": "sign(sin(x))",
 "x_labels": [{"value": -3.14159, "tex": "-\\\\pi"}, {"value": 3.14159, "tex": "\\\\pi"}],
 "beats": [
  {"beat_id": "hook", "narration": "Here's something that sounds *impossible*: smooth, curving waves can combine to build a perfectly sharp corner.",
   "caption": "Can smooth waves build a sharp corner?", "formula": "\\\\sum \\\\sin(kx) \\\\overset{?}{=} \\\\text{square}",
   "color_role": "secondary", "sfx_cue": "whoosh_soft",
   "visual_intent": {"expr": "4/pi*sin(x)", "readout": "value", "badge_label": "1 wave", "cues": []}},
  {"beat_id": "first_wave", "narration": "We start with just one sine wave. It goes up, comes down, perfectly smooth.",
   "caption": "One sine wave: perfectly smooth", "formula": "S_1 = \\\\tfrac{4}{\\\\pi}\\\\sin x",
   "color_role": "primary", "sfx_cue": "tick_soft",
   "visual_intent": {"expr": "4/pi*sin(x)", "readout": "value", "badge_label": "1 wave", "cues": []}},
  {"beat_id": "add_three", "narration": "Add a third harmonic, and the peaks start to <bookmark mark='flat'/>*flatten*. The shape is becoming more square.",
   "caption": "Add 3x: peaks flatten", "formula": "S_3 = \\\\tfrac{4}{\\\\pi}(\\\\sin x + \\\\tfrac{\\\\sin 3x}{3})",
   "color_role": "secondary", "sfx_cue": "pop_marimba",
   "visual_intent": {"expr": "4/pi*sum(k, 1, 3, 2, sin(k*x)/k)", "readout": "error", "show_error_area": true,
                     "badge_label": "2 waves", "cues": [{"mark": "flat", "action": "sweep_curve"}]}},
  "... 4-7 more beats explaining each harmonic, then zoom into the overshoot ...",
  {"beat_id": "payoff", "narration": "Infinitely many smooth sine waves combine to make a perfectly *sharp* edge. The math behind your MP3 player.",
   "caption": "Infinite smooth waves = sharp edge", "formula": "\\\\tfrac{4}{\\\\pi}\\\\sum_{k\\\\,odd}\\\\tfrac{\\\\sin kx}{k} = \\\\text{square}",
   "color_role": "highlight", "sfx_cue": "chord_epiphany",
   "visual_intent": {"expr": "4/pi*sum(k, 1, 39, 2, sin(k*x)/k)", "readout": "error", "badge_label": "20 waves", "cues": []}}]}
"""


class DirectorError(RuntimeError):
    pass


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:limit] or "short"


class GeminiDirector:
    """Provider-backed creative director using LLMProvider structured generation."""

    def __init__(
        self,
        model: str | None = None,
        verbose: bool = True,
        provider_name: str = "gemini",
        free_only: bool = True,
        provider: object | None = None,
    ) -> None:
        from studio.providers import get_provider

        self.provider = provider or get_provider(provider_name, free_only=free_only, model=model)
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"      [director] {msg}", flush=True)

    def _generate(self, prompt: str, schema: type[BaseModel], temperature: float) -> BaseModel:
        from studio.providers import ProviderError

        self._log(f"{self.provider.name}: requesting {schema.__name__}")
        try:
            parsed, resp = self.provider.generate_structured(prompt, schema, system=STYLE_GUIDE, temperature=temperature)
            self._log(f"{resp.provider}/{resp.model}: answered in {resp.latency_s:.1f}s")
            return parsed
        except ProviderError as exc:
            raise DirectorError(f"AI provider failed: {exc}") from exc

    # -- stage 1 ----------------------------------------------------------------
    def select_concept(self, field: str, count: int = 5) -> Concept:
        prompt = (
            f"Math field: {field}\n"
            f"Pitch {count} distinct Short concepts for this field. Each must have a hook that sounds wrong or "
            "surprising, a payoff a curious teenager can follow, and must be honestly showable with ONE of the "
            "renderable templates (graph, parametric, sequence, surface, polytope4d). Score visual_fit and "
            "surprise honestly."
        )
        slate = self._generate(prompt, ConceptSlate, temperature=1.0)
        assert isinstance(slate, ConceptSlate)
        if not slate.concepts:
            raise DirectorError("Gemini returned no concepts.")
        best = max(slate.concepts, key=lambda c: (c.surprise * c.visual_fit, c.visual_fit))
        for c in slate.concepts:
            mark = "*" if c is best else " "
            self._log(f"{mark} [{c.template:<10}] surprise={c.surprise} fit={c.visual_fit}  {c.title}")
        return best

    def assess(self, concept: str) -> Feasibility:
        """Honesty gate: refuse concepts the templates could only fake with unrelated visuals."""
        prompt = (
            f"Concept: {concept}\n"
            "Can this be shown FAITHFULLY with one renderable template, drawing the real mathematical object "
            "(not an unrelated curve or surface with a matching label)? Answer fits=false if the only option "
            "would be a stand-in, and then propose the closest concept the templates can show truthfully."
        )
        verdict = self._generate(prompt, Feasibility, temperature=0.2)
        assert isinstance(verdict, Feasibility)
        self._log(f"feasibility: fits={verdict.fits} template={verdict.template} ({verdict.reason})")
        return verdict

    # -- stage 2 + 3 --------------------------------------------------------------
    def compile_plan(self, brief: str) -> ShortPlan:
        prompt = (
            f"Write the complete, unhurried 45-80 second ShortPlan for this Short.\n\nBRIEF:\n{brief}\n\n"
            "Return only the JSON plan. Requirements:\n"
            "- Write 9 to 12 beats so the concept is explained simply, clearly, and step-by-step for a beginner.\n"
            "- Beat 1 shows the hook object at frame 0.\n"
            "- Keep each beat's narration clear, conversational, and <= 25 spoken words.\n"
            "- Expressions must use the math grammar exactly."
        )
        plan = self._generate(prompt, ShortPlan, temperature=0.8)
        assert isinstance(plan, ShortPlan)
        for attempt in range(1, MAX_REPAIRS + 1):
            problems = validate_plan(plan)
            if not problems:
                self._log(f"plan valid after {attempt - 1} repair round(s)")
                return plan
            self._log(f"repair round {attempt}: {len(problems)} problem(s)")
            for p in problems[:8]:
                self._log(f"  - {p}")
            repair = (
                "This ShortPlan failed automated validation. Fix every problem while keeping the story, "
                "and return the full corrected plan.\n\nPROBLEMS:\n- "
                + "\n- ".join(problems)
                + "\n\nCURRENT PLAN JSON:\n"
                + plan.model_dump_json(indent=1)
            )
            plan = self._generate(repair, ShortPlan, temperature=0.4)
            assert isinstance(plan, ShortPlan)
        problems = validate_plan(plan)
        if problems:
            raise DirectorError("Plan still invalid after repairs:\n- " + "\n- ".join(problems))
        return plan


def create_plan(
    topic: str | None = None,
    prompt: str | None = None,
    model: str | None = None,
    out_dir: Path = PLANS_DIR,
    provider_name: str = "gemini",
    free_only: bool = True,
) -> Path:
    """Topic/field or explicit prompt -> validated plan JSON on disk (re-renderable with --plan)."""
    if not topic and not prompt:
        raise ValueError("Give a topic or a prompt.")
    director = GeminiDirector(model=model, provider_name=provider_name, free_only=free_only)
    looks_like_field = False
    if prompt:
        brief = f"Explicit concept from the user: {prompt}"
        label = prompt
    else:
        assert topic is not None
        looks_like_field = len(topic.split()) <= 3 and not any(ch.isdigit() for ch in topic)
        if looks_like_field:
            concept = director.select_concept(topic)
            brief = (
                f"Field: {topic}\nChosen concept: {concept.title}\nHook: {concept.hook_question}\n"
                f"Why it works: {concept.why_viral}\nTemplate: {concept.template}"
            )
        else:
            brief = f"Topic: {topic}. Find the most counter-intuitive honest hook for it."
        label = topic
    if prompt or not looks_like_field:  # slate concepts were already chosen for template fit
        verdict = director.assess(prompt or topic or "")
        if not verdict.fits:
            raise DirectorError(
                f"This concept can't be shown honestly with the current templates: {verdict.reason}\n"
                f"Closest honest Short: {verdict.honest_alternative}\n"
                f'Re-run with: --prompt "{verdict.honest_alternative}"'
            )
        brief += f"\nUse the {verdict.template} template."
    plan = director.compile_plan(brief)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(label)}.json"
    path.write_text(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


V2_STYLE_GUIDE = """
You are an educational math filmmaker creating a multi-scene VideoPlan (schema_version="2.0").
Each scene has:
- scene_id: unique snake_case id
- purpose: concise educational goal (< 60 chars)
- narration: warm, clear spoken English (<= 25 words), *emphasis* on 1-2 key words
- formula: short LaTeX (no $)
- color_role: primary | secondary | tertiary | success | highlight | danger
- sfx_cue: whoosh_soft | pop_marimba | tick_soft | chime_reveal | chord_epiphany
- primitive:
  - kind: contrast_waves | single_wave | stacked_addition | progressive_sum | zoom_overshoot | spectrum_bars | bar_chart
  - expr: valid NumPy-whitelisted function of x (required for all except spectrum_bars and bar_chart)
  - secondary_expr: optional second function of x for comparison
  - harmonics: list of positive integers (for spectrum_bars ONLY)
  - values: list of finite numbers (for bar_chart ONLY), bar_labels: matching labels, log_scale: bool
  - reference_y: optional horizontal reference line value, highlight_x: optional x-coordinate dot
Ensure displayed LaTeX formulas, degrees, and plotted expressions agree mathematically.
Create 6 to 8 progressive scenes: hook -> intuition -> construction -> discovery -> real-world payoff.
"""


def create_v2_plan(
    topic: str | None = None,
    prompt: str | None = None,
    model: str | None = None,
    out_dir: Path = PLANS_DIR,
    provider_name: str = "gemini",
    free_only: bool = True,
) -> Path:
    """Generate and validate a multi-scene VideoPlan V2 directly from a topic or prompt."""
    from studio.ai.multi_schema import VideoPlan
    from studio.ai.validate_v2 import validate_video_plan

    if not topic and not prompt:
        raise ValueError("Give a topic or a prompt.")
    label = prompt or topic or "video"
    director = GeminiDirector(model=model, provider_name=provider_name, free_only=free_only)
    req = f"{V2_STYLE_GUIDE}\n\nTopic/Concept: {label}\nGenerate a complete VideoPlan V2 JSON."
    plan, resp = director.provider.generate_structured(req, VideoPlan, system=V2_STYLE_GUIDE, temperature=0.5)
    director._log(f"{resp.provider}/{resp.model}: V2 plan generated in {resp.latency_s:.1f}s (tokens: {resp.prompt_tokens}+{resp.completion_tokens})")

    for round_idx in range(1, MAX_REPAIRS + 1):
        issues = validate_video_plan(plan)
        if not issues:
            break
        director._log(f"V2 repair round {round_idx}/{MAX_REPAIRS} ({len(issues)} issues)")
        fix_prompt = f"Fix these VideoPlan V2 validation issues:\n- " + "\n- ".join(issues) + f"\n\nCurrent JSON:\n{plan.model_dump_json()}"
        plan, _ = director.provider.generate_structured(fix_prompt, VideoPlan, system=V2_STYLE_GUIDE, temperature=0.3)

    issues = validate_video_plan(plan)
    if issues:
        raise DirectorError("VideoPlan V2 still invalid after repairs:\n- " + "\n- ".join(issues))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"v2_{slugify(label)}.json"
    path.write_text(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


V3_STYLE_GUIDE = """
You are an educational math filmmaker creating a composable VideoPlanV3 (schema_version="3.0").
Think through: explanation strategy -> prerequisites -> central discovery -> visual objects -> narration -> animation events.
STORYTELLING & DURATION RULES:
- Start Scene 1 with an immediate visual demonstration or concrete mathematical question.
- NEVER use dramatic cliché phrases like "Welcome to a mathematical journey", "Unlocking profound mathematical secrets", "treacherous mountain range", or "Mathematical relationships that last forever".
- Explain WHY mathematical relationships work in clear, warm educator English (no vague hype).
- Put 1-2 key words in *asterisks* for caption emphasis, and use <bookmark mark='name'/> before key words to trigger synchronized animation events.
- Match target_duration_s (~2.5 spoken words/sec): write crisp scenes aligned to natural speaking cadence.
- Objects persist across scenes when persist=true (keep axes, target curves, or geometric shapes on screen!).
- Vary `composition` across scenes (`centered_focus`, `main_with_equation_panel`, `side_by_side_comparison`/`stacked_comparison`, `step_by_step_sequence`, `grid_of_multiple_examples`, `spotlight_zoom`, `timeline_progression`) rather than repeating boxed cards every scene.
- Use `strategy` to state the pedagogical explanation strategy before the storyboard, and `card=true` ONLY on key takeaway equations.
- Prefer deterministic `constructions` (geometry, gradient descent, RK4 ODEs, eigen/matrix transforms, Riemann integrals, arbitrary discrete PMF convolutions, Fourier partial sums, Newton roots, Markov chains) over guessing coordinates!
  * IMPORTANT: Do NOT duplicate `constructions[].output_ids` in `scenes[].objects` (the engine generates them automatically!). Declare the parent `axes` in `objects` and pass `params.axes_id="<axes_id>"` (NEVER include `<axes_id>` inside `output_ids`).
  * For `ode_phase_trajectory`, set `params={dx_expr, dy_expr, x0, y0, dt, steps, axes_id}` and `output_ids=["orbit_1"]` (emits a `data_series`, NOT a separate `parametric_curve`).
  * For any manual `parametric_curve`, ALWAYS set `expr="X(t); Y(t)"` (NEVER leave `expr=null` or put formulas in `label`).
- Every plan MUST include verifiable `claims`:
  * Polygon-ONLY claims (`right_triangle`, `square`, `congruent_triangles`, `area_conservation`, `partition_of`, `nonzero_area`) MUST ONLY reference `kind="polygon"` object_ids (NEVER `particle_system`, `matrix`, `circle`, or `axes`).
  * `point_on_function` requires exact `coords = [x, f(x)]`; if introducing phase-plane trajectories or ODE equilibria, prefer `ode_phase_trajectory` + `ode_equilibrium` (`dx_expr`, `dy_expr`, `x_val`, `expected_value=0.0`).
  * `eigenvalue_determinant` requires `matrix_2x2` + exact determinant `expected_value = a*d - b*c`.
  * `eigenvalue_stability_regime` requires `matrix_2x2` + exact `expected_regime`: asymptotically stable iff ALL `Re(λ) < 0`; unstable iff ANY `Re(λ) > 0` (distinguish `saddle` where eigenvalues have opposite signs from `unstable_node`/`unstable_spiral` where ALL `Re(λ) > 0`; `Re(λ) = 0` is a linear center and non-hyperbolic for nonlinear systems).
  * NEVER emit a bare `axes`-only empty scene — every `axes` MUST have at least one plotted trajectory, curve, vector, polygon, or point with `axes_id="<axes_id>"` (NEVER attach equations/annotations/matrices to `axes_id`).
  * NEVER apply `op="rotate"` to non-geometric objects (`matrix`, `table`, `equation`, `annotation`, `axes`, `bar_chart`).
  * Set `card=true` ONLY on one pivotal equation per chapter (leave `card=false` by default for airy editorial typography).
  * For `markov_chain_evolution`, `transition_matrix` MUST be ROW-stochastic (each row sums to 1.0; if using a PageRank column-stochastic matrix, transpose it so every row sums to 1.0).
  * `stationary_point`, `local_minimum`, `derivative_equality`, `function_equivalence`, `polynomial_degree`, `coefficient_correctness`, `definite_integral_value`, `probability_normalization`, `conditional_probability`, `distribution_stats`, `convergence_assumptions`, `markov_stochastic_invariant`.
Ensure mathematical accuracy: formulas, degrees, plotted expressions, and numerical examples must agree.
"""


def create_v3_plan(
    topic: str | None = None,
    prompt: str | None = None,
    model: str | None = None,
    out_dir: Path = PLANS_DIR,
    provider_name: str = "gemini",
    free_only: bool = True,
    audience: str = "general",
    target_duration_s: float = 55.0,
    narration_mode: str = "single_educator",
    look_name: str = "midnight_studio",
    format_name: str = "widescreen_hd",
    backend: str = "manimgx",
    use_creative_director: bool = True,
    script_ir: object | None = None,
    artifact_dir: Path | None = None,
) -> Path:
    """Generate and validate a composable VideoPlanV3 grounded by CreativeDirector + Registry."""
    from studio.ai.v3_schema import VideoPlanV3
    from studio.ai.validate_v3 import validate_v3_plan
    from studio.ai.world_model import materialize_constructions
    from studio.audio.tts import parse_script
    from studio.creative.director import CreativeDirector
    from studio.design.spec import default_design_spec, export_design_md
    from studio.qa.engine import audit_educational_quality
    from studio.registry import capabilities_prompt_summary
    from studio.storyboard import export_review_md, export_storyboard_md, storyboard_from_v3_plan

    if not topic and not prompt:
        raise ValueError("Give a topic or a prompt.")
    label = prompt or topic or "video"
    slug = slugify(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = artifact_dir or (out_dir / f"v3_{slug}_artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)

    director = GeminiDirector(model=model, provider_name=provider_name, free_only=free_only)
    aspect = "9:16" if "shorts" in format_name else "16:9"
    registry_prompt = capabilities_prompt_summary(backend=backend, aspect=aspect, compact=True)
    system_prompt = f"{V3_STYLE_GUIDE}\n\n{registry_prompt}"

    creative_context = ""
    active_script = script_ir
    if active_script is None and use_creative_director:
        cdir = CreativeDirector(model=model, provider_name=provider_name, free_only=free_only)
        pkg = cdir.produce_creative_package(
            label,
            audience=audience,
            narration_mode=narration_mode,  # type: ignore[arg-type]
            format_name=format_name,
            look_name=look_name,
            backend=backend,
            target_duration_s=target_duration_s,
            artifact_dir=art_dir,
        )
        active_script = pkg.script
        beat_summary = "\n".join(
            f"- {b.scene_id} [speaker={b.speaker}]: {b.spoken_text} | visual_intent={b.visual_intent} | marks={b.anchor_marks} | math={b.latex_refs}"
            for b in pkg.script.beats
        )
        creative_context = (
            f"\n\nAPPROVED CREATIVE STRATEGY & SCRIPT (follow beat count, speakers, narration, and visual intent):\n"
            f"Title: {pkg.script.title}\nStrategy: {pkg.brief.selected_strategy}\nCentral Question: {pkg.brief.concept.central_question}\n"
            f"Beats:\n{beat_summary}\n"
        )
    elif active_script is not None:
        beats = getattr(active_script, "beats", [])
        beat_summary = "\n".join(
            f"- {b.scene_id} [speaker={b.speaker}]: {b.spoken_text} | visual_intent={b.visual_intent} | marks={b.anchor_marks}"
            for b in beats
        )
        creative_context = (
            f"\n\nPROVIDED SCRIPT IR (map each beat 1:1 to a SceneV3 preserving speaker, spoken_text, and marks):\n"
            f"{beat_summary}\n"
        )
        design = default_design_spec(format_name=format_name, look_name=look_name)
        (art_dir / "DESIGN.md").write_text(export_design_md(design), encoding="utf-8")

    min_words = max(45, int(target_duration_s * 1.9))
    max_words = max(min_words + 20, int(target_duration_s * 2.28))
    req = (
        f"{system_prompt}\n\nTopic/Concept: {label}\nAudience: {audience}\n"
        f"Format aspect: {aspect} ({format_name}), Narration mode: {narration_mode}, Look: {look_name}.\n"
        f"Target duration: {target_duration_s:.0f}s (TARGET WORD BUDGET: {min_words}-{max_words} total spoken words; do NOT exceed {max_words} words).\n"
        f"{creative_context}\n"
        "Generate a complete VideoPlanV3 JSON preserving beat IDs (`scene_id`), `speaker`, spoken text, deterministic constructions, varied compositions, and verifiable claims."
    )
    plan, resp = director.provider.generate_structured(req, VideoPlanV3, system=system_prompt, temperature=0.45)
    director._log(f"{resp.provider}/{resp.model}: V3 plan generated in {resp.latency_s:.1f}s (tokens: {resp.prompt_tokens}+{resp.completion_tokens})")

    raw_path = out_dir / f"v3_{slug}_raw.json"
    raw_path.write_text(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

    repair_log: list[dict] = []
    for round_idx in range(1, MAX_REPAIRS + 1):
        issues = validate_v3_plan(plan, require_claims=True)
        total_words = sum(len(parse_script(s.narration)[0].split()) for s in plan.scenes)
        if total_words < min_words:
            issues.append(f"Narration too short for {target_duration_s:.0f}s target ({total_words} words < {min_words}); expand explanations of WHY each step works.")
        elif total_words > max_words:
            issues.append(f"Narration exceeds target duration envelope ({total_words} words > {max_words}); tighten phrasing slightly without losing math rigor.")
        if not issues:
            break
        repair_log.append({"round": round_idx, "issues": issues})
        director._log(f"V3 repair round {round_idx}/{MAX_REPAIRS} ({len(issues)} issues)")
        fix_prompt = "Fix these VideoPlanV3 issues:\n- " + "\n- ".join(issues) + f"\n\nCurrent JSON:\n{plan.model_dump_json()}"
        plan, _ = director.provider.generate_structured(fix_prompt, VideoPlanV3, system=system_prompt, temperature=0.3)

    issues = validate_v3_plan(plan, require_claims=True)
    if issues:
        if repair_log:
            (out_dir / f"v3_{slug}_repairs.json").write_text(json.dumps(repair_log, indent=2), encoding="utf-8")
        raise DirectorError("VideoPlanV3 still invalid after repairs:\n- " + "\n- ".join(issues))

    if repair_log:
        (out_dir / f"v3_{slug}_repairs.json").write_text(json.dumps(repair_log, indent=2), encoding="utf-8")

    plan = materialize_constructions(plan)
    if active_script is not None:
        from studio.creative.script_ir import export_script_md, lock_script_to_plan

        active_script = lock_script_to_plan(plan, active_script, stage="editorial_repair")  # type: ignore[arg-type]
        (art_dir / "SCRIPT.md").write_text(export_script_md(active_script), encoding="utf-8")  # type: ignore[arg-type]

    sb = storyboard_from_v3_plan(plan, format_name=format_name, look_name=look_name, script=active_script)  # type: ignore[arg-type]
    export_storyboard_md(sb, art_dir / "STORYBOARD.md")
    rev = audit_educational_quality(
        topic=plan.topic,
        title=plan.title,
        plan=plan,
        format_name=format_name,
        backend=backend,
        script=active_script,
        look_name=look_name,
    )
    export_review_md(rev, art_dir / "REVIEW.md")  # type: ignore[arg-type]

    path = out_dir / f"v3_{slug}.json"
    path.write_text(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
