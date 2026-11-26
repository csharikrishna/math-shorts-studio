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
Think through: concept -> prerequisites -> central discovery -> visual objects -> narration -> animation events.
Objects persist across scenes when persist=true (keep axes, target curves, or geometric shapes on screen!).
Object kinds:
- axes (x_range, y_range), function_plot (expr, axes_id), parametric_curve (expr="X(t);Y(t)", t_range, axes_id),
- bar_chart (values, bar_labels, log_scale), equation (tex), point (coords, axes_id),
- line / vector (coords, end_coords, axes_id), polygon (vertices), circle (coords, radius), number_line (x_range), annotation (text, coords).
Events reference object ids: op in {create, draw, transform, highlight, move, scale, rotate, remove, focus}, optional source_id (for transform), optional anchor_mark matching <bookmark mark='...'/> in narration.
Create 5 to 8 progressive scenes (hook -> intuition -> construction -> discovery -> payoff) with 12-22 words per narration.
Ensure mathematical accuracy: formulas, degrees, and plotted expressions must agree.
"""


def create_v3_plan(
    topic: str | None = None,
    prompt: str | None = None,
    model: str | None = None,
    out_dir: Path = PLANS_DIR,
    provider_name: str = "gemini",
    free_only: bool = True,
) -> Path:
    """Generate and validate a composable object-and-event VideoPlanV3 from a topic or prompt."""
    from studio.ai.v3_schema import VideoPlanV3
    from studio.ai.validate_v3 import validate_v3_plan

    if not topic and not prompt:
        raise ValueError("Give a topic or a prompt.")
    label = prompt or topic or "video"
    director = GeminiDirector(model=model, provider_name=provider_name, free_only=free_only)
    req = f"{V3_STYLE_GUIDE}\n\nTopic/Concept: {label}\nGenerate a complete VideoPlanV3 JSON."
    plan, resp = director.provider.generate_structured(req, VideoPlanV3, system=V3_STYLE_GUIDE, temperature=0.5)
    director._log(f"{resp.provider}/{resp.model}: V3 plan generated in {resp.latency_s:.1f}s (tokens: {resp.prompt_tokens}+{resp.completion_tokens})")

    for round_idx in range(1, MAX_REPAIRS + 1):
        issues = validate_v3_plan(plan)
        if not issues:
            break
        director._log(f"V3 repair round {round_idx}/{MAX_REPAIRS} ({len(issues)} issues)")
        fix_prompt = "Fix these VideoPlanV3 issues:\n- " + "\n- ".join(issues) + f"\n\nCurrent JSON:\n{plan.model_dump_json()}"
        plan, _ = director.provider.generate_structured(fix_prompt, VideoPlanV3, system=V3_STYLE_GUIDE, temperature=0.3)

    issues = validate_v3_plan(plan)
    if issues:
        raise DirectorError("VideoPlanV3 still invalid after repairs:\n- " + "\n- ".join(issues))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"v3_{slugify(label)}.json"
    path.write_text(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
