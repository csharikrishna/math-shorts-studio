"""Static validation of AI plans before any TTS or GPU time is spent.

Every problem is phrased as an instruction, so the list can be sent straight back to Gemini
as a repair prompt.
"""

from __future__ import annotations

import re

import numpy as np

from studio.ai.expr import ExprError, compile_expr, finite_fraction
from studio.ai.schema import BOOKMARK_RE, MAX_NARRATION_WORDS, ShortPlan, spoken_word_count

TITLE_MAX = 28
TAG_MAX = 24
BADGE_MAX = 14
CAPTION_MAX = 48
MIN_BEATS, MAX_BEATS = 6, 13
SEQ_MAX_TERMS = 40
MAX_EMPHASIS_WORDS = 2
EMPHASIS_SPAN_RE = re.compile(r"\*([^*]+)\*")
VARS = {"graph": ("x",), "parametric": ("t",), "sequence": ("n",), "surface": ("x", "y"), "polytope4d": ()}


def _range_ok(r: list[float] | None, name: str, problems: list[str]) -> bool:
    if r is None or len(r) != 2 or not np.all(np.isfinite(r)) or r[0] >= r[1]:
        problems.append(f"{name} must be [min, max] with min < max.")
        return False
    return True


def _check_tex(tex: str, where: str, problems: list[str]) -> None:
    # The typesetter silently auto-closes groups, so check balance explicitly
    bare = tex.replace(r"\{", "").replace(r"\}", "")
    depth = 0
    for ch in bare:
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            break
    if depth != 0:
        problems.append(f"{where}: LaTeX {tex!r} has unbalanced braces.")
        return
    if "$" in tex:
        problems.append(f"{where}: LaTeX {tex!r} must not contain $ delimiters.")
        return
    try:
        import manimgx as m

        m.MathTex(tex)
    except Exception as exc:  # noqa: BLE001 - any typesetting failure is a plan error
        problems.append(f"{where}: LaTeX {tex!r} does not typeset ({type(exc).__name__}). Simplify it.")


def _inside(vals: np.ndarray, lo: float, hi: float) -> float:
    vals = np.asarray(vals, dtype=float)
    ok = np.isfinite(vals) & (vals >= lo) & (vals <= hi)
    return float(np.mean(ok)) if vals.size else 0.0


def check_expr(
    plan: ShortPlan, source: str, where: str, problems: list[str], *, t_range=None, n_max=None, y_window=None
) -> None:
    template = plan.template
    try:
        f = compile_expr(source.split(";")[0] if template != "parametric" else source.split(";")[0], VARS[template])
        g = compile_expr(source.split(";")[1], ("t",)) if template == "parametric" else None
    except (ExprError, IndexError) as exc:
        hint = "write 'X(t); Y(t)' with a semicolon" if template == "parametric" else str(exc)
        problems.append(f"{where}: expression {source!r} is invalid: {hint}.")
        return
    x0, x1 = plan.x_range
    y0, y1 = y_window or plan.y_range
    try:
        if template == "graph":
            xs = np.linspace(x0, x1, 600)
            ys = f(x=xs)
            if finite_fraction(ys) < 0.85:
                problems.append(f"{where}: {source!r} is undefined on most of x_range; narrow x_range.")
            elif _inside(ys, y0, y1) < 0.45:
                problems.append(f"{where}: {source!r} leaves the y window {[y0, y1]} on most of x_range; adjust y_range.")
        elif template == "parametric":
            lo, hi = t_range or [0.0, 2.0 * np.pi]
            ts = np.linspace(lo, hi, 600)
            xs, ys = f(t=ts), g(t=ts)  # type: ignore[misc]
            inside = np.mean((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1))
            if inside < 0.9:
                problems.append(f"{where}: curve {source!r} leaves the plot window; widen x_range/y_range.")
        elif template == "sequence":
            ns = np.arange(1, (n_max or 10) + 1, dtype=float)
            vals = f(n=ns)
            visible = int(np.sum(np.isfinite(vals) & (vals >= y0) & (vals <= y1)))
            if visible < min(3, len(ns)):
                problems.append(f"{where}: terms of {source!r} mostly fall outside the y window {[y0, y1]}.")
        elif template == "surface":
            z = plan.z_range or [-2.0, 2.0]
            gx, gy = np.meshgrid(np.linspace(x0, x1, 40), np.linspace(y0, y1, 40))
            vals = f(x=gx, y=gy)
            if _inside(vals, z[0], z[1]) < 0.7:
                problems.append(f"{where}: surface {source!r} exceeds z_range {plan.z_range}; widen z_range.")
    except ExprError as exc:
        problems.append(f"{where}: {exc}.")


def validate_plan(plan: ShortPlan) -> list[str]:
    """Return a list of repair instructions; empty means the plan is renderable."""
    problems: list[str] = []
    if len(plan.title) > TITLE_MAX:
        problems.append(f"title is {len(plan.title)} chars; shorten to <= {TITLE_MAX}.")
    if len(plan.tag) > TAG_MAX:
        problems.append(f"tag is {len(plan.tag)} chars; shorten to <= {TAG_MAX}.")
    if not (MIN_BEATS <= len(plan.beats) <= MAX_BEATS):
        problems.append(f"use {MIN_BEATS}-{MAX_BEATS} beats (got {len(plan.beats)}).")
    ok = _range_ok(plan.x_range, "x_range", problems) & _range_ok(plan.y_range, "y_range", problems)
    if plan.template == "surface":
        ok &= _range_ok(plan.z_range, "z_range", problems)
    if not ok:
        return problems

    if plan.reference_expr:
        if plan.template in ("graph", "sequence"):
            try:
                ref = compile_expr(plan.reference_expr, ("x",))
                vals = ref(x=np.linspace(*plan.x_range, 400))
                if _inside(vals, *plan.y_range) < 0.45:
                    problems.append(f"reference_expr {plan.reference_expr!r} mostly leaves y_range.")
            except ExprError as exc:
                problems.append(f"reference_expr: {exc} (it is a function of x).")
        else:
            problems.append(f"reference_expr is only used by graph/sequence templates; set it to null for {plan.template}.")

    if len(plan.x_labels) > 3:
        problems.append("use at most 3 x_labels.")
    for lbl in plan.x_labels:
        if not (plan.x_range[0] <= lbl.value <= plan.x_range[1]):
            problems.append(f"x_label {lbl.tex!r} at {lbl.value} is outside x_range.")
        _check_tex(lbl.tex, "x_labels", problems)

    is_polytope = plan.template == "polytope4d"
    if is_polytope and not plan.polytope:
        problems.append("polytope4d needs polytope: 'tesseract', '16-cell' or '5-cell'.")
    if plan.polytope and not is_polytope:
        problems.append(f"polytope is only used by the polytope4d template; set it to null for {plan.template}.")

    seen: set[str] = set()
    window = list(plan.y_range)
    current_expr: str | None = None
    current_n: int | None = None
    has_object = bool(plan.reference_expr) or is_polytope  # a polytope is on screen from frame 0
    for i, beat in enumerate(plan.beats):
        where = f"beat {i + 1} ({beat.beat_id})"
        if beat.beat_id in seen or not beat.beat_id.replace("_", "").isalnum():
            problems.append(f"{where}: beat_id must be unique snake_case.")
        seen.add(beat.beat_id)
        words = spoken_word_count(beat.narration)
        if words > MAX_NARRATION_WORDS:
            problems.append(f"{where}: narration has {words} words; cut to <= {MAX_NARRATION_WORDS}.")
        if words < 3:
            problems.append(f"{where}: narration is too short to carry a beat.")
        if len(beat.caption) > CAPTION_MAX:
            problems.append(f"{where}: caption is {len(beat.caption)} chars; shorten to <= {CAPTION_MAX}.")
        vi = beat.visual_intent
        if vi.y_range is not None:
            if plan.template not in ("graph", "sequence"):
                problems.append(f"{where}: y_range zoom is only for graph/sequence templates.")
            elif _range_ok(vi.y_range, f"{where} y_range", problems):
                window = list(vi.y_range)
                if current_expr and not vi.expr:
                    check_expr(plan, current_expr, f"{where} (zoomed)", problems, n_max=current_n, y_window=window)
        if len(vi.badge_label) > BADGE_MAX:
            problems.append(f"{where}: badge_label {vi.badge_label!r} is too long; <= {BADGE_MAX} chars.")
        marks = set(BOOKMARK_RE.findall(beat.narration))
        for cue in vi.cues:
            if cue.mark not in marks:
                problems.append(f"{where}: cue mark {cue.mark!r} has no <bookmark mark='{cue.mark}'/> in the narration.")
        _check_tex(beat.formula, f"{where} formula", problems)
        if is_polytope:
            if vi.expr:
                problems.append(f"{where}: polytope4d draws the polytope itself; set expr to null.")
            if vi.readout == "error":
                problems.append(f"{where}: polytope4d readouts are value (rotation angle) or slope (spin rate).")
            if vi.rotation_speed is not None and not (0.0 <= vi.rotation_speed <= 0.5):
                problems.append(f"{where}: rotation_speed must be between 0 and 0.5 turns per second.")
        elif vi.rotation_plane or vi.rotation_speed is not None or vi.highlight:
            problems.append(f"{where}: rotation_plane/rotation_speed/highlight are only for polytope4d.")
        if vi.expr and not is_polytope:
            has_object = True
            if plan.template == "sequence" and vi.n_max is not None and not (2 <= vi.n_max <= SEQ_MAX_TERMS):
                problems.append(f"{where}: n_max must be 2..{SEQ_MAX_TERMS}.")
            current_expr = vi.expr
            current_n = vi.n_max or current_n
            check_expr(plan, vi.expr, where, problems, t_range=vi.t_range, n_max=current_n, y_window=window)
        if i == 0 and not has_object:
            problems.append("beat 1 needs visual_intent.expr (or set reference_expr) so the hook visual shows at frame 0.")
        if vi.readout == "error" and not plan.reference_expr:
            problems.append(f"{where}: readout 'error' needs reference_expr.")
        if vi.show_error_area and (plan.template != "graph" or not plan.reference_expr):
            problems.append(f"{where}: show_error_area needs the graph template and a reference_expr.")
        if vi.slice_axis and plan.template != "surface":
            problems.append(f"{where}: slice_axis is only for the surface template.")
        for span in EMPHASIS_SPAN_RE.findall(beat.narration):
            if len(span.split()) > MAX_EMPHASIS_WORDS:
                problems.append(f"{where}: emphasis *{span}* is {len(span.split())} words; emphasize 1-2 key words only.")
    problems += quality_problems(plan)
    return problems


def quality_problems(plan: ShortPlan) -> list[str]:
    """Storytelling checks that keep weaker (lite) models honest; phrased as fixes."""
    out: list[str] = []
    hook, payoff = plan.beats[0], plan.beats[-1]
    if _norm_tex(hook.formula) == _norm_tex(payoff.formula) or (
        "?" not in hook.formula and _norm_tex(payoff.formula) in _norm_tex(hook.formula)
    ):
        out.append(
            "beat 1 formula already states the payoff; pose it as a question instead "
            "(e.g. use \\overset{?}{=}) and save the answer for the last beat."
        )
    if plan.template == "sequence" and plan.reference_expr and not any(b.visual_intent.y_range for b in plan.beats):
        try:
            limit = float(compile_expr(plan.reference_expr, ("x",))(x=np.array(1.0)))
            expr = next((b.visual_intent.expr for b in reversed(plan.beats) if b.visual_intent.expr), None)
            n_max = max((b.visual_intent.n_max or 0) for b in plan.beats) or 10
            if expr and np.isfinite(limit):
                vals = compile_expr(expr, ("n",))(n=np.arange(1, n_max + 1, dtype=float))
                height = plan.y_range[1] - plan.y_range[0]
                crowded = int(np.sum(np.abs(vals - limit) < 0.02 * height))
                if crowded >= max(3, n_max // 2):
                    out.append(
                        f"{crowded} of {n_max} terms sit on the limit line at this scale, so convergence is invisible; "
                        "add visual_intent.y_range zoom beats that dive into the shrinking gap."
                    )
        except ExprError:
            pass  # expression errors are reported by the main checks
    return out


def _norm_tex(tex: str) -> str:
    return "".join(tex.split()).replace(r"\dots", "...").replace(r"\ldots", "...").replace(r"\equiv", "=")
