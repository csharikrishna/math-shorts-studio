"""Strict plan schema: Gemini's structured output and the scene template's only input."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from studio.audio.tts import BOOKMARK_RE, parse_script  # noqa: F401 - single source of truth for script markup

SfxCue = Literal["pop_marimba", "whoosh_soft", "tick_soft", "chime_reveal", "chord_epiphany"]
ColorRole = Literal["primary", "secondary", "tertiary", "success", "highlight", "danger"]
Template = Literal["graph", "parametric", "sequence", "surface", "polytope4d"]
Readout = Literal["value", "slope", "error", "none"]
CueAction = Literal["flash_point", "indicate_formula", "sweep_curve", "pulse_badge"]
Polytope = Literal["tesseract", "16-cell", "5-cell"]
RotationPlane = Literal["xw", "yw", "zw", "xy", "xz", "yz"]
EdgeHighlight = Literal["none", "inner", "outer", "connectors"]
TEMPLATES_3D = ("surface", "polytope4d")

MAX_NARRATION_WORDS = 25


class Feasibility(BaseModel):
    """Honesty gate: can this concept be shown faithfully with a renderable template?"""

    fits: bool = Field(description="True only if a template can show the real mathematics, not a stand-in.")
    template: Template = Field(description="Best template for the concept.")
    reason: str = Field(description="One sentence: why it fits, or what the templates cannot draw.")
    honest_alternative: str = Field(
        description="If it does not fit: the closest concept the templates CAN show truthfully; else empty."
    )


class Concept(BaseModel):
    """A candidate idea from the hook selector."""

    title: str = Field(description="Punchy on-screen title, at most 28 characters.")
    hook_question: str = Field(description="Counter-intuitive opening line, under 14 words.")
    why_viral: str = Field(description="One sentence: the surprise or payoff that keeps viewers watching.")
    template: Template
    visual_fit: int = Field(description="1-10: how well the idea is shown with the chosen template.")
    surprise: int = Field(description="1-10: how counter-intuitive the hook is.")


class ConceptSlate(BaseModel):
    concepts: list[Concept]


class AxisLabel(BaseModel):
    value: float = Field(description="Position along the x axis (plot coordinates).")
    tex: str = Field(description="Short LaTeX label, e.g. '\\pi' or '1'.")


class CueSpec(BaseModel):
    mark: str = Field(description="Name of a <bookmark mark='...'/> placed in this beat's narration.")
    action: CueAction


class VisualIntent(BaseModel):
    """What the stage shows during one beat (the template renders it with continuous motion)."""

    expr: Optional[str] = Field(
        default=None,
        description=(
            "Main object for this beat in the safe math grammar. graph: f(x). parametric: 'X(t); Y(t)'. "
            "sequence: a(n). surface: f(x, y). null keeps the previous beat's object."
        ),
    )
    t_range: Optional[list[float]] = Field(default=None, description="parametric only: [t_min, t_max].")
    n_max: Optional[int] = Field(default=None, description="sequence only: reveal terms n = 1..n_max (<= 40).")
    y_range: Optional[list[float]] = Field(
        default=None,
        description="graph/sequence only: zoom the vertical window to [min, max] from this beat on (e.g. into a tiny gap).",
    )
    slice_axis: Optional[Literal["x", "y"]] = Field(default=None, description="surface only: draw the f(x,0) or f(0,y) slice.")
    rotation_plane: Optional[RotationPlane] = Field(
        default=None,
        description="polytope4d only: 4D plane to rotate in from this beat on ('xw','yw','zw' turn through the 4th dimension).",
    )
    rotation_speed: Optional[float] = Field(
        default=None, description="polytope4d only: turns per second in that plane, 0 to 0.5 (0 = hold still)."
    )
    highlight: Optional[EdgeHighlight] = Field(
        default=None, description="polytope4d only: emphasize the inner cell, outer cell, or the connector edges."
    )
    show_error_area: bool = Field(default=False, description="graph only: shade the gap between expr and reference_expr.")
    readout: Readout = Field(default="value", description="Live number in the badge: value, slope, error vs reference, or none.")
    badge_label: str = Field(description="Badge text, at most 14 characters, e.g. 'Order 3' or 'N = 19'.")
    cues: list[CueSpec] = Field(default_factory=list, description="Accents fired exactly when a bookmark word is spoken.")


class BeatPlan(BaseModel):
    beat_id: str = Field(description="snake_case id, unique.")
    narration: str = Field(
        description=(
            "Spoken line, conversational and clear, at most 25 words. Explain the concept step clearly "
            "as if talking to a curious 16-year-old. Wrap 1-2 key words in *asterisks* for caption "
            "emphasis. Optionally insert <bookmark mark='name'/> right before a word to time a cue."
        )
    )
    caption: str = Field(description="Subtitle-file summary of the beat, at most 48 characters.")
    formula: str = Field(description="LaTeX for the header formula during this beat (no $ delimiters).")
    color_role: ColorRole
    sfx_cue: SfxCue
    visual_intent: VisualIntent


class ShortPlan(BaseModel):
    """A complete 45-90 second well-explained math Short."""

    topic: str
    tag: str = Field(description="Field label, at most 24 characters, e.g. 'Real Analysis'.")
    title: str = Field(description="On-screen title, at most 28 characters.")
    template: Template
    x_range: list[float] = Field(description="[min, max] plot window for x (or n for sequences, or u for surfaces).")
    y_range: list[float] = Field(description="[min, max] plot window for y (or v for surfaces).")
    z_range: Optional[list[float]] = Field(default=None, description="surface only: [min, max] for z.")
    polytope: Optional[Polytope] = Field(
        default=None, description="polytope4d only: which regular 4D polytope to project (x/y ranges are ignored)."
    )
    reference_expr: Optional[str] = Field(
        default=None, description="Fixed target shown from frame 0 (graph/sequence: function of x; e.g. a limit line '1')."
    )
    x_labels: list[AxisLabel] = Field(default_factory=list, description="Up to 3 landmark labels under the x axis.")
    beats: list[BeatPlan] = Field(description="8 to 12 beats; the first is the hook, the last the payoff. Each beat explains ONE concept step clearly.")


def strip_markup(narration: str) -> str:
    """Spoken text: bookmarks removed, *emphasis* asterisks dropped, whitespace normalized."""
    return parse_script(narration)[0]


def spoken_word_count(narration: str) -> int:
    return len(strip_markup(narration).split())
