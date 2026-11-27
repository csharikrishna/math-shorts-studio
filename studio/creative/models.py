"""Structured creative intelligence models separated cleanly from scene rendering."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

ExplanationStrategyKind = Literal[
    "concrete_then_abstract",
    "visual_proof",
    "thought_experiment",
    "simulation",
    "problem_solution",
    "misconception_correction",
    "comparison",
    "progressive_construction",
    "cause_and_effect",
    "question_and_answer",
]

NarrationMode = Literal[
    "single_educator",
    "conversational_explainer",
    "socratic",
    "discussion",
    "documentary",
    "technical_instructor",
]

SpeakerRole = Literal["narrator", "educator", "learner", "host_a", "host_b"]


class ConceptAnalysis(BaseModel):
    """Deep pedagogical analysis performed before scripting or choosing visuals."""

    topic: str
    meaning: str
    prerequisites: list[str] = Field(default_factory=list)
    central_question: str
    essential_definitions: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)
    mathematical_relationships: list[str] = Field(default_factory=list)
    real_world_examples: list[str] = Field(default_factory=list)
    learning_outcome: str


class StorylineOption(BaseModel):
    """One distinct narrative treatment (device + form + strategy), adapted from HyperFrames."""

    option_id: str
    name: str
    strategy: ExplanationStrategyKind
    narrative_device: str
    visual_form: str
    opening_two_seconds: str
    beat_arc: str
    recommended: bool = False
    selection_reason: str = ""
    fit_score: float = 0.8


class CreativeBrief(BaseModel):
    """Consolidated ConceptAnalysis + explored storylines + selected pedagogical strategy."""

    concept: ConceptAnalysis
    storylines: list[StorylineOption] = Field(default_factory=list)
    selected_storyline_id: str
    selected_strategy: ExplanationStrategyKind
    narration_mode: NarrationMode = "single_educator"
    audience: str = "general"


class EditorialIssue(BaseModel):
    beat_id: str
    category: Literal[
        "mathematical_accuracy",
        "scientific_accuracy",
        "natural_speech",
        "clarity",
        "progression",
        "repetition",
        "jargon",
        "unsupported_claim",
        "missing_prerequisite",
        "excessive_abstraction",
        "filler_cliche",
    ]
    detail: str
    suggested_fix: str = ""


class EditorialReviewReport(BaseModel):
    approved: bool
    math_accuracy_ok: bool = True
    clarity_ok: bool = True
    progression_ok: bool = True
    natural_speech_ok: bool = True
    issues: list[EditorialIssue] = Field(default_factory=list)
    summary: str = ""


def export_storylines_md(brief: CreativeBrief) -> str:
    """Export human-readable STORYLINES.md recording concept analysis and alternatives."""
    c = brief.concept
    lines = [
        f"# STORYLINES — {c.topic}",
        "",
        "## Concept Analysis",
        f"- **Meaning:** {c.meaning}",
        f"- **Central Question:** {c.central_question}",
        f"- **Learning Outcome:** {c.learning_outcome}",
        f"- **Prerequisites:** {', '.join(c.prerequisites) or 'None assumed'}",
        f"- **Essential Definitions:** {', '.join(c.essential_definitions) or 'Core terms introduced in context'}",
        f"- **Common Misconceptions:** {'; '.join(c.common_misconceptions) or 'None flagged'}",
        f"- **Mathematical Relationships:** {'; '.join(c.mathematical_relationships) or 'Core identities'}",
        f"- **Real-World Grounding:** {'; '.join(c.real_world_examples) or 'Concrete worked example'}",
        f"- **Narration Mode:** `{brief.narration_mode}` | **Audience:** `{brief.audience}`",
        "",
        "## Candidate Storylines & Selected Strategy",
        "",
    ]
    for opt in brief.storylines:
        tag = "SELECTED" if opt.option_id == brief.selected_storyline_id else ("RECOMMENDED" if opt.recommended else "ALTERNATIVE")
        lines.extend([
            f"### {opt.option_id}: {opt.name} [{tag}]",
            f"- **Explanation Strategy:** `{opt.strategy}` (fit={opt.fit_score:.2f})",
            f"- **Narrative Device:** {opt.narrative_device}",
            f"- **Visual Form:** {opt.visual_form}",
            f"- **Opening Two Seconds:** {opt.opening_two_seconds}",
            f"- **Beat Arc:** {opt.beat_arc}",
            f"- **Rationale:** {opt.selection_reason or 'Chosen for strongest visual-to-concept unity.'}",
            "",
        ])
    return "\n".join(lines)
