"""Renderer-independent Script Intermediate Representation (ScriptIR + SCRIPT.md)."""

from __future__ import annotations

import re
from typing import Literal
from pydantic import BaseModel, Field

from studio.audio.pronounce import normalize_for_speech_with_map
from studio.audio.tts import BeatSpec, BOOKMARK_RE, parse_script
from studio.creative.dialogue import DIALOGUE_PROFILES, clean_spoken_text, detect_banned_cliches
from studio.creative.models import ConceptAnalysis, ExplanationStrategyKind, NarrationMode, SpeakerRole


class NarrationBeatIR(BaseModel):
    """Single renderer-independent narrative beat linked to visual intent and mathematical claims."""

    beat_id: str
    scene_id: str
    shot_id: str = ""
    speaker: SpeakerRole = "narrator"
    spoken_text: str
    displayed_caption: str = ""
    latex_refs: list[str] = Field(default_factory=list)
    educational_purpose: str
    mathematical_claims: list[str] = Field(default_factory=list)
    referenced_node_ids: list[str] = Field(default_factory=list)
    emotional_tone: str = "curious"
    emphasis_words: list[str] = Field(default_factory=list)
    pause_after_s: float = 0.25
    visual_intent: str
    anchor_marks: list[str] = Field(default_factory=list)
    estimated_duration_s: float = 4.5
    prerequisites: list[str] = Field(default_factory=list)


class ScriptRevision(BaseModel):
    revision: int
    stage: Literal["initial", "editorial_repair", "manual"]
    notes: str = ""
    beats: list[NarrationBeatIR] = Field(default_factory=list)


class ScriptIR(BaseModel):
    """Authoritative pre-render educational script decoupled from ManimGX or HyperFrames."""

    topic: str
    title: str
    audience: str = "general"
    narration_mode: NarrationMode = "single_educator"
    strategy: ExplanationStrategyKind = "concrete_then_abstract"
    central_question: str = ""
    concept_analysis: ConceptAnalysis | None = None
    beats: list[NarrationBeatIR] = Field(default_factory=list)
    revisions: list[ScriptRevision] = Field(default_factory=list)


def validate_script_ir(script: ScriptIR) -> list[str]:
    """Deterministically validate ScriptIR quality, speaker consistency, and math/spoken grounding."""
    issues: list[str] = []
    if not script.beats:
        return ["ScriptIR must contain at least one NarrationBeatIR."]
    profile = DIALOGUE_PROFILES.get(script.narration_mode, DIALOGUE_PROFILES["single_educator"])
    allowed_speakers = {s.role for s in profile.speakers}
    seen_ids: set[str] = set()

    for idx, b in enumerate(script.beats, 1):
        if not b.beat_id or b.beat_id in seen_ids:
            issues.append(f"beat {idx}: beat_id must be non-empty and unique (got {b.beat_id!r}).")
        seen_ids.add(b.beat_id)
        if b.speaker not in allowed_speakers:
            issues.append(f"beat {b.beat_id}: speaker {b.speaker!r} invalid for mode {script.narration_mode!r}.")
        txt = clean_spoken_text(b.spoken_text)
        if not txt:
            issues.append(f"beat {b.beat_id}: spoken_text must not be empty.")
        if re.search(r"\\(?:frac|sum|int|alpha|beta|gamma|theta|sqrt|cdot|times)\b", txt):
            issues.append(f"beat {b.beat_id}: spoken_text contains raw LaTeX macro; write speakable prose.")
        cliches = detect_banned_cliches(txt)
        if cliches:
            issues.append(f"beat {b.beat_id}: contains banned filler/cliche {cliches}.")
        clean_words = parse_script(txt)[0].split()
        if len(clean_words) > profile.max_turn_words:
            issues.append(f"beat {b.beat_id}: {len(clean_words)} words exceeds max {profile.max_turn_words}.")
        marks_found = set(BOOKMARK_RE.findall(txt))
        missing_marks = [m for m in b.anchor_marks if m not in marks_found]
        if missing_marks:
            issues.append(f"beat {b.beat_id}: anchor_marks missing <bookmark/> tags in spoken_text: {missing_marks}.")

    if len(profile.speakers) > 1:
        used_speakers = {b.speaker for b in script.beats}
        if len(used_speakers) < 2:
            issues.append(f"dialogue mode {script.narration_mode!r} requires at least 2 distinct speakers.")
    return issues


def script_ir_to_beat_specs(script: ScriptIR) -> list[BeatSpec]:
    """Convert ScriptIR into BeatSpecs while keeping caption, spoken map, and math explicitly linked."""
    specs: list[BeatSpec] = []
    for b in script.beats:
        cap = b.displayed_caption.strip() or parse_script(b.spoken_text)[0]
        # Verify spoken pronunciation normalization succeeds cleanly
        _ = normalize_for_speech_with_map(b.spoken_text)
        specs.append(
            BeatSpec(
                id=b.beat_id,
                narration=b.spoken_text,
                caption=cap,
                pause_after=b.pause_after_s,
                speaker=b.speaker,
            )
        )
    return specs


def export_script_md(script: ScriptIR) -> str:
    """Export human-readable SCRIPT.md preserving metadata, visual links, and spoken normalization."""
    lines = [
        f"# SCRIPT — {script.title}",
        "",
        f"- **Topic:** {script.topic}",
        f"- **Strategy:** `{script.strategy}` | **Narration Mode:** `{script.narration_mode}` | **Audience:** `{script.audience}`",
        f"- **Central Question:** {script.central_question or script.topic}",
        f"- **Revisions Preserved:** {len(script.revisions)}",
        "",
    ]
    for b in script.beats:
        mapped = normalize_for_speech_with_map(b.spoken_text)
        lines.extend([
            f"## Beat `{b.beat_id}` (Scene: `{b.scene_id}`, Speaker: `{b.speaker}`, ~{b.estimated_duration_s:.1f}s)",
            f"- **Purpose:** {b.educational_purpose}",
            f"- **Visual Intent:** {b.visual_intent}",
            f"- **Spoken Text:** {b.spoken_text}",
            f"- **TTS Normalized:** {mapped['spoken']}",
        ])
        if b.displayed_caption:
            lines.append(f"- **Caption:** {b.displayed_caption}")
        if b.latex_refs:
            lines.append(f"- **Equations / Math:** {', '.join(f'`{e}`' for e in b.latex_refs)}")
        if b.mathematical_claims:
            lines.append(f"- **Claims:** {'; '.join(b.mathematical_claims)}")
        if b.anchor_marks:
            lines.append(f"- **Bookmarks:** {', '.join(b.anchor_marks)}")
        lines.append("")
    return "\n".join(lines)
