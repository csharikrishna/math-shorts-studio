"""Renderer-independent Script Intermediate Representation with immutable approved revisions."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal
from pydantic import BaseModel, Field

from studio.audio.pronounce import normalize_for_speech_with_map
from studio.audio.tts import BOOKMARK_RE, BeatSpec, parse_script
from studio.creative.dialogue import (
    DIALOGUE_PROFILES,
    clean_spoken_text,
    detect_banned_cliches,
    detect_unpronounced_identifiers,
)
from studio.creative.models import ConceptAnalysis, ExplanationStrategyKind, NarrationMode, SpeakerRole


class NarrationBeatIR(BaseModel):
    """Single renderer-independent narrative beat linked to visual intent and mathematical claims."""

    beat_id: str
    scene_id: str
    shot_id: str = ""
    chapter_id: str = ""
    speaker: SpeakerRole = "narrator"
    spoken_text: str
    displayed_caption: str = ""
    latex_refs: list[str] = Field(default_factory=list)
    educational_purpose: str
    mathematical_claims: list[str] = Field(default_factory=list)
    referenced_node_ids: list[str] = Field(default_factory=list)
    symbols_introduced: list[str] = Field(default_factory=list)
    transition_from_prev: str = ""
    emotional_tone: str = "curious"
    emphasis_words: list[str] = Field(default_factory=list)
    pause_after_s: float = 0.25
    visual_intent: str
    anchor_marks: list[str] = Field(default_factory=list)
    anchor_token_offsets: dict[str, int] = Field(default_factory=dict)
    estimated_duration_s: float = 4.5
    prerequisites: list[str] = Field(default_factory=list)


class ScriptRevision(BaseModel):
    revision: int
    stage: Literal["initial", "editorial_repair", "math_repair", "approved", "manual"]
    notes: str = ""
    approved: bool = False
    content_sha256: str = ""
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
    approved_revision: int | None = None
    locked: bool = False


def compute_script_sha256(beats: list[NarrationBeatIR]) -> str:
    payload = [
        {"beat_id": b.beat_id, "scene_id": b.scene_id, "speaker": b.speaker, "spoken_text": b.spoken_text}
        for b in beats
    ]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def split_anchors_and_spoken_text(text_with_marks: str) -> tuple[str, dict[str, int]]:
    """Separate `<bookmark mark='...'/>` visual anchors cleanly from authoritative spoken text."""
    clean, _tokens, mark_idx = parse_script(text_with_marks)
    return clean, mark_idx


def inject_anchors_preserving_words(plain_text: str, anchor_offsets: dict[str, int]) -> str:
    """Attach visual `<bookmark/>` anchors at word boundaries without altering a single spoken word."""
    words = plain_text.split()
    if not words or not anchor_offsets:
        return plain_text
    by_idx: dict[int, list[str]] = {}
    for mk, w_idx in anchor_offsets.items():
        pos = max(0, min(len(words), int(w_idx)))
        by_idx.setdefault(pos, []).append(mk)
    out: list[str] = []
    for idx, w in enumerate(words):
        for mk in by_idx.get(idx, []):
            out.append(f"<bookmark mark='{mk}'/>")
        out.append(w)
    for mk in by_idx.get(len(words), []):
        out.append(f"<bookmark mark='{mk}'/>")
    return " ".join(out)


def commit_script_revision(
    script: ScriptIR,
    *,
    stage: Literal["initial", "editorial_repair", "math_repair", "approved", "manual"],
    notes: str = "",
    approve: bool = False,
) -> ScriptRevision:
    """Append an immutable ScriptRevision snapshot; non-approval edits clear locked/approved until re-verified."""
    rev_num = (script.revisions[-1].revision + 1) if script.revisions else 1
    rev = ScriptRevision(
        revision=rev_num,
        stage=stage,
        notes=notes,
        approved=bool(approve),
        content_sha256=compute_script_sha256(script.beats),
        beats=[b.model_copy(deep=True) for b in script.beats],
    )
    script.revisions.append(rev)
    if approve:
        script.approved_revision = rev_num
        script.locked = True
    else:
        script.locked = False
    return rev


def sync_beat_anchors(beat: NarrationBeatIR) -> None:
    """Resolve inline `<bookmark/>` or separate conceptual `anchor_marks` into `anchor_token_offsets`."""
    clean, inline_offsets = split_anchors_and_spoken_text(beat.spoken_text)
    if inline_offsets:
        beat.anchor_token_offsets.update(inline_offsets)
    words = [w.lower().strip(".,!?;:\"'()-") for w in clean.split()]
    n_words = max(1, len(words))
    missing = [m for m in beat.anchor_marks if m not in beat.anchor_token_offsets]
    for idx, mk in enumerate(missing):
        # Match anchor slug/phrase against spoken tokens if present; else distribute evenly
        mk_tokens = [tok for tok in re.split(r"[\s_\-]+", mk.lower()) if tok]
        matched_pos: int | None = None
        if mk_tokens:
            for pos, w in enumerate(words):
                if w.startswith(mk_tokens[0]) or mk_tokens[0] in w:
                    matched_pos = pos
                    break
        if matched_pos is None:
            matched_pos = int(round((idx + 1) * n_words / (len(missing) + 1)))
        beat.anchor_token_offsets[mk] = max(0, min(n_words - 1, matched_pos))


def approve_script_ir(script: ScriptIR, *, notes: str = "Approved after editorial & math checks") -> ScriptIR:
    """Validate and freeze ScriptIR as an immutable approved revision."""
    for b in script.beats:
        sync_beat_anchors(b)
    issues = validate_script_ir(script)
    if issues:
        raise ValueError("Cannot approve invalid ScriptIR:\n- " + "\n- ".join(issues))
    commit_script_revision(script, stage="approved", notes=notes, approve=True)
    return script


def revise_beat_text(
    script: ScriptIR,
    beat_id: str,
    new_spoken_text: str,
    *,
    stage: Literal["editorial_repair", "math_repair", "manual"] = "manual",
    notes: str = "Spoken text revised",
) -> ScriptRevision:
    """Modify spoken text only by recording a new unapproved revision requiring re-review."""
    target = next((b for b in script.beats if b.beat_id == beat_id or b.scene_id == beat_id), None)
    if target is None:
        raise KeyError(f"Unknown beat_id/scene_id {beat_id!r}")
    target.spoken_text = clean_spoken_text(new_spoken_text)
    _, offsets = split_anchors_and_spoken_text(target.spoken_text)
    if offsets:
        target.anchor_token_offsets = offsets
    return commit_script_revision(script, stage=stage, notes=notes, approve=False)


def validate_script_ir(script: ScriptIR) -> list[str]:
    """Deterministically validate ScriptIR quality, speaker consistency, anchors, and speech cleanliness."""
    issues: list[str] = []
    if not script.beats:
        return ["ScriptIR must contain at least one NarrationBeatIR."]
    profile = DIALOGUE_PROFILES.get(script.narration_mode, DIALOGUE_PROFILES["single_educator"])
    allowed_speakers = {s.role for s in profile.speakers}
    seen_ids: set[str] = set()
    seen_stems: set[str] = set()

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
        raw_idents = detect_unpronounced_identifiers(txt)
        if raw_idents:
            issues.append(f"beat {b.beat_id}: spoken_text contains unpronounced code identifier(s) {raw_idents}.")
        cliches = detect_banned_cliches(txt)
        if cliches:
            issues.append(f"beat {b.beat_id}: contains banned filler/cliche {cliches}.")
        clean_words = parse_script(txt)[0].split()
        if len(clean_words) > profile.max_turn_words:
            issues.append(f"beat {b.beat_id}: {len(clean_words)} words exceeds max {profile.max_turn_words}.")
        if len(clean_words) >= 6:
            stem = " ".join(w.lower() for w in clean_words[:6])
            if stem in seen_stems:
                issues.append(f"beat {b.beat_id}: repeats boilerplate opening stem {stem!r}.")
            seen_stems.add(stem)
        marks_found = set(BOOKMARK_RE.findall(txt)) | set(b.anchor_token_offsets.keys())
        missing_marks = [m for m in b.anchor_marks if m not in marks_found]
        if missing_marks:
            issues.append(f"beat {b.beat_id}: anchor_marks missing <bookmark/> or anchor_token_offsets mapping: {missing_marks}.")

    if len(profile.speakers) > 1:
        used_speakers = {b.speaker for b in script.beats}
        if len(used_speakers) < 2:
            issues.append(f"dialogue mode {script.narration_mode!r} requires at least 2 distinct speakers.")

    # Verify immutable revision integrity when locked
    if script.locked and script.approved_revision is not None:
        app_rev = next((r for r in script.revisions if r.revision == script.approved_revision and r.approved), None)
        if app_rev is None or app_rev.content_sha256 != compute_script_sha256(script.beats):
            issues.append("ScriptIR marked locked/approved but beats diverged from approved_revision SHA-256 without creating a new reviewed revision.")
    return issues


def script_ir_to_beat_specs(script: ScriptIR) -> list[BeatSpec]:
    """Convert ScriptIR into BeatSpecs while keeping caption, spoken map, and math explicitly linked."""
    specs: list[BeatSpec] = []
    for b in script.beats:
        cap = b.displayed_caption.strip() or parse_script(b.spoken_text)[0]
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


def lock_script_to_plan(
    plan: object,
    script: ScriptIR,
    *,
    stage: Literal["initial", "editorial_repair", "math_repair", "approved", "manual"] = "editorial_repair",
    restore_locked_spoken_text: bool = True,
    strict_ids: bool = True,
) -> ScriptIR:
    """Bind VideoPlanV3 scenes strictly by stable beat_id/scene_id (NEVER positional fallback or silent prose override)."""
    beats_by_scene = {b.scene_id: b for b in script.beats}
    beats_by_id = {b.beat_id: b for b in script.beats}
    scenes = getattr(plan, "scenes", [])
    changed = False
    updated_beats: list[NarrationBeatIR] = []

    for sc in scenes:
        sid = str(getattr(sc, "scene_id", "") or "").strip()
        b_ref = beats_by_scene.get(sid) or beats_by_id.get(sid)
        if b_ref is None:
            if strict_ids:
                raise ValueError(
                    f"lock_script_to_plan: scene_id {sid!r} has no matching ScriptIR beat_id/scene_id "
                    "(positional fallback disabled for authoritative script locking)."
                )
            continue

        if getattr(sc, "speaker", None) != b_ref.speaker:
            setattr(sc, "speaker", b_ref.speaker)

        sc_narr = clean_spoken_text(str(getattr(sc, "narration", "") or ""))
        locked_txt = clean_spoken_text(b_ref.spoken_text)

        if restore_locked_spoken_text:
            if detect_banned_cliches(locked_txt):
                raise ValueError(f"Approved ScriptIR beat {b_ref.beat_id!r} contains banned cliche.")
            # Preserve identical spoken words while optionally syncing bookmark tags cleanly if words match
            sc_plain, sc_marks = split_anchors_and_spoken_text(sc_narr)
            lk_plain, lk_marks = split_anchors_and_spoken_text(locked_txt)
            if sc_marks and not lk_marks and sc_plain.lower() == lk_plain.lower():
                anchored = inject_anchors_preserving_words(lk_plain, sc_marks)
                setattr(sc, "narration", anchored)
                new_beat = b_ref.model_copy(deep=True)
                new_beat.anchor_token_offsets = dict(sc_marks)
                new_beat.anchor_marks = sorted(sc_marks.keys())
                updated_beats.append(new_beat)
                continue
            setattr(sc, "narration", b_ref.spoken_text)
            updated_beats.append(b_ref.model_copy(deep=True))
        else:
            # Explicit math/editorial revision path: never mutate locked revision silently without incrementing revision!
            if script.locked and sc_narr and sc_narr != locked_txt:
                raise ValueError(
                    f"Cannot overwrite locked approved ScriptIR beat {b_ref.beat_id!r}; call revise_beat_text() and re-approve."
                )
            new_beat = b_ref.model_copy(deep=True)
            if sc_narr and sc_narr != locked_txt:
                new_beat.spoken_text = sc_narr
                changed = True
            updated_beats.append(new_beat)

    if updated_beats:
        merged_ids = {b.beat_id for b in updated_beats}
        for orig in script.beats:
            if orig.beat_id not in merged_ids:
                updated_beats.append(orig)
        script.beats = updated_beats

    if changed:
        commit_script_revision(
            script,
            stage=stage,
            notes="Recorded new ScriptIR revision from explicit plan revision.",
            approve=False,
        )
    elif not script.revisions:
        commit_script_revision(
            script,
            stage=stage,
            notes="Initial locked ScriptIR binding.",
            approve=False,
        )
    return script


def export_script_md(script: ScriptIR) -> str:
    """Export human-readable SCRIPT.md preserving metadata, revisions, visual links, and spoken normalization."""
    lines = [
        f"# SCRIPT — {script.title}",
        "",
        f"- **Topic:** {script.topic}",
        f"- **Strategy:** `{script.strategy}` | **Narration Mode:** `{script.narration_mode}` | **Audience:** `{script.audience}`",
        f"- **Central Question:** {script.central_question or script.topic}",
        f"- **Locked:** `{script.locked}` | **Approved Revision:** `{script.approved_revision}` | **Revisions Preserved:** {len(script.revisions)}",
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
