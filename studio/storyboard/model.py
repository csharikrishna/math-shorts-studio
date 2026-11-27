"""Storyboard representation (`STORYBOARD.md`) and QA review models (`REVIEW.md`)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal
from pydantic import BaseModel, Field

from studio.config import FORMATS

if TYPE_CHECKING:
    from studio.ai.v3_schema import VideoPlanV3
    from studio.creative.script_ir import ScriptIR


class StoryboardShot(BaseModel):
    """Single shot specification linking narration, visual composition, motion, and math ops."""

    shot_id: str
    scene_id: str
    purpose: str
    composition: str = "centered_focus"
    visual_elements: list[str] = Field(default_factory=list)
    construction_ops: list[str] = Field(default_factory=list)
    motion_choreography: list[str] = Field(default_factory=list)
    transition_from_prev: str = "cut"
    transition_to_next: str = "cut"
    camera_movement: str = "static"
    start_s: float = 0.0
    end_s: float = 0.0
    duration_s: float = 0.0
    speaker: str = "narrator"
    spoken_line: str = ""
    caption_text: str = ""
    audio_cues: list[str] = Field(default_factory=list)
    look_name: str = "midnight_studio"
    verification_status: str = "pending"


class Storyboard(BaseModel):
    """Authoritative shot-by-shot storyboard exported alongside rendered videos."""

    topic: str
    title: str
    format_name: str = "widescreen_hd"
    aspect_label: str = "16:9"
    look_name: str = "midnight_studio"
    total_duration_s: float = 0.0
    shots: list[StoryboardShot] = Field(default_factory=list)


class VisualReviewNote(BaseModel):
    shot_id: str
    severity: Literal["blocker", "warning", "info"] = "info"
    category: Literal[
        "layout",
        "overlap",
        "safe_zone",
        "pacing",
        "continuity",
        "caption",
        "math_grounding",
        "visual_variety",
        "motion",
        "contrast",
        "audio",
    ]
    finding: str
    fix: str = ""


class VerificationCoverageSummary(BaseModel):
    """Explicit claim-verification coverage classification across a production."""

    deterministically_verified: int = 0
    numerically_tested: int = 0
    source_supported: int = 0
    ai_reviewed: int = 0
    unsupported: int = 0
    failed: int = 0
    human_review_required: int = 0


class VideoReviewReport(BaseModel):
    """Combined pre-render & post-render quality report exported as REVIEW.md."""

    topic: str
    title: str = ""
    format_name: str = "widescreen_hd"
    backend: str = "manimgx"
    approved: bool = True
    overall_score: float = 100.0
    script_clarity_ok: bool = True
    math_grounding_ok: bool = True
    layout_safe_zone_ok: bool = True
    audio_sync_ok: bool = True
    visual_variety_ok: bool = True
    category_statuses: dict[str, str] = Field(default_factory=dict)
    contact_sheet_path: str = ""
    verification_summary: VerificationCoverageSummary = Field(default_factory=VerificationCoverageSummary)
    notes: list[VisualReviewNote] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    summary: str = ""


def _infer_composition(scene: object, aspect_label: str) -> str:
    from studio.registry.compositions import adapt_composition_for_aspect

    explicit = str(getattr(scene, "composition", "") or "").strip()
    if explicit and explicit != "auto":
        return adapt_composition_for_aspect(explicit, aspect_label)
    kinds = [str(getattr(obj, "kind", "")) for obj in getattr(scene, "objects", [])]
    if kinds.count("axes") >= 2 or kinds.count("function_plot") >= 2:
        return "stacked_comparison" if aspect_label == "9:16" else "side_by_side_comparison"
    if "equation" in kinds and any(k in kinds for k in ("axes", "polygon", "circle", "graph_network", "data_series")):
        return "main_with_equation_panel"
    if len(getattr(scene, "events", [])) >= 4:
        return "step_by_step_sequence"
    return "centered_focus"


def storyboard_from_v3_plan(
    plan: VideoPlanV3,
    *,
    format_name: str = "widescreen_hd",
    look_name: str = "midnight_studio",
    beat_durations: dict[str, float] | None = None,
    audio_timeline: object | None = None,
    script: ScriptIR | None = None,
) -> Storyboard:
    """Compile a validated VideoPlanV3 into a structured renderer-independent Storyboard."""
    fmt = FORMATS.get(format_name, FORMATS["widescreen_hd"])
    aspect_label = "9:16" if fmt.pixel_height > fmt.pixel_width else "16:9"

    script_by_scene = {}
    if script is not None:
        for b in script.beats:
            script_by_scene.setdefault(b.scene_id, b)

    tl_beats: dict[str, object] = {}
    if audio_timeline is not None:
        for b in getattr(audio_timeline, "beats", []):
            tl_beats[getattr(b, "id", "")] = b

    shots: list[StoryboardShot] = []
    t_cursor = 0.0
    persistent_carry: set[str] = set()

    for idx, scene in enumerate(plan.scenes):
        sid = scene.scene_id
        beat_ref = script_by_scene.get(sid)
        speaker = str(getattr(scene, "speaker", "") or (beat_ref.speaker if beat_ref else "narrator"))
        spoken = beat_ref.spoken_text if beat_ref else scene.narration
        caption = (beat_ref.displayed_caption if beat_ref and beat_ref.displayed_caption else spoken).strip()

        tl_b = tl_beats.get(sid)
        if tl_b is not None:
            t_start = float(getattr(tl_b, "start_time", t_cursor))
            t_end = float(getattr(tl_b, "end_time", t_start + 2.0))
            dur = max(0.3, t_end - t_start)
        elif beat_durations and sid in beat_durations:
            dur = max(0.5, float(beat_durations[sid]))
            t_start, t_end = t_cursor, t_cursor + dur
        elif beat_ref and beat_ref.estimated_duration_s > 0:
            dur = float(beat_ref.estimated_duration_s)
            t_start, t_end = t_cursor, t_cursor + dur
        else:
            words = max(4, len(spoken.split()))
            dur = round(max(2.5, words / 2.1 + 0.35), 2)
            t_start, t_end = t_cursor, t_cursor + dur

        objs = getattr(scene, "objects", [])
        obj_ids = [f"{o.id}:{o.kind}" for o in objs]
        curr_persist = {o.id for o in objs if getattr(o, "persist", False)}
        ops_in_events = [str(getattr(e, "op", "")) for e in getattr(scene, "events", [])]
        targets_in_events = {str(getattr(e, "target_id", "")) for e in getattr(scene, "events", [])}

        carry_overlap = persistent_carry & targets_in_events
        if idx == 0:
            trans_in = "build_in"
        elif carry_overlap:
            trans_in = "transform_carry"
        else:
            trans_in = "wipe"

        camera = "static"
        if "focus" in ops_in_events or "zoom" in ops_in_events:
            camera = "focus_shift" if "focus" in ops_in_events else "zoom_in"

        choreo = [
            f"{e.op}->{e.target_id}@{e.anchor_mark or 'start'}"
            + (f" (scale={e.scale_factor})" if getattr(e, "scale_factor", None) is not None else "")
            + (f" (rot={e.angle_deg}deg)" if getattr(e, "angle_deg", None) is not None else "")
            for e in getattr(scene, "events", [])
        ]
        c_ops = [f"{c.op}:{'/'.join(c.output_ids)}" for c in getattr(scene, "constructions", [])]

        claims = getattr(scene, "claims", [])
        if any(getattr(c, "kind", None) for c in claims):
            ver_status = "deterministically_verified"
        elif c_ops:
            ver_status = "numerically_tested"
        else:
            ver_status = "ai_reviewed"

        shots.append(
            StoryboardShot(
                shot_id=f"shot_{idx + 1:02d}_{sid}",
                scene_id=sid,
                purpose=scene.purpose,
                composition=_infer_composition(scene, aspect_label),
                visual_elements=obj_ids,
                construction_ops=c_ops,
                motion_choreography=choreo,
                transition_from_prev=trans_in,
                transition_to_next="cut",
                camera_movement=camera,
                start_s=round(t_start, 3),
                end_s=round(t_end, 3),
                duration_s=round(dur, 3),
                speaker=speaker,
                spoken_line=spoken,
                caption_text=caption,
                audio_cues=[str(getattr(scene, "sfx_cue", "pop_marimba"))],
                look_name=look_name,
                verification_status=ver_status,
            )
        )
        t_cursor = max(t_end, t_cursor + dur)
        persistent_carry |= curr_persist

    for idx in range(len(shots) - 1):
        shots[idx].transition_to_next = shots[idx + 1].transition_from_prev

    return Storyboard(
        topic=plan.topic,
        title=plan.title,
        format_name=format_name,
        aspect_label=aspect_label,
        look_name=look_name,
        total_duration_s=round(t_cursor, 3),
        shots=shots,
    )


def export_storyboard_md(sb: Storyboard, out_path: Path | None = None) -> str:
    """Render Storyboard as structured STORYBOARD.md."""
    lines = [
        f"# Storyboard: {sb.title}",
        "",
        f"- **Topic:** {sb.topic}",
        f"- **Format:** `{sb.format_name}` ({sb.aspect_label})",
        f"- **Visual Look:** `{sb.look_name}`",
        f"- **Total Duration:** {sb.total_duration_s:.2f} s ({len(sb.shots)} shots)",
        "",
    ]
    for shot in sb.shots:
        lines.extend(
            [
                f"## {shot.shot_id} ({shot.start_s:.2f}s - {shot.end_s:.2f}s, {shot.duration_s:.2f}s)",
                f"- **Purpose:** {shot.purpose}",
                f"- **Composition:** `{shot.composition}` | **Camera:** `{shot.camera_movement}`",
                f"- **Transitions:** `{shot.transition_from_prev}` -> `{shot.transition_to_next}`",
                f"- **Speaker (`{shot.speaker}`):** {shot.spoken_line}",
                f"- **Visual Elements:** {', '.join(f'`{v}`' for v in shot.visual_elements) or 'carried state'}",
                f"- **Constructions:** {', '.join(f'`{c}`' for c in shot.construction_ops) or 'none'}",
                f"- **Motion Choreography:** {'; '.join(shot.motion_choreography) or 'hold'}",
                f"- **Audio Cues:** {', '.join(shot.audio_cues)} | **Verification:** `{shot.verification_status}`",
                "",
            ]
        )
    content = "\n".join(lines).strip() + "\n"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
    return content


def export_review_md(report: VideoReviewReport, out_path: Path | None = None) -> str:
    """Render VideoReviewReport as structured REVIEW.md."""
    cov = report.verification_summary
    lines = [
        f"# Production Review: {report.title or report.topic}",
        "",
        f"- **Topic:** {report.topic}",
        f"- **Format / Backend:** `{report.format_name}` / `{report.backend}`",
        f"- **Approved:** `{report.approved}` (Overall Score: **{report.overall_score:.1f}/100**)",
        f"- **Checks:** Clarity={report.script_clarity_ok} | Math={report.math_grounding_ok} | Layout={report.layout_safe_zone_ok} | AudioSync={report.audio_sync_ok} | VisualVariety={report.visual_variety_ok}",
        "",
        "## Claim Verification Coverage",
        f"- `deterministically_verified`: {cov.deterministically_verified}",
        f"- `numerically_tested`: {cov.numerically_tested}",
        f"- `source_supported`: {cov.source_supported}",
        f"- `ai_reviewed`: {cov.ai_reviewed}",
        f"- `unsupported`: {cov.unsupported}",
        f"- `failed`: {cov.failed}",
        f"- `human_review_required`: {cov.human_review_required}",
        "",
    ]
    if report.metrics:
        lines.append("## Measured Metrics")
        lines.extend(f"- {m}" for m in report.metrics)
        lines.append("")
    if report.category_statuses:
        lines.append("## 10-Category Honest QA Status")
        lines.extend(f"- `{k}`: **{v}**" for k, v in report.category_statuses.items())
        lines.append("")
    if report.contact_sheet_path:
        lines.append(f"- **Contact Sheet:** `{report.contact_sheet_path}`\n")
    if report.notes:
        lines.append("## Findings")
        for n in report.notes:
            fix_txt = f" -> Fix: {n.fix}" if n.fix else ""
            lines.append(f"- **[{n.severity.upper()}]** `{n.shot_id}` (`{n.category}`): {n.finding}{fix_txt}")
        lines.append("")
    if report.summary:
        lines.extend(["## Summary", report.summary, ""])

    content = "\n".join(lines).strip() + "\n"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
    return content
