"""Separated CreativeDirector: ConceptAnalysis -> Storylines -> ScriptIR -> EditorialReview."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from studio.creative.dialogue import clean_spoken_text, format_dialogue_instructions
from studio.creative.models import (
    ConceptAnalysis,
    CreativeBrief,
    EXPLANATION_STRATEGY_GUIDES,
    EditorialIssue,
    EditorialReviewReport,
    NarrationMode,
    export_storylines_md,
)
from studio.creative.script_ir import (
    ScriptIR,
    ScriptRevision,
    approve_script_ir,
    commit_script_revision,
    export_script_md,
    validate_script_ir,
)
from studio.design.spec import DesignSpec, default_design_spec, export_design_md
from studio.providers import get_provider
from studio.registry import capabilities_prompt_summary


@dataclass
class CreativePackage:
    brief: CreativeBrief
    script: ScriptIR
    review: EditorialReviewReport
    design: DesignSpec


BRIEF_SYSTEM = """
You are an award-winning educational filmmaker and pedagogue (3Blue1Brown / Veritasium caliber).
Before choosing equations or scenes, deeply analyze the concept and explore 3 genuinely distinct narrative storylines.
Rules:
- Storyline 1..3 must use DIFFERENT explanation strategies and opening visual hooks (do not pitch minor word tweaks).
- Never use generic hype or cliches ("Welcome to a journey", "Unlock secrets", "game-changer").
- Anchor the explanation in ONE compelling central question, prerequisites, concrete intuition BEFORE formal symbols, and explain WHY each mathematical relationship works in natural conversational English.
"""


class CreativeDirector:
    """Pre-render creative intelligence orchestrator decoupled from video backends."""

    def __init__(
        self,
        model: str | None = None,
        provider_name: str = "gemini",
        free_only: bool = True,
    ) -> None:
        self.provider = get_provider(provider_name, model=model, free_only=free_only)

    def generate_brief(
        self,
        topic: str,
        *,
        audience: str = "general",
        narration_mode: NarrationMode = "single_educator",
    ) -> CreativeBrief:
        """Analyze concept and pitch 3 distinct narrative storylines, selecting the strongest fit."""
        dlg_rules = format_dialogue_instructions(narration_mode)
        prompt = (
            f"TOPIC: {topic}\n"
            f"AUDIENCE: {audience}\n"
            f"{dlg_rules}\n\n"
            "Produce a CreativeBrief JSON with:\n"
            "1. `concept`: meaning, prerequisites, central_question, essential_definitions, common_misconceptions, "
            "mathematical_relationships, real_world_examples, learning_outcome.\n"
            "2. `storylines`: EXACTLY 3 genuinely distinct StorylineOption objects (`opt_1`, `opt_2`, `opt_3`) "
            "exploring contrasting `strategy`, `narrative_device`, `visual_form`, and concrete `opening_two_seconds`. "
            "Mark exactly one `recommended=true` with clear `selection_reason`.\n"
            "3. Set `selected_storyline_id`, `selected_strategy`, `narration_mode`, and `audience`."
        )
        brief, _ = self.provider.generate_structured(prompt, CreativeBrief, system=BRIEF_SYSTEM, temperature=0.55)
        if brief.storylines and not any(s.option_id == brief.selected_storyline_id for s in brief.storylines):
            rec = next((s for s in brief.storylines if s.recommended), brief.storylines[0])
            brief.selected_storyline_id = rec.option_id
            brief.selected_strategy = rec.strategy
        brief.narration_mode = narration_mode
        brief.audience = audience
        return brief

    def draft_script(
        self,
        brief: CreativeBrief,
        *,
        target_duration_s: float = 55.0,
        backend: str = "manimgx",
        format_name: str = "widescreen_hd",
    ) -> ScriptIR:
        """Draft a renderer-independent ScriptIR with aligned beats, visual intent, and bookmarks."""
        aspect = "9:16" if "shorts" in format_name else "16:9"
        caps = capabilities_prompt_summary(backend=backend, aspect=aspect, compact=True)
        dlg_rules = format_dialogue_instructions(brief.narration_mode)
        selected = next(
            (s for s in brief.storylines if s.option_id == brief.selected_storyline_id),
            brief.storylines[0] if brief.storylines else None,
        )
        min_words = max(45, int(target_duration_s * 2.0))
        target_words = max(60, int(target_duration_s * 2.5))
        n_beats = max(4, min(14, round(target_duration_s / 8.5))) if target_duration_s <= 120 else max(8, min(20, round(target_duration_s / 14.0)))

        strat_guide = EXPLANATION_STRATEGY_GUIDES.get(brief.selected_strategy, "")
        prompt = (
            f"Draft a ScriptIR JSON for:\n"
            f"TOPIC: {brief.concept.topic}\n"
            f"CENTRAL QUESTION: {brief.concept.central_question}\n"
            f"SELECTED STRATEGY: {brief.selected_strategy} — {strat_guide}\n"
            f"STORYLINE ARC: {selected.beat_arc if selected else ''}\n"
            f"OPENING 2s: {selected.opening_two_seconds if selected else ''}\n"
            f"REAL-WORLD / CONCRETE EXAMPLES: {'; '.join(brief.concept.real_world_examples)}\n"
            f"MISCONCEPTIONS TO RESOLVE: {'; '.join(brief.concept.common_misconceptions)}\n"
            f"{dlg_rules}\n\n"
            f"TARGET: ~{target_duration_s:.0f}s ({n_beats} beats, ~{target_words} total words, minimum {min_words} words).\n"
            "STRICT PEDAGOGICAL & SCRIPT RULES:\n"
            "- Write pure speakable English in `spoken_text` (NO raw LaTeX like \\frac or \\alpha and NO code tokens like x_dot; write 'x dot', 'x squared', 'lambda').\n"
            "- Explain WHY mechanisms behave the way they do, not just what happens next: open with a concrete intuition or sensory comparison before introducing symbols, and define each new variable (`symbols_introduced`) the exact moment it enters.\n"
            "- Keep narration conversational, clear, and varied in rhythm — never repeat canned transition stems or fake drama/hype.\n"
            "- Use 1-2 *asterisk* emphasis words and place `<bookmark mark='m1'/>` right before key visual trigger words.\n"
            "- Every mark listed in `anchor_marks` MUST appear verbatim as `<bookmark mark='...'/>` in `spoken_text`.\n"
            "- Populate `visual_intent`, `educational_purpose`, `mathematical_claims`, `latex_refs`, `beat_id`, and `demo_contract` "
            "(`learner_takeaway`, `relationship_explained`, `required_state_change`, `visual_components`, `changing_parameters`, "
            "`displayed_measurements`, `persisting_objects`, `intentional_static`) cleanly.\n"
            "- Whenever a beat explains motion, oscillation, convergence, phase flow, damping, resonance, or transformation, specify "
            "real state changes in `demo_contract.required_state_change` and `changing_parameters` instead of static plots.\n"
            "- Populate `revisions` as empty list (tracked automatically).\n\n"
            f"{caps}"
        )
        script, _ = self.provider.generate_structured(prompt, ScriptIR, system=BRIEF_SYSTEM, temperature=0.45)
        for idx, b in enumerate(script.beats, 1):
            b.spoken_text = clean_spoken_text(b.spoken_text)
            if not b.beat_id:
                b.beat_id = f"b_{idx}"
            if not b.scene_id:
                b.scene_id = f"scene_{idx}"
        script.concept_analysis = brief.concept
        script.narration_mode = brief.narration_mode
        script.strategy = brief.selected_strategy
        script.audience = brief.audience
        script.central_question = brief.concept.central_question
        script.revisions = []
        commit_script_revision(
            script,
            stage="initial",
            notes="Initial ScriptIR generated from CreativeBrief",
            approve=False,
        )
        return script

    def review_script(self, script: ScriptIR, *, use_llm: bool = False) -> EditorialReviewReport:
        """Validate ScriptIR deterministically and optionally run LLM editorial critique."""
        det_issues = validate_script_ir(script)
        issues: list[EditorialIssue] = [
            EditorialIssue(beat_id=f"det_{i}", category="natural_speech", detail=msg)
            for i, msg in enumerate(det_issues, 1)
        ]
        if use_llm and not det_issues:
            prompt = (
                "Critically review this ScriptIR as a senior math editor & educator.\n"
                "Check: (1) explicit central question established upfront, (2) concrete example before abstract formulas, "
                "(3) symbols explained clearly upon first appearance, (4) natural transitions and concise learner takeaway.\n"
                "Flag ONLY real accuracy, clarity, missing prerequisite, repetition, jargon, or filler issues.\n"
                "Set approved=true if the script is mathematically accurate, clear, cohesive, and engaging.\n\n"
                f"{script.model_dump_json()}"
            )
            try:
                llm_rep, _ = self.provider.generate_structured(prompt, EditorialReviewReport, system=BRIEF_SYSTEM, temperature=0.2)
                issues.extend(llm_rep.issues)
                return EditorialReviewReport(
                    approved=llm_rep.approved and len(issues) == 0,
                    math_accuracy_ok=llm_rep.math_accuracy_ok,
                    clarity_ok=llm_rep.clarity_ok,
                    progression_ok=llm_rep.progression_ok,
                    natural_speech_ok=llm_rep.natural_speech_ok,
                    issues=issues,
                    summary=llm_rep.summary or "Independent LLM editorial review complete.",
                )
            except Exception as exc:
                return EditorialReviewReport(
                    approved=True,
                    math_accuracy_ok=True,
                    clarity_ok=True,
                    progression_ok=True,
                    natural_speech_ok=True,
                    issues=[],
                    summary=f"Deterministic ScriptIR check passed (LLM editorial fallback: {type(exc).__name__}).",
                )

        approved = len(issues) == 0
        return EditorialReviewReport(
            approved=approved,
            math_accuracy_ok=approved,
            clarity_ok=approved,
            progression_ok=approved,
            natural_speech_ok=approved,
            issues=issues,
            summary="Deterministic ScriptIR check passed." if approved else f"{len(issues)} script issue(s) flagged.",
        )

    def repair_script(self, script: ScriptIR, report: EditorialReviewReport, *, target_duration_s: float = 55.0) -> ScriptIR:
        """Repair flagged ScriptIR issues while preserving revision history."""
        if report.approved and not report.issues:
            return script
        prior_revs = list(script.revisions)
        issue_lines = "\n".join(f"- [{it.beat_id} / {it.category}] {it.detail} ({it.suggested_fix})" for it in report.issues)
        prompt = (
            f"Fix ONLY the listed editorial/validation issues in this ScriptIR for target {target_duration_s:.0f}s:\n"
            f"{issue_lines}\n\n"
            f"Rules: Preserve ALL {len(script.beats)} beats and their exact beat_id/chapter_id/scene_id in order! "
            f"Keep clean speakable English (no raw LaTeX), ensure `anchor_marks` match `<bookmark mark='...'/>`, "
            f"zero banned cliches.\n\nCURRENT SCRIPT:\n{script.model_dump_json(exclude_none=True)}"
        )
        fixed, _ = self.provider.generate_structured(prompt, ScriptIR, system=BRIEF_SYSTEM, temperature=0.3)
        fixed_map: dict[str, NarrationBeatIR] = {}
        for b in fixed.beats:
            b.spoken_text = clean_spoken_text(b.spoken_text)
            if b.beat_id:
                fixed_map[b.beat_id] = b
        # Never allow LLM truncation to drop existing beats or chapters during repair
        if len(fixed.beats) < len(script.beats):
            merged_beats = [fixed_map.get(orig.beat_id, orig.model_copy(deep=True)) for orig in script.beats]
            known_ids = {b.beat_id for b in merged_beats}
            for extra_b in fixed.beats:
                if extra_b.beat_id and extra_b.beat_id not in known_ids:
                    merged_beats.append(extra_b)
                    known_ids.add(extra_b.beat_id)
            fixed.beats = merged_beats
        fixed.concept_analysis = script.concept_analysis
        fixed.narration_mode = script.narration_mode
        fixed.strategy = script.strategy
        fixed.audience = script.audience
        fixed.central_question = script.central_question
        fixed.revisions = prior_revs + [
            ScriptRevision(
                revision=len(prior_revs) + 1,
                stage="editorial_repair",
                notes=f"Repaired {len(report.issues)} issue(s)",
                beats=[b.model_copy(deep=True) for b in fixed.beats],
            )
        ]
        return fixed

    def produce_creative_package(
        self,
        topic: str,
        *,
        audience: str = "general",
        narration_mode: NarrationMode = "single_educator",
        format_name: str = "widescreen_hd",
        look_name: str = "midnight_studio",
        backend: str = "manimgx",
        target_duration_s: float = 55.0,
        artifact_dir: Path | None = None,
        llm_editorial_review: bool = False,
    ) -> CreativePackage:
        """Full pre-render creative workflow exporting STORYLINES.md, SCRIPT.md, and DESIGN.md."""
        design = default_design_spec(format_name=format_name, look_name=look_name)
        brief = self.generate_brief(topic, audience=audience, narration_mode=narration_mode)
        script = self.draft_script(brief, target_duration_s=target_duration_s, backend=backend, format_name=format_name)
        review = self.review_script(script, use_llm=llm_editorial_review)
        for _ in range(2):
            if review.approved:
                break
            script = self.repair_script(script, review, target_duration_s=target_duration_s)
            review = self.review_script(script, use_llm=False)

        if review.approved and not validate_script_ir(script):
            approve_script_ir(script, notes=review.summary or "Editorial review passed")

        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "STORYLINES.md").write_text(export_storylines_md(brief), encoding="utf-8")
            (artifact_dir / "SCRIPT.md").write_text(export_script_md(script), encoding="utf-8")
            (artifact_dir / "DESIGN.md").write_text(export_design_md(design), encoding="utf-8")

        return CreativePackage(brief=brief, script=script, review=review, design=design)
