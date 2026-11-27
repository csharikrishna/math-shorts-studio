"""Autonomous long-form chapter-based educational video production pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from studio.ai.director import GeminiDirector, MAX_REPAIRS, V3_STYLE_GUIDE, slugify
from studio.ai.knowledge import AudienceLevel, ChapterSpec, ContinuityState, KnowledgeTree, LongFormPlan
from studio.ai.v3_schema import VideoPlanV3
from studio.ai.validate_v3 import validate_v3_plan
from studio.ai.world_model import materialize_constructions
from studio.audio.tts import AudioTimeline, find_ffmpeg
from studio.creative.dialogue import clean_spoken_text, format_dialogue_instructions, resolve_speaker_voices
from studio.creative.director import CreativeDirector
from studio.creative.script_ir import (
    NarrationBeatIR,
    ScriptIR,
    approve_script_ir,
    commit_script_revision,
    export_script_md,
    lock_script_to_plan,
    sync_beat_anchors,
    validate_script_ir,
)
from studio.design.spec import default_design_spec, export_design_md, resolve_look
from studio.pipeline import PROJECT_ROOT, run_plan
from studio.qa.engine import audit_educational_quality, verify_audio_loudness, verify_longform_bundle, verify_media_streams
from studio.registry import capabilities_prompt_summary
from studio.registry.compositions import adapt_composition_for_aspect
from studio.storage.cache import ArtifactCache, JobStore, compute_dependency_fingerprints, validate_sidecar_bundle
from studio.storyboard import export_review_md, export_storyboard_md, storyboard_from_v3_plan
from studio.timeline import _fmt_srt_ts


def _probe_duration(mp4: Path) -> float:
    ff = find_ffmpeg()
    p = subprocess.run([ff, "-hide_banner", "-i", str(mp4)], capture_output=True, text=True, timeout=30)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def _shift_srt_ts(ts: str, offset_s: float) -> str:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    total = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0 + offset_s
    return _fmt_srt_ts(total)


def _fmt_yt_ts(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# Measured Kokoro (-4% speed) + inter-beat pauses yields ~2.08 spoken words/sec end-to-end
KOKORO_EFFECTIVE_WPS = 2.08


def _allocate_chapter_budgets(chapters: list[ChapterSpec], total_duration_s: float) -> list[float]:
    """Allocate concept-aware target durations across chapters (never flat uniform division)."""
    if not chapters:
        return []
    weights: list[float] = []
    n = len(chapters)
    for idx, ch in enumerate(chapters):
        title_low = (ch.title + " " + ch.summary).lower()
        w = 1.0
        if idx == 0 or any(k in title_low for k in ("hook", "intuition", "intro", "question")):
            w = 0.85
        elif any(k in title_low for k in ("proof", "derivation", "formal", "mechanism", "theorem", "eigen", "stability", "core")):
            w = 1.18
        elif any(k in title_low for k in ("edge", "regime", "limit", "misconception", "contrast", "case")):
            w = 1.10
        elif idx == n - 1 or any(k in title_low for k in ("summary", "synthesis", "application", "takeaway")):
            w = 0.90
        if ch.target_duration_s and ch.target_duration_s > 10.0:
            w = 0.5 * w + 0.5 * (float(ch.target_duration_s) / max(1.0, total_duration_s / n))
        weights.append(max(0.65, w))
    w_sum = sum(weights) or float(n)
    return [round(float(total_duration_s) * (w / w_sum), 2) for w in weights]


def _source_code_digest() -> str:
    fps = compute_dependency_fingerprints(PROJECT_ROOT)
    return JobStore.content_hash(fps)


def chapter_cache_key(
    plan: VideoPlanV3,
    *,
    format_name: str,
    tts_engine: str,
    voice: str,
    rate: str,
    ui: str,
    look: str,
    look_name: str = "midnight_studio",
    caption_style: str = "editorial",
    pitch: str = "-2Hz",
    backend: str = "manimgx",
) -> str:
    """Compute dependency-aware cache key across math, script, captions, TTS, layout, backend, and audio mastering."""
    fps = compute_dependency_fingerprints(PROJECT_ROOT)
    backend_fp = fps["hyperframes_backend"] if backend == "hyperframes" else (
        f"{fps['manimgx_backend']}+{fps['hyperframes_backend']}" if backend == "hybrid" else fps["manimgx_backend"]
    )
    return JobStore.content_hash(
        {
            "plan": plan.model_dump(),
            "format_name": format_name,
            "tts_engine": tts_engine,
            "voice": voice,
            "rate": rate,
            "pitch": pitch,
            "ui": ui,
            "look": look,
            "look_name": look_name,
            "caption_style": caption_style,
            "backend": backend,
            "deps": {
                "math": f"{fps['math_compute']}:{fps['math_validate']}",
                "captions": fps["captions"],
                "tts_voice": fps["tts_voice"],
                "fonts": fps["fonts"],
                "design_tokens": fps["design_tokens"],
                "components": fps["components"],
                "layout_solver": fps["layout_solver"],
                "audio_master": fps["audio_master"],
                "backend_impl": backend_fp,
            },
        }
    )


def _atomic_write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class _VarDef(BaseModel):
    symbol: str
    meaning: str


class _ColorDef(BaseModel):
    role: str
    color: str


class _ChapterBridge(BaseModel):
    chapter_id: str
    prev_chapter_summary: str = ""
    established_definitions: list[str] = Field(default_factory=list)
    introduced_variables: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    next_transition: str = ""


class _LongFormOutline(BaseModel):
    topic: str
    audience: AudienceLevel = "general"
    central_question: str = ""
    narrative_arc: str = ""
    selected_strategy: str = "concrete_then_abstract"
    knowledge_tree: KnowledgeTree
    chapters: list[ChapterSpec] = Field(default_factory=list)
    chapter_bridges: list[_ChapterBridge] = Field(default_factory=list)
    defined_variables: list[_VarDef] = Field(default_factory=list)
    introduced_terms: list[str] = Field(default_factory=list)
    color_map: list[_ColorDef] = Field(default_factory=list)


def create_longform_plan(
    topic: str,
    *,
    target_duration_s: float = 300.0,
    audience: AudienceLevel = "general",
    narration_mode: str = "documentary",
    model: str | None = None,
    provider_name: str = "gemini",
    free_only: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[LongFormPlan, list[VideoPlanV3]]:
    """Autonomously generate CreativeBrief/ScriptIR -> GlobalNarrativePlan -> ChapterBridges -> validated chapters."""
    slug = slugify(topic)
    lf_dir = PROJECT_ROOT / "build" / f"longform_{slug}"
    logs_dir = lf_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    gen_log: list[dict[str, object]] = []

    director = GeminiDirector(model=model, provider_name=provider_name, free_only=free_only)
    if target_duration_s >= 300.0:
        n_chapters = max(5, min(10, round(target_duration_s / 82.0)))
    else:
        n_chapters = max(3, min(8, round(target_duration_s / 36.0)))
    per_ch_s = target_duration_s / n_chapters

    registry_caps = capabilities_prompt_summary(backend="manimgx", aspect="16:9", compact=True)
    dlg_rules = format_dialogue_instructions(narration_mode)  # type: ignore[arg-type]
    ch_system = f"{V3_STYLE_GUIDE}\n\n{dlg_rules}\n\n{registry_caps}"

    req_sig = JobStore.content_hash(
        {"topic": topic, "dur": target_duration_s, "aud": audience, "mode": narration_mode, "n": n_chapters, "v": "v38"}
    )
    # Run CreativeDirector upfront for STORYLINES.md + central question + strategy + misconceptions
    brief_cache = logs_dir / f"brief_{req_sig}.json"
    if brief_cache.exists():
        from studio.creative.models import CreativeBrief, export_storylines_md

        brief = CreativeBrief.model_validate_json(brief_cache.read_text(encoding="utf-8"))
    else:
        from studio.creative.models import export_storylines_md

        cdir = CreativeDirector(model=model, provider_name=provider_name, free_only=free_only)
        brief = cdir.generate_brief(topic, audience=audience, narration_mode=narration_mode)  # type: ignore[arg-type]
        brief_cache.write_text(brief.model_dump_json(indent=2), encoding="utf-8")
    (lf_dir / "STORYLINES.md").write_text(export_storylines_md(brief), encoding="utf-8")

    if on_progress:
        on_progress(f"planning global narrative arc, knowledge tree & {n_chapters} chapters (~{target_duration_s:.0f}s target)...")

    outline_prompt = (
        f"Create a structured LongFormPlan for an educational documentary on: {topic}\n"
        f"Audience: {audience}. Narration mode: {narration_mode}.\n"
        f"Selected Strategy: {brief.selected_strategy}. Central Question: {brief.concept.central_question}.\n"
        f"Key Misconceptions to Address: {'; '.join(brief.concept.common_misconceptions)}\n"
        f"Target total duration: {target_duration_s:.0f}s across EXACTLY {n_chapters} substantive chapters "
        f"(assign realistic `target_duration_s` per chapter summing to ~{target_duration_s:.0f}s based on conceptual depth, NOT equal chunks).\n"
        "Include:\n"
        "- central_question & narrative_arc & selected_strategy explaining WHY the topic matters\n"
        "- knowledge_tree with prerequisite-ordered nodes and verifiable claims\n"
        "- chapters covering intuition -> construction -> formal mechanism -> edge cases/regimes -> applications -> synthesis\n"
        "- chapter_bridges for EACH chapter detailing prev_chapter_summary, established_definitions, introduced_variables, unresolved_questions, and next_transition\n"
        "- continuity with shared defined_variables, introduced_terms, and color_map."
    )
    outline_cache = logs_dir / "outline_cached.json"
    if outline_cache.exists() and (logs_dir / "outline_sig.txt").exists() and (logs_dir / "outline_sig.txt").read_text(encoding="utf-8").strip() == req_sig:
        outline = _LongFormOutline.model_validate_json(outline_cache.read_text(encoding="utf-8"))
        gen_log.append({"stage": "outline_cache_hit"})
    else:
        outline, resp = director.provider.generate_structured(outline_prompt, _LongFormOutline, temperature=0.4)
        outline_cache.write_text(outline.model_dump_json(indent=2), encoding="utf-8")
        (logs_dir / "outline_sig.txt").write_text(req_sig, encoding="utf-8")
        gen_log.append({"stage": "outline", "model": resp.model, "latency_s": resp.latency_s, "tokens": [resp.prompt_tokens, resp.completion_tokens]})
    chapter_budgets = _allocate_chapter_budgets(outline.chapters, target_duration_s)
    for ch_obj, b_s in zip(outline.chapters, chapter_budgets):
        ch_obj.target_duration_s = b_s

    lf_plan = LongFormPlan(
        topic=outline.topic,
        audience=outline.audience,
        knowledge_tree=outline.knowledge_tree,
        chapters=outline.chapters,
        continuity=ContinuityState(
            defined_variables={v.symbol: v.meaning for v in outline.defined_variables},
            introduced_terms=outline.introduced_terms,
            color_map={c.role: c.color for c in outline.color_map},
        ),
    )
    (logs_dir / "longform_plan_raw.json").write_text(lf_plan.model_dump_json(indent=2), encoding="utf-8")
    _atomic_write_json(logs_dir / "narrative_bridges.json", outline.model_dump())

    bridge_map = {b.chapter_id: b for b in outline.chapter_bridges}
    node_map = {n.node_id: n for n in lf_plan.knowledge_tree.nodes}

    from studio.audio.tts import parse_script

    # --- STAGE 2: Authoritative Script-First Generation (ScriptIR BEFORE Scene Plans) ---
    script_cache_file = logs_dir / f"script_{req_sig}_approved.json"
    if script_cache_file.exists():
        doc_script = ScriptIR.model_validate_json(script_cache_file.read_text(encoding="utf-8"))
        gen_log.append({"stage": "script_ir_cache_hit", "beats": len(doc_script.beats)})
    else:
        cdir = CreativeDirector(model=model, provider_name=provider_name, free_only=free_only)
        ch_beat_lists: list[list[NarrationBeatIR]] = [None] * len(lf_plan.chapters)  # type: ignore[list-item]

        def _draft_ch_script(idx: int, ch: ChapterSpec) -> tuple[int, list[NarrationBeatIR]]:
            ch_target_s = float(ch.target_duration_s or per_ch_s)
            ch_min_words = max(55, int(ch_target_s * 1.85))
            ch_max_words = max(ch_min_words + 25, int(ch_target_s * 2.28))
            prev_fallback = (
                f"Previous Chapter ({lf_plan.chapters[idx - 2].title}): {lf_plan.chapters[idx - 2].summary}"
                if idx >= 2
                else "Opening of documentary: pose the concrete central puzzle before introducing formulas."
            )
            node_ctx = "; ".join(f"{node_map[nid].title}: {[c.statement for c in node_map[nid].claims]}" for nid in ch.node_ids if nid in node_map)
            bridge = bridge_map.get(ch.chapter_id)
            bridge_ctx = (
                f"Previous Summary: {bridge.prev_chapter_summary or prev_fallback}\n"
                f"Established Definitions: {bridge.established_definitions}\n"
                f"Unresolved Question Entering Chapter: {bridge.unresolved_questions}\n"
                f"Next Chapter Transition: {bridge.next_transition}\n"
                if bridge
                else f"Previous Summary: {prev_fallback}\n"
            )
            n_beats = max(4, min(8, round(ch_target_s / 13.0)))
            s_prompt = (
                f"Authoritative Script-First Beat Sequence (NO rendering code) for Chapter {idx}/{len(lf_plan.chapters)}:\n"
                f"TOPIC: {topic}\nCENTRAL QUESTION: {brief.concept.central_question or outline.central_question}\n"
                f"STRATEGY: {brief.selected_strategy} | MODE: {narration_mode} | AUDIENCE: {audience}\n"
                f"CHAPTER: {ch.chapter_id} — {ch.title}\nSUMMARY: {ch.summary}\n{bridge_ctx}"
                f"KNOWLEDGE NODES: {node_ctx}\nVARIABLES: {json.dumps(lf_plan.continuity.defined_variables)}\n"
                f"TARGET: {n_beats} beats totaling {ch_min_words}-{ch_max_words} spoken words (~{ch_target_s:.0f}s).\n"
                "RULES:\n"
                f"- Use beat_id='{ch.chapter_id}__s1'..'s{n_beats}' and scene_id='scene_1'..'scene_{n_beats}'.\n"
                "- Pure speakable English (NO LaTeX macros, NO underscore identifiers like x_dot; write 'x dot').\n"
                "- Concrete example before formulas; define symbols when first introduced (`symbols_introduced`).\n"
                "- Connect naturally from previous beat without cliches or repeated opening stems.\n"
                "- Populate visual_intent, educational_purpose, mathematical_claims, and anchor_marks cleanly."
            )
            ch_ir, s_resp = cdir.provider.generate_structured(s_prompt, ScriptIR, temperature=0.4)
            with log_lock:
                gen_log.append({"stage": f"script_{ch.chapter_id}", "model": s_resp.model, "latency_s": s_resp.latency_s})
            seen_sids: set[str] = set()
            for bi, b in enumerate(ch_ir.beats, 1):
                b.chapter_id = ch.chapter_id
                cand_sid = (b.scene_id or f"scene_{bi}").strip()
                if cand_sid in seen_sids:
                    cand_sid = f"scene_{bi}"
                seen_sids.add(cand_sid)
                b.scene_id = cand_sid
                b.beat_id = f"{ch.chapter_id}__{b.scene_id}"
                b.spoken_text = clean_spoken_text(b.spoken_text)
                b.estimated_duration_s = round(max(2.5, len(parse_script(b.spoken_text)[0].split()) / KOKORO_EFFECTIVE_WPS + 0.35), 2)
            # Editorial review & automatic repair of this chapter's ScriptIR BEFORE freezing
            rep = cdir.review_script(ch_ir, use_llm=False)
            if not rep.approved:
                ch_ir = cdir.repair_script(ch_ir, rep, target_duration_s=ch_target_s)
                seen_sids.clear()
                for bi, b in enumerate(ch_ir.beats, 1):
                    b.chapter_id = ch.chapter_id
                    cand_sid = (b.scene_id or f"scene_{bi}").strip()
                    if cand_sid in seen_sids:
                        cand_sid = f"scene_{bi}"
                    seen_sids.add(cand_sid)
                    b.scene_id = cand_sid
                    b.beat_id = f"{ch.chapter_id}__{b.scene_id}"
                    b.spoken_text = clean_spoken_text(b.spoken_text)
            return idx, ch_ir.beats

        log_lock = threading.Lock()
        script_workers = max(1, min(len(lf_plan.chapters), int(os.environ.get("STUDIO_LLM_WORKERS", "3"))))
        with ThreadPoolExecutor(max_workers=script_workers) as pool:
            sfuts = [pool.submit(_draft_ch_script, idx, ch) for idx, ch in enumerate(lf_plan.chapters, 1)]
            for fut in as_completed(sfuts):
                s_idx, s_beats = fut.result()
                ch_beat_lists[s_idx - 1] = s_beats

        all_beats = [b for bl in ch_beat_lists if bl for b in bl]
        doc_script = ScriptIR(
            topic=lf_plan.topic,
            title=lf_plan.topic,
            audience=audience,
            narration_mode=narration_mode if narration_mode in ("single_educator", "conversational_explainer", "socratic", "discussion", "documentary", "technical_instructor") else "documentary",  # type: ignore[arg-type]
            strategy=brief.selected_strategy,
            central_question=brief.concept.central_question,
            concept_analysis=brief.concept,
            beats=all_beats,
        )
        commit_script_revision(doc_script, stage="initial", notes="Assembled chapter-ordered ScriptIR before visual scene planning", approve=False)
        rep_global = cdir.review_script(doc_script, use_llm=False)
        if not rep_global.approved:
            doc_script = cdir.repair_script(doc_script, rep_global, target_duration_s=target_duration_s)
            for bi, b in enumerate(doc_script.beats, 1):
                if not b.chapter_id:
                    b.chapter_id = b.beat_id.split("__", 1)[0] if "__" in b.beat_id else "ch_1"
                b.beat_id = f"{b.chapter_id}__{b.scene_id}"
                b.spoken_text = clean_spoken_text(b.spoken_text)
        approve_script_ir(doc_script, notes="Global ScriptIR approved prior to scene compilation")
        script_cache_file.write_text(doc_script.model_dump_json(indent=2), encoding="utf-8")

    (lf_dir / "SCRIPT.md").write_text(export_script_md(doc_script), encoding="utf-8")
    _atomic_write_json(lf_dir / "script_ir_approved.json", doc_script.model_dump())

    beats_by_chapter: dict[str, list[NarrationBeatIR]] = {}
    for b in doc_script.beats:
        cid = b.chapter_id or (b.beat_id.split("__", 1)[0] if "__" in b.beat_id else "")
        beats_by_chapter.setdefault(cid, []).append(b)

    log_lock = threading.Lock()
    ordered_plans: list[VideoPlanV3 | None] = [None] * len(lf_plan.chapters)

    def _compile_chapter_from_script(idx: int, ch: ChapterSpec) -> tuple[int, VideoPlanV3]:
        ch_beats = beats_by_chapter.get(ch.chapter_id, [])
        ch_script = ScriptIR(
            topic=doc_script.topic,
            title=ch.title,
            audience=doc_script.audience,
            narration_mode=doc_script.narration_mode,
            strategy=doc_script.strategy,
            central_question=doc_script.central_question,
            concept_analysis=doc_script.concept_analysis,
            beats=[b.model_copy(deep=True) for b in ch_beats],
            revisions=list(doc_script.revisions),
            approved_revision=doc_script.approved_revision,
            locked=doc_script.locked,
        )
        ch_target_s = float(ch.target_duration_s or per_ch_s)
        ch_cache_file = logs_dir / f"ch_{req_sig}_{ch.chapter_id}_valid.json"
        if ch_cache_file.exists():
            cached_cp = VideoPlanV3.model_validate_json(ch_cache_file.read_text(encoding="utf-8"))
            from studio.ai.validate_v3 import verify_v3_plan_detailed as _v3_det

            cached_det = _v3_det(cached_cp, require_claims=True)
            cached_cov = cached_det.get("verification_coverage", {}) if isinstance(cached_det, dict) else {}
            if not cached_det.get("issues") and int(cached_cov.get("unsupported", 0)) == 0:
                try:
                    if ch_beats:
                        lock_script_to_plan(cached_cp, ch_script, stage="approved", restore_locked_spoken_text=True, strict_ids=True)
                    if on_progress:
                        on_progress(f"reusing cached chapter {idx}/{len(lf_plan.chapters)}: {ch.title}")
                    return idx, cached_cp
                except ValueError:
                    pass
        if on_progress:
            on_progress(f"compiling visuals from approved script {idx}/{len(lf_plan.chapters)}: {ch.title} (~{ch_target_s:.0f}s)")
        node_ctx = "; ".join(f"{node_map[nid].title}: {[c.statement for c in node_map[nid].claims]}" for nid in ch.node_ids if nid in node_map)
        locked_spec_lines = "\n".join(
            f"- scene_id='{b.scene_id}' | speaker='{b.speaker}' | visual_intent='{b.visual_intent}' | claims={b.mathematical_claims} | spoken_text={json.dumps(b.spoken_text)}"
            for b in ch_beats
        )
        ch_prompt = (
            f"{ch_system}\n\n"
            f"Compile an exact VideoPlanV3 VISUAL PLAN bound 1-to-1 to the APPROVED ScriptIR beats below.\n"
            f"Overall Topic: {topic}\nChapter {idx}/{len(lf_plan.chapters)}: {ch.title}\n"
            f"Summary: {ch.summary}\nKnowledge context: {node_ctx}\n"
            f"Shared Continuity Variables: {json.dumps(lf_plan.continuity.defined_variables)}\n"
            "APPROVED BEATS (preserve exact `scene_id`, `speaker`, and `narration` verbatim; add deterministic constructions, visual objects, meaningful motion events, and verified StructuredClaim objects):\n"
            "CLAIM & CONSTRUCTION RULES:\n"
            "- NEVER attach polygon claims (`nonzero_area`, `square`, `right_triangle`, `area_conservation`) to `axes`, `parametric_curve`, `function_plot`, `point`, or `vector`.\n"
            "- For curves, waves, winding paths, spectra, and calculus/physics scenes use `function_equivalence`, `point_on_function`, `derivative_equality`, `definite_integral_value`, `coefficient_correctness`, or verified constructions (`fourier_series_partial_sum`, `continuous_fourier_spectrum`, `parametric_trajectory`, `riemann_sum`).\n"
            f"{locked_spec_lines}"
        )
        ch_plan = None
        issues: list[str] = []
        for _attempt in range(2):
            ch_plan, ch_resp = director.provider.generate_structured(ch_prompt, VideoPlanV3, system=ch_system, temperature=0.35)
            with log_lock:
                gen_log.append({"stage": f"chapter_{ch.chapter_id}", "model": ch_resp.model, "latency_s": ch_resp.latency_s})
            if ch_beats:
                lock_script_to_plan(ch_plan, ch_script, stage="approved", restore_locked_spoken_text=True, strict_ids=True)
            for r_idx in range(1, MAX_REPAIRS + 1):
                from studio.ai.validate_v3 import verify_v3_plan_detailed as _v3_det

                det_r = _v3_det(ch_plan, require_claims=True)
                issues = list(det_r.get("issues", []))
                for u_msg in det_r.get("unclaimed_math_scenes", []):
                    issues.append(str(u_msg))
                if not issues:
                    break
                fix_prompt = (
                    "Fix EVERY listed VideoPlanV3 validation issue strictly WITHOUT altering scene_id or spoken narration:\n- "
                    + "\n- ".join(issues)
                    + "\nRULES WHEN FIXING:\n"
                    "- Never attach polygon claims (`nonzero_area`, `square`, `right_triangle`, `area_conservation`) to non-polygons.\n"
                    "- For `markov_stochastic_invariant`, reference a `matrix` or `bar_chart` in `object_ids` whose entries verify row sums = 1.0 and invariant stationary distribution.\n"
                    "- Ensure every narrative mathematical claim has a matching verified StructuredClaim or ConstructionOp.\n"
                    f"\nCurrent JSON:\n{ch_plan.model_dump_json()}"
                )
                ch_plan, _ = director.provider.generate_structured(fix_prompt, VideoPlanV3, system=ch_system, temperature=0.25)
                if ch_beats:
                    lock_script_to_plan(ch_plan, ch_script, stage="approved", restore_locked_spoken_text=True, strict_ids=True)
                with log_lock:
                    gen_log.append({"stage": f"chapter_{ch.chapter_id}_repair_{r_idx}", "issues": issues})
            issues = validate_v3_plan(ch_plan, require_claims=True)
            if not issues:
                break

        if issues or ch_plan is None:
            _atomic_write_json(logs_dir / f"failed_{ch.chapter_id}.json", ch_plan.model_dump() if ch_plan else {})
            raise RuntimeError(f"Chapter {ch.chapter_id} failed validation after repairs: {issues}")
        mat = materialize_constructions(ch_plan)
        if ch_beats:
            lock_script_to_plan(mat, ch_script, stage="approved", restore_locked_spoken_text=True, strict_ids=True)
        post_issues = validate_v3_plan(mat, require_claims=True)
        if post_issues:
            raise RuntimeError(f"Post-script-lock validation failed in {ch.chapter_id}: {post_issues}")
        ch_cache_file.write_text(mat.model_dump_json(indent=2), encoding="utf-8")
        return idx, mat

    llm_workers = max(1, min(len(lf_plan.chapters), int(os.environ.get("STUDIO_LLM_WORKERS", "3"))))
    with ThreadPoolExecutor(max_workers=llm_workers) as pool:
        futs = [pool.submit(_compile_chapter_from_script, idx, ch) for idx, ch in enumerate(lf_plan.chapters, 1)]
        for fut in as_completed(futs):
            c_idx, mat = fut.result()
            ordered_plans[c_idx - 1] = mat

    chapter_plans: list[VideoPlanV3] = [cp for cp in ordered_plans if cp is not None]
    last_ch = lf_plan.chapters[-1] if lf_plan.chapters else None
    prev_summary = f"{last_ch.title}: {last_ch.summary}" if last_ch else ""

    # Calibrated duration check using true Kokoro effective WPS (~2.08 wps including beat pauses)
    est_s = sum(len(parse_script(s.narration)[0].split()) for cp in chapter_plans for s in cp.scenes) / KOKORO_EFFECTIVE_WPS + len(chapter_plans) * 1.2
    extra_idx = 1
    base_chapter_count = len(lf_plan.chapters)
    while est_s < target_duration_s * 0.90 and extra_idx <= 4:
        ext_id = f"ext_{extra_idx}"
        ext_title = f"Deep Dive {extra_idx}: Case Studies & Edge Regimes in {topic}"
        ext_target_s = min(90.0, max(45.0, target_duration_s - est_s))
        ext_min_words = max(75, int(ext_target_s * 1.65))
        ext_max_words = max(ext_min_words + 30, int(ext_target_s * 2.25))
        if on_progress:
            on_progress(f"extending duration ({est_s:.0f}s < 90% of {target_duration_s:.0f}s): {ext_title}")
        ext_cache_file = logs_dir / f"ch_{req_sig}_{ext_id}_valid.json"
        mat = None
        if ext_cache_file.exists():
            try:
                cand_ext = VideoPlanV3.model_validate_json(ext_cache_file.read_text(encoding="utf-8"))
                from studio.ai.validate_v3 import verify_v3_plan_detailed as _v3_det

                c_det = _v3_det(cand_ext, require_claims=True)
                if not c_det.get("issues") and not c_det.get("unclaimed_math_scenes"):
                    if all(len(parse_script(s.narration)[0].split()) <= 45 for s in cand_ext.scenes):
                        mat = cand_ext
            except Exception:
                mat = None

        if mat is None:
            ext_prompt = (
                f"{ch_system}\n\n"
                f"Overall Topic: {topic}\nContinuation Chapter: {ext_title}\n"
                f"Previous Chapter Context: {prev_summary}\n"
                f"Shared Continuity Variables: {json.dumps(lf_plan.continuity.defined_variables)}\n"
                f"Target duration: {ext_target_s:.0f}s ({ext_min_words}-{ext_max_words} spoken words across 6-8 concise scenes; EVERY single scene narration MUST be <= 40 words).\n"
                "Write a complete VideoPlanV3 deepening intuition through contrasting concrete regimes, deterministic constructions, and verifiable claims."
            )
            ch_plan, _ch_resp = director.provider.generate_structured(ext_prompt, VideoPlanV3, system=ch_system, temperature=0.4)
            ext_issues: list[str] = []
            for r_idx in range(1, MAX_REPAIRS + 1):
                from studio.ai.validate_v3 import verify_v3_plan_detailed as _v3_det

                det_e = _v3_det(ch_plan, require_claims=True)
                ext_issues = list(det_e.get("issues", []))
                for u_msg in det_e.get("unclaimed_math_scenes", []):
                    ext_issues.append(str(u_msg))
                for s_chk in ch_plan.scenes:
                    wc_s = len(parse_script(s_chk.narration)[0].split())
                    if wc_s > 45:
                        ext_issues.append(f"scene {s_chk.scene_id}: narration {wc_s} words exceeds 45-word documentary turn limit; split or tighten.")
                w_cnt = sum(len(parse_script(s.narration)[0].split()) for s in ch_plan.scenes)
                if w_cnt < ext_min_words:
                    ext_issues.append(f"Narration too brief ({w_cnt} words < {ext_min_words}); add another concise scene.")
                elif w_cnt > ext_max_words:
                    ext_issues.append(f"Narration too long ({w_cnt} words > {ext_max_words}); tighten phrasing.")
                if not ext_issues:
                    break
                fix_prompt = "Fix EVERY listed VideoPlanV3 issue:\n- " + "\n- ".join(ext_issues) + f"\n\nCurrent JSON:\n{ch_plan.model_dump_json()}"
                ch_plan, _ = director.provider.generate_structured(fix_prompt, VideoPlanV3, system=ch_system, temperature=0.25)
            # Final re-check after loop
            from studio.ai.validate_v3 import verify_v3_plan_detailed as _v3_det

            det_final = _v3_det(ch_plan, require_claims=True)
            ext_ok = (
                not det_final.get("issues")
                and not det_final.get("unclaimed_math_scenes")
                and all(len(parse_script(s.narration)[0].split()) <= 48 for s in ch_plan.scenes)
            )
            if ext_ok:
                mat = materialize_constructions(ch_plan)
                ext_cache_file.write_text(mat.model_dump_json(indent=2), encoding="utf-8")

        if mat is not None:
            lf_plan.chapters.append(ChapterSpec(chapter_id=ext_id, title=ext_title, node_ids=[], target_duration_s=ext_target_s))
            doc_script.locked = False
            for s_item in mat.scenes:
                s_item.narration = clean_spoken_text(s_item.narration)
                nb = NarrationBeatIR(
                    beat_id=f"{ext_id}__{s_item.scene_id}",
                    chapter_id=ext_id,
                    scene_id=s_item.scene_id,
                    speaker=s_item.speaker if s_item.speaker in ("narrator", "educator", "learner", "host_a", "host_b") else "narrator",  # type: ignore[arg-type]
                    spoken_text=s_item.narration,
                    educational_purpose=s_item.purpose,
                    mathematical_claims=[c.statement or c.kind for c in s_item.claims],
                    visual_intent=", ".join(f"{o.id}:{o.kind}" for o in s_item.objects[:5]),
                    estimated_duration_s=round(max(2.5, len(parse_script(s_item.narration)[0].split()) / KOKORO_EFFECTIVE_WPS + 0.35), 2),
                )
                sync_beat_anchors(nb)
                doc_script.beats.append(nb)
            chapter_plans.append(mat)
            est_s += sum(len(parse_script(s.narration)[0].split()) for s in mat.scenes) / KOKORO_EFFECTIVE_WPS + 1.2
            prev_summary = ext_title
        extra_idx += 1

    if len(chapter_plans) > base_chapter_count:
        cdir_ext = CreativeDirector(model=model, provider_name=provider_name, free_only=free_only)
        rep_ext = cdir_ext.review_script(doc_script, use_llm=False)
        if not rep_ext.approved:
            doc_script = cdir_ext.repair_script(doc_script, rep_ext, target_duration_s=target_duration_s)
            for b in doc_script.beats:
                if not b.chapter_id:
                    b.chapter_id = b.beat_id.split("__", 1)[0] if "__" in b.beat_id else "c1"
                b.beat_id = f"{b.chapter_id}__{b.scene_id}"
                b.spoken_text = clean_spoken_text(b.spoken_text)
        approve_script_ir(doc_script, notes="Re-approved ScriptIR after duration extension chapters")
        for cp_ext, ch_spec in zip(chapter_plans, lf_plan.chapters):
            sub_beats = [b for b in doc_script.beats if b.chapter_id == ch_spec.chapter_id]
            if sub_beats:
                sub_ir = ScriptIR(
                    topic=doc_script.topic,
                    title=ch_spec.title,
                    audience=doc_script.audience,
                    narration_mode=doc_script.narration_mode,
                    beats=sub_beats,
                )
                lock_script_to_plan(cp_ext, sub_ir, stage="approved", restore_locked_spoken_text=True, strict_ids=False)
    (lf_dir / "SCRIPT.md").write_text(export_script_md(doc_script), encoding="utf-8")
    _atomic_write_json(lf_dir / "script_ir_approved.json", doc_script.model_dump())

    _atomic_write_json(logs_dir / "generation_log.json", gen_log)
    return lf_plan, chapter_plans


def build_longform_video(
    slug: str,
    longform_plan: LongFormPlan,
    chapter_plans: list[VideoPlanV3],
    *,
    format_name: str = "widescreen_hd",
    tts_engine: str = "kokoro",
    allow_edge_fallback: bool = True,
    voice: str = "en-US-AndrewNeural",
    rate: str = "-4%",
    pitch: str = "-2Hz",
    ui: str = "minimal",
    look: str = "cinematic",
    look_name: str = "midnight_studio",
    backend: str = "manimgx",
    caption_style: str | None = None,
    min_duration_s: float = 0.0,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Render chapters independently with multi-stage dependency cache and assemble a verified documentary."""
    if len(longform_plan.chapters) != len(chapter_plans):
        raise ValueError("chapters count must match chapter_plans count.")

    resolved_look = resolve_look(look_name).name
    resolved_cap = caption_style or ("kinetic" if "shorts" in format_name else "editorial")

    lf_dir = PROJECT_ROOT / "build" / f"longform_{slug}"
    lf_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = lf_dir / "checkpoint.json"
    checkpoint: dict[str, dict[str, str]] = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}

    (lf_dir / "longform_plan.json").write_text(longform_plan.model_dump_json(indent=2), encoding="utf-8")
    design = default_design_spec(format_name=format_name, look_name=resolved_look)
    (lf_dir / "DESIGN.md").write_text(export_design_md(design), encoding="utf-8")

    jobs = JobStore(PROJECT_ROOT / "build" / "studio_jobs.db")
    acache = ArtifactCache(store=jobs)
    know_key = ArtifactCache.compute_stage_key("knowledge", longform_plan.knowledge_tree.model_dump())
    design_key = ArtifactCache.compute_stage_key("design", design.model_dump())

    chapter_mp4s: list[Path] = []
    chapter_srts: list[Path] = []
    chapter_transcripts: list[Path] = []
    chapter_builds: list[Path] = []

    try:
        from studio.audio.tts import BeatSpec, build_voice_timeline
        from studio.timeline import MasterTimeline

        sp_voices = resolve_speaker_voices("documentary", voice_override=voice if voice != "en-US-AndrewNeural" else None)

        pending_jobs: list[tuple[int, ChapterSpec, VideoPlanV3, Path, str, str, dict[str, str], Path]] = []
        slots_mp4: list[Path | None] = [None] * len(longform_plan.chapters)
        slots_srt: list[Path | None] = [None] * len(longform_plan.chapters)
        slots_txt: list[Path | None] = [None] * len(longform_plan.chapters)
        slots_bdir: list[Path | None] = [None] * len(longform_plan.chapters)
        hf_timelines: dict[int, object] = {}

        for idx, (ch, plan) in enumerate(zip(longform_plan.chapters, chapter_plans)):
            issues = validate_v3_plan(plan)
            if issues:
                raise ValueError(f"Chapter {ch.chapter_id} invalid:\n- " + "\n- ".join(issues))

            ch_plan_path = lf_dir / f"{slug}_{ch.chapter_id}.json"
            ch_plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

            fps_map = compute_dependency_fingerprints(PROJECT_ROOT)
            math_key = ArtifactCache.compute_stage_key(
                "math_model",
                {"plan": plan.model_dump(), "math_fp": f"{fps_map['math_compute']}:{fps_map['math_validate']}"},
                upstream_keys={"knowledge": know_key},
            )
            ckey = chapter_cache_key(
                plan,
                format_name=format_name,
                tts_engine=tts_engine,
                voice=voice,
                rate=rate,
                pitch=pitch,
                ui=ui,
                look=look,
                look_name=resolved_look,
                caption_style=resolved_cap,
                backend=backend,
            )
            upstream = {"knowledge": know_key, "math_model": math_key, "design": design_key}
            stage_hit, stage_key, stage_path, _reason = acache.lookup(
                "chapter_assembly",
                f"{slug}_{ch.chapter_id}_{format_name}_{resolved_look}_{resolved_cap}_{backend}",
                {"ckey": ckey, "backend": backend},
                upstream_keys=upstream,
            )

            ch_stem = f"V3_{ch_plan_path.stem}"
            ch_build = (
                PROJECT_ROOT / "build" / f"backend_hyperframes_{slug}_{ch.chapter_id}_{format_name}"
                if backend == "hyperframes"
                else PROJECT_ROOT / "build" / f"{ch_stem}_{format_name}"
            )
            srt_p = ch_build / f"{ch_stem}.srt"
            txt_p = ch_build / f"{ch_stem}_transcript.txt"
            cap_ir_p = ch_build / "caption_ir.json"
            slots_srt[idx] = srt_p
            slots_txt[idx] = txt_p
            slots_bdir[idx] = ch_build

            # Enforce strict SHA-256 + sidecar bundle + stream verification
            sidecars_ok, _sc_msg = validate_sidecar_bundle([srt_p, txt_p, cap_ir_p]) if stage_hit else (False, "miss")
            cached = stage_path if (stage_hit and sidecars_ok) else None
            if cached and verify_media_streams(cached).passed:
                mp4 = cached
                if on_progress:
                    on_progress(f"content-cache hit [{ckey[:8]}]: {ch.chapter_id} -> {mp4.name}")
                if backend in ("hyperframes", "hybrid") and (not srt_p.exists() or not txt_p.exists()):
                    tl = build_voice_timeline(
                        [BeatSpec(id=s.scene_id, narration=s.narration, caption=s.purpose, speaker=s.speaker) for s in plan.scenes],
                        build_dir=ch_build,
                        voice=voice,
                        rate=rate,
                        pitch=pitch,
                        engine=tts_engine,
                        allow_edge_fallback=allow_edge_fallback,
                        speaker_voices=sp_voices,
                    )
                    MasterTimeline(audio=tl).export_subtitles(ch_build, ch_stem, caption_style=resolved_cap, format_name=format_name, look_name=resolved_look)
                slots_mp4[idx] = mp4
                checkpoint[ch.chapter_id] = {"cache_key": ckey, "mp4": str(mp4), "backend": backend}
                _atomic_write_json(ckpt_path, checkpoint)
            else:
                if backend in ("hyperframes", "hybrid"):
                    # Pre-synthesize once in main thread to keep Kokoro warm in RAM and avoid double TTS
                    tl = build_voice_timeline(
                        [BeatSpec(id=s.scene_id, narration=s.narration, caption=s.purpose, speaker=s.speaker) for s in plan.scenes],
                        build_dir=ch_build,
                        voice=voice,
                        rate=rate,
                        pitch=pitch,
                        engine=tts_engine,
                        allow_edge_fallback=allow_edge_fallback,
                        speaker_voices=sp_voices,
                    )
                    MasterTimeline(audio=tl).export_subtitles(ch_build, ch_stem, caption_style=resolved_cap, format_name=format_name, look_name=resolved_look)
                    hf_timelines[idx] = tl
                pending_jobs.append((idx, ch, plan, ch_plan_path, ckey, stage_key, upstream, ch_build))

        if pending_jobs:
            if backend in ("hyperframes", "hybrid"):
                from studio.renderers import get_backend

                def _render_hf(job: tuple[int, ChapterSpec, VideoPlanV3, Path, str, str, dict[str, str], Path]) -> tuple[int, ChapterSpec, str, str, dict[str, str], Path]:
                    j_idx, j_ch, j_plan, _jp, j_ckey, j_skey, j_up, j_bdir = job
                    if on_progress:
                        on_progress(f"rendering chapter [{j_ckey[:8]} via {backend}]: {j_ch.chapter_id} ({j_ch.title})")
                    prefix = "HYB" if backend == "hybrid" else "HF"
                    hf_res = get_backend(backend).render(
                        j_plan,
                        slug=f"{slug}_{j_ch.chapter_id}",
                        format_name=format_name,
                        look_name=resolved_look,
                        audio_timeline=hf_timelines.get(j_idx),  # type: ignore[arg-type]
                        build_dir=j_bdir,
                        out_path=PROJECT_ROOT / "output" / f"{prefix}_{slug}_{j_ch.chapter_id}_{format_name}.mp4",
                        caption_style=resolved_cap,
                        tts_engine=tts_engine,
                        allow_edge_fallback=allow_edge_fallback,
                        voice=voice,
                        rate=rate,
                        pitch=pitch,
                    )
                    return j_idx, j_ch, j_ckey, j_skey, j_up, hf_res.video_path

                hf_workers = max(1, min(len(pending_jobs), int(os.environ.get("STUDIO_HF_WORKERS", str(min(3, max(1, (os.cpu_count() or 4) // 2)))))))
                with ThreadPoolExecutor(max_workers=hf_workers) as pool:
                    for j_idx, j_ch, j_ckey, j_skey, j_up, mp4 in pool.map(_render_hf, pending_jobs):
                        jobs.put_cached(j_ckey, mp4)
                        acache.store_artifact(
                            "chapter_assembly",
                            f"{slug}_{j_ch.chapter_id}_{format_name}_{resolved_look}_{resolved_cap}_{backend}",
                            j_skey,
                            mp4,
                            upstream_keys=j_up,
                        )
                        slots_mp4[j_idx] = mp4
                        checkpoint[j_ch.chapter_id] = {"cache_key": j_ckey, "mp4": str(mp4), "backend": backend}
                        _atomic_write_json(ckpt_path, checkpoint)
            else:
                for j_idx, j_ch, _j_plan, j_plan_path, j_ckey, j_skey, j_up, _j_bdir in pending_jobs:
                    if on_progress:
                        on_progress(f"rendering chapter [{j_ckey[:8]} via {backend}]: {j_ch.chapter_id} ({j_ch.title})")
                    mp4 = run_plan(
                        j_plan_path,
                        format_name=format_name,
                        theme_name=resolved_look,
                        tts_engine=tts_engine,
                        allow_edge_fallback=allow_edge_fallback,
                        voice=voice,
                        rate=rate,
                        pitch=pitch,
                        ui=ui,
                        caption_style=resolved_cap,
                        look=look,
                        strict_layout=False,
                    )
                    jobs.put_cached(j_ckey, mp4)
                    acache.store_artifact(
                        "chapter_assembly",
                        f"{slug}_{j_ch.chapter_id}_{format_name}_{resolved_look}_{resolved_cap}_{backend}",
                        j_skey,
                        mp4,
                        upstream_keys=j_up,
                    )
                    slots_mp4[j_idx] = mp4
                    checkpoint[j_ch.chapter_id] = {"cache_key": j_ckey, "mp4": str(mp4), "backend": backend}
                    _atomic_write_json(ckpt_path, checkpoint)

        chapter_mp4s = [p for p in slots_mp4 if p is not None]
        chapter_srts = [p for p in slots_srt if p is not None]
        chapter_transcripts = [p for p in slots_txt if p is not None]
        chapter_builds = [p for p in slots_bdir if p is not None]
    finally:
        cache_stats = acache.stats()
        jobs.close()

    chapters_meta: list[dict[str, object]] = []
    yt_lines: list[str] = []
    merged_srt: list[str] = []
    merged_vtt: list[str] = ["WEBVTT", ""]
    merged_transcript: list[str] = []
    srt_idx = 1
    cursor_s = 0.0
    all_shots = []

    for ch, cp_plan, mp4, srt_p, txt_p, c_bdir in zip(
        longform_plan.chapters, chapter_plans, chapter_mp4s, chapter_srts, chapter_transcripts, chapter_builds
    ):
        dur = _probe_duration(mp4)
        chapters_meta.append(
            {"chapter_id": ch.chapter_id, "title": ch.title, "start_s": round(cursor_s, 3), "end_s": round(cursor_s + dur, 3), "duration_s": round(dur, 3)}
        )
        yt_lines.append(f"{_fmt_yt_ts(cursor_s)} {ch.title}")
        tl_path = c_bdir / "timeline.json"
        ch_tl = AudioTimeline.load(tl_path) if tl_path.exists() else None
        ch_sb = storyboard_from_v3_plan(cp_plan, format_name=format_name, look_name=resolved_look, audio_timeline=ch_tl)
        for sh in ch_sb.shots:
            sh.shot_id = f"{ch.chapter_id}_{sh.shot_id}"
            sh.start_s = round(sh.start_s + cursor_s, 3)
            sh.end_s = round(sh.end_s + cursor_s, 3)
            all_shots.append(sh)

        if not srt_p.exists() or not txt_p.exists():
            raise RuntimeError(f"Missing required chapter subtitle/transcript for {ch.chapter_id}: {srt_p}, {txt_p}")

        blocks = srt_p.read_text(encoding="utf-8").strip().split("\n\n")
        for blk in blocks:
            lines = blk.splitlines()
            if len(lines) >= 3 and "-->" in lines[1]:
                a_ts, b_ts = [x.strip() for x in lines[1].split("-->")]
                sa, sb = _shift_srt_ts(a_ts, cursor_s), _shift_srt_ts(b_ts, cursor_s)
                body = "\n".join(lines[2:])
                merged_srt.extend([str(srt_idx), f"{sa} --> {sb}", body, ""])
                merged_vtt.extend([f"{sa.replace(',', '.')} --> {sb.replace(',', '.')}", body, ""])
                srt_idx += 1

        merged_transcript.append(f"## {ch.title}\n\n{txt_p.read_text(encoding='utf-8').strip()}")
        cursor_s += dur

    concat_list = lf_dir / "concat_list.txt"
    concat_list.write_text("\n".join(f"file '{p.as_posix()}'" for p in chapter_mp4s), encoding="utf-8")

    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = out_dir / f"LF_{slug}_{format_name}.mp4"
    tmp_mp4 = out_dir / f"LF_{slug}_{format_name}.tmp.mp4"
    ffmpeg_bin = find_ffmpeg()
    subprocess.run(
        [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(tmp_mp4)],
        check=True,
        capture_output=True,
    )
    os.replace(tmp_mp4, final_mp4)

    chapters_json = lf_dir / f"{slug}_chapters.json"
    chapters_txt = lf_dir / f"{slug}_chapters.txt"
    srt_out = lf_dir / f"{slug}.srt"
    vtt_out = lf_dir / f"{slug}.vtt"
    txt_out = lf_dir / f"{slug}_transcript.txt"

    _atomic_write_json(chapters_json, chapters_meta)
    chapters_txt.write_text("\n".join(yt_lines), encoding="utf-8")
    srt_out.write_text("\n".join(merged_srt), encoding="utf-8")
    vtt_out.write_text("\n".join(merged_vtt), encoding="utf-8")
    txt_out.write_text("\n\n".join(merged_transcript), encoding="utf-8")

    # Export unified documentary STORYBOARD.md and REVIEW.md
    from studio.ai.v3_schema import VideoPlanV3 as _VP3
    from studio.storyboard.model import Storyboard

    merged_plan = _VP3(
        topic=longform_plan.topic,
        title=longform_plan.topic,
        scenes=[sc for cp in chapter_plans for sc in cp.scenes],
    )
    full_sb = Storyboard(
        topic=longform_plan.topic,
        title=longform_plan.topic,
        format_name=format_name,
        aspect_label="9:16" if "shorts" in format_name else "16:9",
        look_name=look_name,
        total_duration_s=round(cursor_s, 3),
        shots=all_shots,
    )
    export_storyboard_md(full_sb, lf_dir / "STORYBOARD.md")

    bundle_chk = verify_longform_bundle(
        final_mp4, chapters_json, srt_out, vtt_out, txt_out, min_duration_s=min_duration_s
    )
    if not bundle_chk.passed:
        raise RuntimeError(f"Long-form final bundle QA failed: {bundle_chk.detail}")

    approved_script_p = lf_dir / "script_ir_approved.json"
    if not approved_script_p.exists():
        # Check chapter build/script outputs or longform dir
        cand_p = PROJECT_ROOT / "build" / f"longform_{slug}" / "script_ir_approved.json"
        if cand_p.exists():
            approved_script_p = cand_p
    doc_script = None
    if approved_script_p.exists():
        try:
            doc_script = ScriptIR.model_validate_json(approved_script_p.read_text(encoding="utf-8"))
        except Exception:
            doc_script = None
    first_cap_ir = (chapter_builds[0] / "caption_ir.json") if chapter_builds and (chapter_builds[0] / "caption_ir.json").exists() else None
    loud = verify_audio_loudness(final_mp4)
    streams = verify_media_streams(final_mp4)
    base_target = sum(float(c.target_duration_s or 0.0) for c in longform_plan.chapters if not str(c.chapter_id).startswith("ext_"))
    target_sum = (base_target if base_target > 0 else sum(float(c.target_duration_s or 0.0) for c in longform_plan.chapters)) or None
    rev_report = audit_educational_quality(
        topic=longform_plan.topic,
        title=longform_plan.topic,
        plan=merged_plan,
        format_name=format_name,
        backend=backend,
        video_path=final_mp4,
        contact_sheet_out=lf_dir / f"QA_CONTACT_SHEET_{resolved_look}.png",
        script=doc_script,
        look_name=resolved_look,
        chapter_plans=list(chapter_plans),
        target_duration_s=target_sum,
        caption_ir_path=first_cap_ir,
        srt_path=srt_out,
    )
    export_review_md(rev_report, lf_dir / "REVIEW.md")  # type: ignore[arg-type]

    return {
        "output_mp4": str(final_mp4),
        "total_duration_s": round(cursor_s, 3),
        "chapters": chapters_meta,
        "chapters_txt": str(chapters_txt),
        "srt": str(srt_out),
        "vtt": str(vtt_out),
        "transcript": str(txt_out),
        "storyboard_md": str(lf_dir / "STORYBOARD.md"),
        "review_md": str(lf_dir / "REVIEW.md"),
        "design_md": str(lf_dir / "DESIGN.md"),
        "contact_sheet": str(getattr(rev_report, "contact_sheet_path", "")),
        "production_state": str(getattr(rev_report, "production_state", "review_required")),
        "qa_review_approved": bool(getattr(rev_report, "approved", False)),
        "qa_review_score": float(getattr(rev_report, "overall_score", 0.0)),
        "cache_stats": cache_stats,
        "qa_loudness": {"passed": loud.passed, "detail": loud.detail},
        "qa_streams": {"passed": streams.passed, "detail": streams.detail},
        "qa_bundle": {"passed": bundle_chk.passed, "detail": bundle_chk.detail},
    }


def derive_recomposed_short(
    slug: str,
    longform_plan: LongFormPlan,
    chapter_plans: list[VideoPlanV3],
    *,
    chapter_index: int | None = None,
    format_name: str = "shorts_hd",
    tts_engine: str = "kokoro",
    allow_edge_fallback: bool = True,
    voice: str = "en-US-AndrewNeural",
    rate: str = "-4%",
    pitch: str = "-2Hz",
    ui: str = "minimal",
    look: str = "cinematic",
    look_name: str = "midnight_studio",
    backend: str = "manimgx",
    caption_style: str | None = None,
) -> dict[str, object]:
    """Derive a native recomposed 9:16 portrait Short from the core chapter (no 16:9 crop)."""
    resolved_look = resolve_look(look_name).name
    resolved_cap = caption_style or ("kinetic" if "shorts" in format_name else "editorial")
    if not chapter_plans:
        raise ValueError("Need at least one chapter plan to derive a recomposed Short.")
    # Choose highest-value contiguous 3..6-scene climax window (~75..115 words) with real motion ops & claims
    _anim_ops = {"transform", "morph", "move", "rotate", "scale", "trace", "animate_param", "iterate_states", "interpolate_values"}
    from studio.renderers.scene_state import SceneRuntimeState, evaluate_plan_states, materialize_carry_objects

    best_score = -10**9
    best_win = (0, 0, min(4, len(chapter_plans[0].scenes)))
    chap_indices = [chapter_index] if chapter_index is not None else list(range(len(chapter_plans)))
    for c_idx in chap_indices:
        cp = chapter_plans[c_idx]
        is_ext = c_idx < len(longform_plan.chapters) and str(longform_plan.chapters[c_idx].chapter_id).startswith("ext_")
        for s_start in range(len(cp.scenes)):
            w_acc = 0
            for s_end in range(s_start + 1, min(len(cp.scenes), s_start + 6) + 1):
                w_acc += len(cp.scenes[s_end - 1].narration.split())
                if w_acc > 120:
                    break
                win_scenes = cp.scenes[s_start:s_end]
                has_anim = any(e.op in _anim_ops for sc in win_scenes for e in sc.events)
                uniq_kinds = len({o.kind for sc in win_scenes for o in sc.objects})
                claims_cnt = sum(len(sc.claims) for sc in win_scenes)
                con_cnt = sum(len(sc.constructions) for sc in win_scenes)
                sc_score = (
                    (120 if has_anim else 0)
                    + (45 if 75 <= w_acc <= 115 else (15 if 60 <= w_acc <= 120 else -30))
                    + claims_cnt * 6
                    + con_cnt * 4
                    + uniq_kinds * 5
                    - (25 if is_ext else 0)
                )
                if sc_score > best_score:
                    best_score = sc_score
                    best_win = (c_idx, s_start, s_end)

    chapter_index, w_start, w_end = best_win
    src = chapter_plans[chapter_index].model_copy(deep=True)
    prior_state = evaluate_plan_states(src)[w_start - 1] if w_start > 0 else SceneRuntimeState()
    kept_scenes = []
    declared_ids: set[str] = set()
    word_budget = 0
    for idx_w, sc in enumerate(src.scenes[w_start:w_end]):
        sc_copy = sc.model_copy(deep=True)
        sc_copy.composition = adapt_composition_for_aspect(sc_copy.composition, "9:16")  # type: ignore[assignment]
        for o in sc_copy.objects:
            if o.panel_slot == "left":
                o.panel_slot = "top"
            elif o.panel_slot == "right":
                o.panel_slot = "bottom"
            declared_ids.add(o.id)
        if idx_w == 0 and w_start > 0:
            needed = declared_ids | {o.axes_id for o in sc_copy.objects if o.axes_id} | {
                e.target_id for e in sc_copy.events if e.target_id
            } | {e.source_id for e in sc_copy.events if e.source_id}
            carried = [o for o in materialize_carry_objects(prior_state, needed, include_visible_persistent=False) if o.id not in declared_ids]
            for co in carried:
                declared_ids.add(co.id)
            sc_copy.objects = carried + list(sc_copy.objects)
        sc_copy.events = [e for e in sc_copy.events if e.target_id in declared_ids and (not e.source_id or e.source_id in declared_ids)]
        kept_scenes.append(sc_copy)
        word_budget += len(sc_copy.narration.split())

    short_target_s = round(max(28.0, min(58.0, word_budget / 2.20 + len(kept_scenes) * 0.25)), 1)
    short_plan = VideoPlanV3(
        schema_version="3.0",
        topic=f"{longform_plan.topic} — Key Insight",
        title=kept_scenes[0].purpose if kept_scenes else longform_plan.topic,
        audience=src.audience,
        strategy=src.strategy,
        target_duration_s=short_target_s,
        scenes=kept_scenes,
    )
    short_dir = PROJECT_ROOT / "build" / f"short_from_{slug}"
    short_dir.mkdir(parents=True, exist_ok=True)
    design = default_design_spec(format_name=format_name, look_name=resolved_look)
    (short_dir / "DESIGN.md").write_text(export_design_md(design), encoding="utf-8")
    short_json = short_dir / f"v3_{slug}_short.json"
    short_json.write_text(json.dumps(short_plan.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

    sb = storyboard_from_v3_plan(short_plan, format_name=format_name, look_name=resolved_look)
    export_storyboard_md(sb, short_dir / "STORYBOARD.md")

    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_mp4 = out_dir / f"SHORT_{slug}_{format_name}.mp4"
    if backend in ("hyperframes", "hybrid"):
        from studio.renderers import get_backend

        res = get_backend(backend).render(
            short_plan,
            slug=f"{slug}_derived_short",
            format_name=format_name,
            look_name=resolved_look,
            out_path=target_mp4,
            caption_style=resolved_cap,
            tts_engine=tts_engine,
            allow_edge_fallback=allow_edge_fallback,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )
        mp4 = res.video_path
    else:
        rendered = run_plan(
            short_json,
            format_name=format_name,
            theme_name=resolved_look,
            tts_engine=tts_engine,
            allow_edge_fallback=allow_edge_fallback,
            voice=voice,
            rate=rate,
            pitch=pitch,
            ui=ui,
            caption_style=resolved_cap,
            look=look,
            strict_layout=False,
        )
        if rendered.resolve() != target_mp4.resolve():
            target_mp4.write_bytes(rendered.read_bytes())
        mp4 = target_mp4

    loud = verify_audio_loudness(mp4)
    streams = verify_media_streams(mp4)
    v3_short_bdir = PROJECT_ROOT / "build" / f"V3_{short_json.stem}_{format_name}"
    short_cap_ir = next(
        (p for p in (v3_short_bdir / "caption_ir.json", short_dir / "caption_ir.json") if p.exists()),
        short_dir / "caption_ir.json",
    )
    short_srt = next(
        (p for p in (v3_short_bdir / f"V3_{short_json.stem}.srt", mp4.with_suffix(".srt")) if p.exists()),
        mp4.with_suffix(".srt"),
    )
    rev = audit_educational_quality(
        topic=short_plan.topic,
        title=short_plan.title,
        plan=short_plan,
        format_name=format_name,
        backend=backend,
        video_path=mp4,
        contact_sheet_out=short_dir / f"QA_CONTACT_SHEET_{resolved_look}.png",
        look_name=resolved_look,
        target_duration_s=short_target_s,
        caption_ir_path=short_cap_ir if short_cap_ir.exists() else None,
        srt_path=short_srt if short_srt.exists() else None,
    )
    export_review_md(rev, short_dir / "REVIEW.md")  # type: ignore[arg-type]
    return {
        "output_mp4": str(mp4),
        "source_chapter": longform_plan.chapters[chapter_index].chapter_id,
        "duration_s": _probe_duration(mp4),
        "storyboard_md": str(short_dir / "STORYBOARD.md"),
        "review_md": str(short_dir / "REVIEW.md"),
        "production_state": str(getattr(rev, "production_state", "review_required")),
        "qa_review_approved": bool(getattr(rev, "approved", False)),
        "qa_review_score": float(getattr(rev, "overall_score", 0.0)),
        "qa_loudness": {"passed": loud.passed, "detail": loud.detail},
        "qa_streams": {"passed": streams.passed, "detail": streams.detail},
    }
