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
from studio.creative.script_ir import NarrationBeatIR, ScriptIR, export_script_md, lock_script_to_plan
from studio.design.spec import default_design_spec, export_design_md, resolve_look
from studio.pipeline import PROJECT_ROOT, run_plan
from studio.qa.engine import audit_educational_quality, verify_audio_loudness, verify_longform_bundle, verify_media_streams
from studio.registry import capabilities_prompt_summary
from studio.registry.compositions import adapt_composition_for_aspect
from studio.storage.cache import ArtifactCache, JobStore
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
    h = hashlib.sha256()
    for rel in (
        "scenes/v3_scene.py",
        "studio/ai/world_model.py",
        "studio/layout/components.py",
        "studio/layout/solver.py",
        "studio/audio/bgm.py",
        "studio/audio/master.py",
        "studio/audio/sfx.py",
    ):
        p = PROJECT_ROOT / rel
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


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
) -> str:
    """Compute content-addressed cache key covering plan, math model, audio, theme, captions, and code."""
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
            "src_digest": _source_code_digest(),
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
        {"topic": topic, "dur": target_duration_s, "aud": audience, "mode": narration_mode, "n": n_chapters, "v": "v37"}
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

    log_lock = threading.Lock()
    ordered_plans: list[VideoPlanV3 | None] = [None] * len(lf_plan.chapters)

    def _script_chapter(idx: int, ch: ChapterSpec) -> tuple[int, VideoPlanV3]:
        ch_target_s = float(ch.target_duration_s or per_ch_s)
        ch_min_words = max(55, int(ch_target_s * 1.85))
        ch_max_words = max(ch_min_words + 25, int(ch_target_s * 2.28))
        ch_cache_file = logs_dir / f"ch_{req_sig}_{ch.chapter_id}_valid.json"
        if ch_cache_file.exists():
            cached_cp = VideoPlanV3.model_validate_json(ch_cache_file.read_text(encoding="utf-8"))
            if not validate_v3_plan(cached_cp, require_claims=True):
                if on_progress:
                    on_progress(f"reusing cached chapter {idx}/{len(lf_plan.chapters)}: {ch.title}")
                return idx, cached_cp
        if on_progress:
            on_progress(f"scripting chapter {idx}/{len(lf_plan.chapters)}: {ch.title} (~{ch_target_s:.0f}s)")
        prev_fallback = (
            f"Previous Chapter ({lf_plan.chapters[idx - 2].title}): {lf_plan.chapters[idx - 2].summary}"
            if idx >= 2
            else "Opening of documentary: establish the concrete central paradox cleanly."
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
        ch_prompt = (
            f"{ch_system}\n\n"
            f"Overall Topic: {topic}\n"
            f"Central Question: {brief.concept.central_question or outline.central_question}\n"
            f"Selected Strategy: {brief.selected_strategy}\n"
            f"Chapter {idx}/{len(lf_plan.chapters)}: {ch.title}\n"
            f"Summary: {ch.summary}\n"
            f"{bridge_ctx}"
            f"Knowledge context: {node_ctx}\n"
            f"Shared Continuity Variables: {json.dumps(lf_plan.continuity.defined_variables)}\n"
            f"Narration mode: {narration_mode}. Target duration: {ch_target_s:.0f}s "
            f"(STRICT WORD BUDGET: {ch_min_words}-{ch_max_words} spoken words across 5-9 crisp scenes; do NOT exceed {ch_max_words} words).\n"
            "Write a deep, natural, cliche-free self-contained VideoPlanV3 for this chapter using deterministic constructions, varied compositions, and verifiable claims."
        )
        ch_plan = None
        issues: list[str] = []
        for _attempt in range(2):
            ch_plan, ch_resp = director.provider.generate_structured(ch_prompt, VideoPlanV3, system=ch_system, temperature=0.45)
            with log_lock:
                gen_log.append({"stage": f"chapter_{ch.chapter_id}", "model": ch_resp.model, "latency_s": ch_resp.latency_s})
            for r_idx in range(1, MAX_REPAIRS + 1):
                issues = validate_v3_plan(ch_plan, require_claims=True)
                w_cnt = sum(len(parse_script(s.narration)[0].split()) for s in ch_plan.scenes)
                if w_cnt < ch_min_words:
                    issues.append(f"Chapter narration too thin ({w_cnt} words < {ch_min_words} minimum for ~{ch_target_s:.0f}s); expand intuitive reasoning and step-by-step explanations.")
                elif w_cnt > ch_max_words:
                    issues.append(f"Chapter narration too long ({w_cnt} words > {ch_max_words} ceiling for ~{ch_target_s:.0f}s); tighten prose crisply without cutting mathematical facts.")
                if not issues:
                    break
                fix_prompt = (
                    "Fix EVERY listed VideoPlanV3 validation issue strictly:\n- "
                    + "\n- ".join(issues)
                    + "\nRULES WHEN FIXING:\n"
                    "- Never attach polygon claims (`nonzero_area`, `square`, `right_triangle`, `area_conservation`) to non-polygons like `particle_system` or `axes`; remove or switch claim kind (e.g. `ode_equilibrium`, `eigenvalue_determinant`, `eigenvalue_stability_regime`).\n"
                    "- If `point_on_function` reports a computed `f(x)=...` value, either update `coords[1]` to that exact value or replace with `ode_equilibrium` / remove misclassified phase point claim.\n"
                    "- If `parametric_curve requires 'X(t); Y(t)'`, set `expr='X(t); Y(t)'` explicitly (or remove the duplicate `parametric_curve` when an `ode_phase_trajectory` ConstructionOp already creates the trajectory `data_series`).\n"
                    "- Do NOT list an already-declared `axes` ID inside `constructions[].output_ids`; put it in `params.axes_id` instead.\n"
                    "- For `markov_chain_evolution`, `transition_matrix` MUST be ROW-stochastic (each row sums strictly to 1.0; transpose if column-stochastic).\n"
                    f"\nCurrent JSON:\n{ch_plan.model_dump_json()}"
                )
                ch_plan, _ = director.provider.generate_structured(fix_prompt, VideoPlanV3, system=ch_system, temperature=0.3)
                with log_lock:
                    gen_log.append({"stage": f"chapter_{ch.chapter_id}_repair_{r_idx}", "issues": issues})
            issues = validate_v3_plan(ch_plan, require_claims=True)
            if not issues:
                break

        if issues or ch_plan is None:
            _atomic_write_json(logs_dir / f"failed_{ch.chapter_id}.json", ch_plan.model_dump() if ch_plan else {})
            raise RuntimeError(f"Chapter {ch.chapter_id} failed validation after repairs: {issues}")
        mat = materialize_constructions(ch_plan)
        for s_item in mat.scenes:
            s_item.narration = clean_spoken_text(s_item.narration)
        ch_cache_file.write_text(mat.model_dump_json(indent=2), encoding="utf-8")
        return idx, mat

    llm_workers = max(1, min(len(lf_plan.chapters), int(os.environ.get("STUDIO_LLM_WORKERS", "3"))))
    with ThreadPoolExecutor(max_workers=llm_workers) as pool:
        futs = [pool.submit(_script_chapter, idx, ch) for idx, ch in enumerate(lf_plan.chapters, 1)]
        for fut in as_completed(futs):
            c_idx, mat = fut.result()
            ordered_plans[c_idx - 1] = mat

    chapter_plans: list[VideoPlanV3] = [cp for cp in ordered_plans if cp is not None]
    last_ch = lf_plan.chapters[-1] if lf_plan.chapters else None
    prev_summary = f"{last_ch.title}: {last_ch.summary}" if last_ch else ""

    # Calibrated duration check using true Kokoro effective WPS (~2.08 wps including beat pauses)
    est_s = sum(len(parse_script(s.narration)[0].split()) for cp in chapter_plans for s in cp.scenes) / KOKORO_EFFECTIVE_WPS + len(chapter_plans) * 1.2
    extra_idx = 1
    while est_s < target_duration_s * 0.90 and extra_idx <= 4:
        ext_id = f"ext_{extra_idx}"
        ext_title = f"Deep Dive {extra_idx}: Case Studies & Edge Regimes in {topic}"
        ext_target_s = min(90.0, max(45.0, target_duration_s - est_s))
        ext_min_words = max(90, int(ext_target_s * 1.85))
        ext_max_words = max(ext_min_words + 25, int(ext_target_s * 2.25))
        if on_progress:
            on_progress(f"extending duration ({est_s:.0f}s < 90% of {target_duration_s:.0f}s): {ext_title}")
        ext_prompt = (
            f"{ch_system}\n\n"
            f"Overall Topic: {topic}\nContinuation Chapter: {ext_title}\n"
            f"Previous Chapter Context: {prev_summary}\n"
            f"Shared Continuity Variables: {json.dumps(lf_plan.continuity.defined_variables)}\n"
            f"Target duration: {ext_target_s:.0f}s ({ext_min_words}-{ext_max_words} spoken words across 5-8 scenes).\n"
            "Write a complete VideoPlanV3 deepening intuition through contrasting concrete regimes, deterministic constructions, and verifiable claims."
        )
        ch_plan, ch_resp = director.provider.generate_structured(ext_prompt, VideoPlanV3, system=ch_system, temperature=0.45)
        for r_idx in range(1, MAX_REPAIRS + 1):
            issues = validate_v3_plan(ch_plan, require_claims=True)
            w_cnt = sum(len(parse_script(s.narration)[0].split()) for s in ch_plan.scenes)
            if w_cnt < ext_min_words:
                issues.append(f"Narration too brief ({w_cnt} words < {ext_min_words}); add step-by-step analysis.")
            elif w_cnt > ext_max_words:
                issues.append(f"Narration too long ({w_cnt} words > {ext_max_words}); tighten phrasing.")
            if not issues:
                break
            fix_prompt = "Fix these VideoPlanV3 issues:\n- " + "\n- ".join(issues) + f"\n\nCurrent JSON:\n{ch_plan.model_dump_json()}"
            ch_plan, _ = director.provider.generate_structured(fix_prompt, VideoPlanV3, system=ch_system, temperature=0.3)
        if not validate_v3_plan(ch_plan, require_claims=True):
            lf_plan.chapters.append(ChapterSpec(chapter_id=ext_id, title=ext_title, node_ids=[], target_duration_s=ext_target_s))
            mat = materialize_constructions(ch_plan)
            chapter_plans.append(mat)
            est_s += sum(len(parse_script(s.narration)[0].split()) for s in mat.scenes) / KOKORO_EFFECTIVE_WPS + 1.2
            prev_summary = ext_title
        extra_idx += 1

    # Export canonical unified ScriptIR & SCRIPT.md capturing every locked beat ID & chapter link
    all_beats: list[NarrationBeatIR] = []
    for ch_obj, cp in zip(lf_plan.chapters, chapter_plans):
        for s_item in cp.scenes:
            clean_sp = parse_script(s_item.narration)[0]
            all_beats.append(
                NarrationBeatIR(
                    beat_id=f"{ch_obj.chapter_id}__{s_item.scene_id}",
                    scene_id=s_item.scene_id,
                    speaker=s_item.speaker if s_item.speaker in ("narrator", "educator", "learner", "host_a", "host_b") else "narrator",  # type: ignore[arg-type]
                    spoken_text=s_item.narration,
                    educational_purpose=s_item.purpose,
                    mathematical_claims=[c.statement or c.kind for c in s_item.claims],
                    visual_intent=", ".join(f"{o.id}:{o.kind}" for o in s_item.objects[:5]),
                    estimated_duration_s=round(max(2.5, len(clean_sp.split()) / KOKORO_EFFECTIVE_WPS + 0.35), 2),
                )
            )
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
    (lf_dir / "SCRIPT.md").write_text(export_script_md(doc_script), encoding="utf-8")

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

            math_key = ArtifactCache.compute_stage_key("math_model", plan.model_dump(), upstream_keys={"knowledge": know_key})
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
            ) + f"_{backend}"
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
            slots_srt[idx] = srt_p
            slots_txt[idx] = txt_p
            slots_bdir[idx] = ch_build

            # Enforce strict SHA-256 + upstream dependency verification (never bypass via raw JobStore lookup)
            cached = stage_path if stage_hit else None
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

    loud = verify_audio_loudness(final_mp4)
    streams = verify_media_streams(final_mp4)
    target_sum = sum(float(c.target_duration_s or 0.0) for c in longform_plan.chapters) or None
    rev_report = audit_educational_quality(
        topic=longform_plan.topic,
        title=longform_plan.topic,
        plan=merged_plan,
        format_name=format_name,
        backend=backend,
        video_path=final_mp4,
        contact_sheet_out=lf_dir / f"QA_CONTACT_SHEET_{resolved_look}.png",
        look_name=resolved_look,
        chapter_plans=list(chapter_plans),
        target_duration_s=target_sum,
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
    # Choose core mechanism chapter cleanly (prefer highest claims + constructions + animated events density)
    if chapter_index is None:
        _anim_ops = {"transform", "morph", "move", "rotate", "scale", "trace", "animate_param", "iterate_states", "interpolate_values"}
        scores = [
            sum(
                len(s.claims) * 3
                + len(s.constructions) * 2
                + sum(3 for e in s.events if e.op in _anim_ops)
                for s in cp.scenes
            )
            for cp in chapter_plans
        ]
        chapter_index = int(max(range(len(chapter_plans)), key=lambda i: scores[i]))

    src = chapter_plans[chapter_index].model_copy(deep=True)
    # Trim/focus cleanly to <= 125 spoken words (~45-60s standalone Short) keeping object/axes closure
    kept_scenes = []
    declared_ids: set[str] = set()
    word_budget = 0
    for sc in src.scenes[:6]:
        words_sc = len(sc.narration.split())
        if kept_scenes and (word_budget + words_sc > 135 or len(kept_scenes) >= 4):
            break
        sc_copy = sc.model_copy(deep=True)
        sc_copy.composition = adapt_composition_for_aspect(sc_copy.composition, "9:16")  # type: ignore[assignment]
        for o in sc_copy.objects:
            if o.panel_slot == "left":
                o.panel_slot = "top"
            elif o.panel_slot == "right":
                o.panel_slot = "bottom"
            declared_ids.add(o.id)
        sc_copy.events = [e for e in sc_copy.events if e.target_id in declared_ids and (not e.source_id or e.source_id in declared_ids)]
        kept_scenes.append(sc_copy)
        word_budget += words_sc

    short_plan = VideoPlanV3(
        schema_version="3.0",
        topic=f"{longform_plan.topic} — Key Insight",
        title=kept_scenes[0].purpose if kept_scenes else longform_plan.topic,
        audience=src.audience,
        strategy=src.strategy,
        target_duration_s=55.0,
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
    rev = audit_educational_quality(
        topic=short_plan.topic,
        title=short_plan.title,
        plan=short_plan,
        format_name=format_name,
        backend=backend,
        video_path=mp4,
        contact_sheet_out=short_dir / f"QA_CONTACT_SHEET_{resolved_look}.png",
        look_name=resolved_look,
        target_duration_s=55.0,
    )
    export_review_md(rev, short_dir / "REVIEW.md")  # type: ignore[arg-type]
    return {
        "output_mp4": str(mp4),
        "source_chapter": longform_plan.chapters[chapter_index].chapter_id,
        "duration_s": _probe_duration(mp4),
        "storyboard_md": str(short_dir / "STORYBOARD.md"),
        "review_md": str(short_dir / "REVIEW.md"),
        "qa_loudness": {"passed": loud.passed, "detail": loud.detail},
        "qa_streams": {"passed": streams.passed, "detail": streams.detail},
    }
