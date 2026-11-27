"""Autonomous long-form chapter-based educational video production pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from studio.ai.director import GeminiDirector, MAX_REPAIRS, V3_STYLE_GUIDE, slugify
from studio.ai.knowledge import AudienceLevel, ChapterSpec, ContinuityState, KnowledgeTree, LongFormPlan
from studio.ai.v3_schema import VideoPlanV3
from studio.ai.validate_v3 import validate_v3_plan
from studio.ai.world_model import materialize_constructions
from studio.audio.tts import find_ffmpeg
from studio.pipeline import PROJECT_ROOT, run_plan
from studio.qa.engine import verify_audio_loudness, verify_longform_bundle, verify_media_streams
from studio.storage.cache import JobStore
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


def _source_code_digest() -> str:
    h = hashlib.sha256()
    for rel in ("scenes/v3_scene.py", "studio/ai/world_model.py", "studio/layout/components.py"):
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
) -> str:
    """Compute content-addressed cache key covering plan, math model, audio, render config, and code."""
    return JobStore.content_hash(
        {
            "plan": plan.model_dump(),
            "format_name": format_name,
            "tts_engine": tts_engine,
            "voice": voice,
            "rate": rate,
            "ui": ui,
            "look": look,
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


class _LongFormOutline(BaseModel):
    topic: str
    audience: AudienceLevel = "general"
    knowledge_tree: KnowledgeTree
    chapters: list[ChapterSpec] = Field(default_factory=list)
    defined_variables: list[_VarDef] = Field(default_factory=list)
    introduced_terms: list[str] = Field(default_factory=list)
    color_map: list[_ColorDef] = Field(default_factory=list)


def create_longform_plan(
    topic: str,
    *,
    target_duration_s: float = 300.0,
    audience: AudienceLevel = "general",
    model: str | None = None,
    provider_name: str = "gemini",
    free_only: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[LongFormPlan, list[VideoPlanV3]]:
    """Autonomously generate KnowledgeTree -> LongFormPlan -> validated VideoPlanV3 chapters via Gemini."""
    slug = slugify(topic)
    lf_dir = PROJECT_ROOT / "build" / f"longform_{slug}"
    logs_dir = lf_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    gen_log: list[dict[str, object]] = []

    director = GeminiDirector(model=model, provider_name=provider_name, free_only=free_only)
    n_chapters = max(3, min(24, round(target_duration_s / 20.0)))
    per_ch_s = target_duration_s / n_chapters

    if on_progress:
        on_progress(f"planning knowledge tree & {n_chapters} chapters (~{target_duration_s:.0f}s total)...")

    outline_prompt = (
        f"Create a structured LongFormPlan for an educational math documentary on: {topic}\n"
        f"Audience: {audience}. Target total duration: {target_duration_s:.0f}s across EXACTLY {n_chapters} chapters (~{per_ch_s:.0f}s each).\n"
        "Include:\n"
        "- knowledge_tree with prerequisite-ordered nodes and verified claims\n"
        "- chapters covering the full progression from intuition -> mechanics -> proof/examples -> synthesis\n"
        "- continuity with shared defined_variables, introduced_terms, and color_map."
    )
    outline, resp = director.provider.generate_structured(outline_prompt, _LongFormOutline, temperature=0.4)
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
    gen_log.append({"stage": "outline", "model": resp.model, "latency_s": resp.latency_s, "tokens": [resp.prompt_tokens, resp.completion_tokens]})
    (logs_dir / "longform_plan_raw.json").write_text(lf_plan.model_dump_json(indent=2), encoding="utf-8")

    chapter_plans: list[VideoPlanV3] = []
    node_map = {n.node_id: n for n in lf_plan.knowledge_tree.nodes}
    for idx, ch in enumerate(lf_plan.chapters, 1):
        if on_progress:
            on_progress(f"scripting chapter {idx}/{len(lf_plan.chapters)}: {ch.title}")
        node_ctx = "; ".join(f"{node_map[nid].title}: {[c.statement for c in node_map[nid].claims]}" for nid in ch.node_ids if nid in node_map)
        ch_prompt = (
            f"{V3_STYLE_GUIDE}\n\n"
            f"Overall Topic: {topic}\n"
            f"Chapter {idx}/{len(lf_plan.chapters)}: {ch.title}\n"
            f"Summary: {ch.summary}\n"
            f"Knowledge context: {node_ctx}\n"
            f"Shared Continuity Variables: {json.dumps(lf_plan.continuity.defined_variables)}\n"
            f"Target duration: {max(25.0, ch.target_duration_s):.0f}s (~{int(max(25.0, ch.target_duration_s) * 2.4)} spoken words).\n"
            "Write a complete, self-contained VideoPlanV3 for this chapter with deterministic constructions and verifiable claims."
        )
        ch_plan = None
        issues: list[str] = []
        for _attempt in range(2):
            ch_plan, ch_resp = director.provider.generate_structured(ch_prompt, VideoPlanV3, system=V3_STYLE_GUIDE, temperature=0.45)
            gen_log.append({"stage": f"chapter_{ch.chapter_id}", "model": ch_resp.model, "latency_s": ch_resp.latency_s})
            for r_idx in range(1, MAX_REPAIRS + 1):
                issues = validate_v3_plan(ch_plan, require_claims=True)
                if not issues:
                    break
                fix_prompt = "Fix these VideoPlanV3 issues:\n- " + "\n- ".join(issues) + f"\n\nCurrent JSON:\n{ch_plan.model_dump_json()}"
                ch_plan, _ = director.provider.generate_structured(fix_prompt, VideoPlanV3, system=V3_STYLE_GUIDE, temperature=0.3)
                gen_log.append({"stage": f"chapter_{ch.chapter_id}_repair_{r_idx}", "issues": issues})
            issues = validate_v3_plan(ch_plan, require_claims=True)
            if not issues:
                break

        if issues or ch_plan is None:
            raise RuntimeError(f"Chapter {ch.chapter_id} failed validation after repairs: {issues}")
        chapter_plans.append(materialize_constructions(ch_plan))

    # Duration-aware extension: if total spoken duration is short of target_duration_s, add continuation chapters
    from studio.audio.tts import parse_script

    est_s = sum(len(parse_script(s.narration)[0].split()) for cp in chapter_plans for s in cp.scenes) / 2.4 + len(chapter_plans) * 1.2
    extra_idx = 1
    while est_s < target_duration_s * 0.98 and extra_idx <= 6:
        ext_id = f"ext_{extra_idx}"
        ext_title = f"Deep Dive {extra_idx}: Applying {topic}"
        if on_progress:
            on_progress(f"extending duration ({est_s:.0f}s < {target_duration_s:.0f}s): {ext_title}")
        ext_prompt = (
            f"{V3_STYLE_GUIDE}\n\n"
            f"Overall Topic: {topic}\nContinuation Chapter: {ext_title}\n"
            f"Shared Continuity Variables: {json.dumps(lf_plan.continuity.defined_variables)}\n"
            f"Target duration: 30s (~75 spoken words across 4 scenes).\n"
            "Write a complete VideoPlanV3 deepening the intuition with concrete examples, deterministic constructions, and verifiable claims."
        )
        ch_plan, ch_resp = director.provider.generate_structured(ext_prompt, VideoPlanV3, system=V3_STYLE_GUIDE, temperature=0.45)
        for r_idx in range(1, MAX_REPAIRS + 1):
            issues = validate_v3_plan(ch_plan, require_claims=True)
            if not issues:
                break
            fix_prompt = "Fix these VideoPlanV3 issues:\n- " + "\n- ".join(issues) + f"\n\nCurrent JSON:\n{ch_plan.model_dump_json()}"
            ch_plan, _ = director.provider.generate_structured(fix_prompt, VideoPlanV3, system=V3_STYLE_GUIDE, temperature=0.3)
        if not validate_v3_plan(ch_plan, require_claims=True):
            lf_plan.chapters.append(ChapterSpec(chapter_id=ext_id, title=ext_title, node_ids=[], target_duration_s=25.0))
            mat = materialize_constructions(ch_plan)
            chapter_plans.append(mat)
            est_s += sum(len(parse_script(s.narration)[0].split()) for s in mat.scenes) / 2.4 + 1.2
        extra_idx += 1

    _atomic_write_json(logs_dir / "generation_log.json", gen_log)
    return lf_plan, chapter_plans


def build_longform_video(
    slug: str,
    longform_plan: LongFormPlan,
    chapter_plans: list[VideoPlanV3],
    *,
    format_name: str = "widescreen_hd",
    tts_engine: str = "kokoro",
    voice: str = "en-US-AndrewNeural",
    rate: str = "-4%",
    ui: str = "card",
    look: str = "cinematic",
    min_duration_s: float = 0.0,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Render chapters independently with content-addressed caching and assemble a verified long-form MP4."""
    if len(longform_plan.chapters) != len(chapter_plans):
        raise ValueError("chapters count must match chapter_plans count.")

    lf_dir = PROJECT_ROOT / "build" / f"longform_{slug}"
    lf_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = lf_dir / "checkpoint.json"
    checkpoint: dict[str, dict[str, str]] = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}

    (lf_dir / "longform_plan.json").write_text(longform_plan.model_dump_json(indent=2), encoding="utf-8")
    jobs = JobStore(PROJECT_ROOT / "build" / "studio_jobs.db")

    chapter_mp4s: list[Path] = []
    chapter_srts: list[Path] = []
    chapter_transcripts: list[Path] = []

    try:
        for ch, plan in zip(longform_plan.chapters, chapter_plans):
            issues = validate_v3_plan(plan)
            if issues:
                raise ValueError(f"Chapter {ch.chapter_id} invalid:\n- " + "\n- ".join(issues))

            ch_plan_path = lf_dir / f"{slug}_{ch.chapter_id}.json"
            ch_plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

            ckey = chapter_cache_key(
                plan,
                format_name=format_name,
                tts_engine=tts_engine,
                voice=voice,
                rate=rate,
                ui=ui,
                look=look,
            )
            cached = jobs.get_cached(ckey)
            if cached and verify_media_streams(cached).passed:
                mp4 = cached
                if on_progress:
                    on_progress(f"content-cache hit [{ckey[:8]}]: {ch.chapter_id} -> {mp4.name}")
            else:
                if on_progress:
                    on_progress(f"rendering chapter [{ckey[:8]}]: {ch.chapter_id} ({ch.title})")
                mp4 = run_plan(
                    ch_plan_path,
                    format_name=format_name,
                    tts_engine=tts_engine,
                    voice=voice,
                    rate=rate,
                    ui=ui,
                    look=look,
                )
                jobs.put_cached(ckey, mp4)

            checkpoint[ch.chapter_id] = {"cache_key": ckey, "mp4": str(mp4)}
            _atomic_write_json(ckpt_path, checkpoint)

            chapter_mp4s.append(mp4)
            ch_stem = f"V3_{ch_plan_path.stem}"
            ch_build = PROJECT_ROOT / "build" / f"{ch_stem}_{format_name}"
            chapter_srts.append(ch_build / f"{ch_stem}.srt")
            chapter_transcripts.append(ch_build / f"{ch_stem}_transcript.txt")
    finally:
        jobs.close()

    chapters_meta: list[dict[str, object]] = []
    yt_lines: list[str] = []
    merged_srt: list[str] = []
    merged_vtt: list[str] = ["WEBVTT", ""]
    merged_transcript: list[str] = []
    srt_idx = 1
    cursor_s = 0.0

    for ch, mp4, srt_p, txt_p in zip(longform_plan.chapters, chapter_mp4s, chapter_srts, chapter_transcripts):
        dur = _probe_duration(mp4)
        chapters_meta.append(
            {"chapter_id": ch.chapter_id, "title": ch.title, "start_s": round(cursor_s, 3), "end_s": round(cursor_s + dur, 3), "duration_s": round(dur, 3)}
        )
        yt_lines.append(f"{_fmt_yt_ts(cursor_s)} {ch.title}")

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

    bundle_chk = verify_longform_bundle(
        final_mp4, chapters_json, srt_out, vtt_out, txt_out, min_duration_s=min_duration_s
    )
    if not bundle_chk.passed:
        raise RuntimeError(f"Long-form final bundle QA failed: {bundle_chk.detail}")

    loud = verify_audio_loudness(final_mp4)
    streams = verify_media_streams(final_mp4)
    return {
        "output_mp4": str(final_mp4),
        "total_duration_s": round(cursor_s, 3),
        "chapters": chapters_meta,
        "chapters_txt": str(chapters_txt),
        "srt": str(srt_out),
        "vtt": str(vtt_out),
        "transcript": str(txt_out),
        "qa_loudness": {"passed": loud.passed, "detail": loud.detail},
        "qa_streams": {"passed": streams.passed, "detail": streams.detail},
        "qa_bundle": {"passed": bundle_chk.passed, "detail": bundle_chk.detail},
    }
