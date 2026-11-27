"""Long-form chapter-based educational video production pipeline (5+ min widescreen_hd)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable

from studio.ai.knowledge import ChapterSpec, ContinuityState, KnowledgeTree, LongFormPlan
from studio.ai.v3_schema import VideoPlanV3
from studio.ai.validate_v3 import validate_v3_plan
from studio.audio.tts import find_ffmpeg
from studio.pipeline import PROJECT_ROOT, run_plan
from studio.qa.engine import verify_audio_loudness, verify_media_streams
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
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Render chapters independently with checkpoint recovery and assemble a master long-form MP4."""
    if len(longform_plan.chapters) != len(chapter_plans):
        raise ValueError("chapters count must match chapter_plans count.")

    lf_dir = PROJECT_ROOT / "build" / f"longform_{slug}"
    lf_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = lf_dir / "checkpoint.json"
    checkpoint: dict[str, str] = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}

    # Save KnowledgeTree + LongFormPlan + ContinuityState
    (lf_dir / "longform_plan.json").write_text(longform_plan.model_dump_json(indent=2), encoding="utf-8")

    chapter_mp4s: list[Path] = []
    chapter_srts: list[Path] = []
    chapter_transcripts: list[Path] = []

    for ch, plan in zip(longform_plan.chapters, chapter_plans):
        issues = validate_v3_plan(plan)
        if issues:
            raise ValueError(f"Chapter {ch.chapter_id} invalid:\n- " + "\n- ".join(issues))

        ch_plan_path = lf_dir / f"{slug}_{ch.chapter_id}.json"
        ch_plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        cached_mp4 = checkpoint.get(ch.chapter_id)
        if cached_mp4 and Path(cached_mp4).exists():
            mp4 = Path(cached_mp4)
            if on_progress:
                on_progress(f"checkpoint hit: {ch.chapter_id} -> {mp4.name}")
        else:
            if on_progress:
                on_progress(f"rendering chapter: {ch.chapter_id} ({ch.title})")
            mp4 = run_plan(
                ch_plan_path,
                format_name=format_name,
                tts_engine=tts_engine,
                voice=voice,
                rate=rate,
                ui=ui,
                look=look,
            )
            checkpoint[ch.chapter_id] = str(mp4)
            ckpt_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        chapter_mp4s.append(mp4)
        ch_stem = f"V3_{ch_plan_path.stem}"
        ch_build = PROJECT_ROOT / "build" / f"{ch_stem}_{format_name}"
        chapter_srts.append(ch_build / f"{ch_stem}.srt")
        chapter_transcripts.append(ch_build / f"{ch_stem}_transcript.txt")

    # Assemble chapter timestamps, semantic SRT/VTT, and transcript
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

        if srt_p.exists():
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

        if txt_p.exists():
            merged_transcript.append(f"## {ch.title}\n\n{txt_p.read_text(encoding='utf-8').strip()}")
        cursor_s += dur

    concat_list = lf_dir / "concat_list.txt"
    concat_list.write_text("\n".join(f"file '{p.as_posix()}'" for p in chapter_mp4s), encoding="utf-8")

    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = out_dir / f"LF_{slug}_{format_name}.mp4"
    ffmpeg_bin = find_ffmpeg()
    subprocess.run(
        [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(final_mp4)],
        check=True,
        capture_output=True,
    )

    chapters_json = lf_dir / f"{slug}_chapters.json"
    chapters_txt = lf_dir / f"{slug}_chapters.txt"
    srt_out = lf_dir / f"{slug}.srt"
    vtt_out = lf_dir / f"{slug}.vtt"
    txt_out = lf_dir / f"{slug}_transcript.txt"

    chapters_json.write_text(json.dumps(chapters_meta, indent=2), encoding="utf-8")
    chapters_txt.write_text("\n".join(yt_lines), encoding="utf-8")
    srt_out.write_text("\n".join(merged_srt), encoding="utf-8")
    vtt_out.write_text("\n".join(merged_vtt), encoding="utf-8")
    txt_out.write_text("\n\n".join(merged_transcript), encoding="utf-8")

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
    }
