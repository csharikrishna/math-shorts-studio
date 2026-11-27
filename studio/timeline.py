"""Unified master timeline and subtitle/transcript exporters (SRT, WebVTT, JSON, TXT)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from studio.audio.pronounce import normalize_for_speech
from studio.audio.tts import AudioTimeline, SyncedBeat


def _fmt_srt_ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_ts(seconds: float) -> str:
    return _fmt_srt_ts(seconds).replace(",", ".")


@dataclass
class TimelineTrackEvent:
    track: str
    start_s: float
    end_s: float
    label: str
    meta: dict = field(default_factory=dict)


@dataclass
class CaptionSegment:
    """Semantic phrase-level caption segment shared across burned-in, SRT, and WebVTT."""

    beat_id: str
    start_s: float
    end_s: float
    text: str
    lines: list[str]
    words: list[list] = field(default_factory=list)


_CLAUSE_BREAKS = (".", ",", "!", "?", ";", ":", "—")
_SOFT_BREAK_WORDS = {"and", "or", "but", "so", "because", "while", "where", "when", "then", "which", "that", "plus", "equals"}


def _balance_two_lines(words: list[str], max_chars: int) -> list[str]:
    joined = " ".join(words)
    if len(joined) <= max_chars or len(words) <= 1:
        return [joined]
    best_split = max(1, len(words) // 2)
    best_diff = 10**9
    for i in range(1, len(words)):
        l1 = " ".join(words[:i])
        l2 = " ".join(words[i:])
        if len(l1) <= max_chars and len(l2) <= max_chars:
            diff = abs(len(l1) - len(l2))
            if diff < best_diff:
                best_diff = diff
                best_split = i
    return [" ".join(words[:best_split]), " ".join(words[best_split:])]


def build_semantic_captions(
    beats: list[SyncedBeat],
    max_chars_per_line: int = 44,
    max_lines: int = 2,
    font_px: int = 44,
    max_width_px: float = 1480.0,
) -> list[CaptionSegment]:
    """Group timed words into grammatical 1-2 line phrases via unified CaptionIR chunking."""
    from studio.captions.layout import chunk_caption_cues

    cues = chunk_caption_cues(
        beats,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        font_px=font_px,
        max_width_px=max_width_px,
    )
    return [
        CaptionSegment(
            beat_id=c.beat_id,
            start_s=c.start_s,
            end_s=c.end_s,
            text=c.text,
            lines=list(c.lines),
            words=[[w.text, w.start_s, w.end_s, w.emphasis] for w in c.words],
        )
        for c in cues
    ]


@dataclass
class MasterTimeline:
    """Authoritative synchronization timeline across narration, captions, visuals, and audio."""

    audio: AudioTimeline
    events: list[TimelineTrackEvent] = field(default_factory=list)

    def validate(self) -> list[str]:
        issues: list[str] = []
        prev_end = 0.0
        for b in self.audio.beats:
            if b.start_time < prev_end - 1e-3:
                issues.append(f"Beat {b.id} overlaps previous beat ({b.start_time:.3f} < {prev_end:.3f}).")
            if b.end_time <= b.start_time:
                issues.append(f"Beat {b.id} has non-positive duration.")
            prev_end = b.end_time
        return issues

    def export_subtitles(
        self,
        out_dir: Path,
        stem: str,
        *,
        caption_style: str = "editorial",
        format_name: str = "widescreen_hd",
        look_name: str = "midnight_studio",
    ) -> dict[str, Path]:
        from studio.captions import build_caption_ir, export_ass
        from studio.config import FORMATS

        out_dir.mkdir(parents=True, exist_ok=True)
        prof = FORMATS.get(format_name, FORMATS["widescreen_hd"])
        srt_path = out_dir / f"{stem}.srt"
        vtt_path = out_dir / f"{stem}.vtt"
        ass_path = out_dir / f"{stem}.ass"
        txt_path = out_dir / f"{stem}_transcript.txt"
        words_path = out_dir / f"{stem}_words.json"
        captions_path = out_dir / f"{stem}_captions.json"

        ir = build_caption_ir(
            self.audio.beats,
            style=caption_style,
            width_px=prof.pixel_width,
            height_px=prof.pixel_height,
        )
        segments = [
            CaptionSegment(
                beat_id=c.beat_id,
                start_s=c.start_s,
                end_s=c.end_s,
                text=c.text,
                lines=list(c.lines),
                words=[[w.text, w.start_s, w.end_s, w.emphasis] for w in c.words],
            )
            for c in ir.cues
        ]
        srt_lines: list[str] = []
        vtt_lines: list[str] = ["WEBVTT", ""]

        for idx, seg in enumerate(segments, 1):
            t0 = _fmt_srt_ts(seg.start_s)
            t1 = _fmt_srt_ts(seg.end_s)
            srt_lines.extend([str(idx), f"{t0} --> {t1}", seg.text, ""])
            vtt_lines.extend([f"{_fmt_vtt_ts(seg.start_s)} --> {_fmt_vtt_ts(seg.end_s)}", seg.text, ""])

        export_ass(ir, ass_path, look_name=look_name)
        transcript_lines = [b.narration for b in self.audio.beats]
        all_words = [
            {"beat_id": b.id, "word": w[0], "start": w[1], "end": w[2], "emphasis": bool(w[3]) if len(w) > 3 else False}
            for b in self.audio.beats
            for w in b.words
        ]
        caption_meta = {
            "style": ir.style,
            "aspect": ir.aspect_label,
            "reserve_band_px": ir.reserve_band_px,
            "segments": [asdict(s) for s in segments],
            "beats": [
                {
                    "beat_id": b.id,
                    "speaker": getattr(b, "speaker", "narrator"),
                    "original_narration": b.narration,
                    "spoken_text": normalize_for_speech(b.narration),
                    "bookmarks": b.marks,
                }
                for b in self.audio.beats
            ],
        }

        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")
        txt_path.write_text("\n\n".join(transcript_lines), encoding="utf-8")
        words_path.write_text(json.dumps(all_words, indent=2), encoding="utf-8")
        captions_path.write_text(json.dumps(caption_meta, indent=2), encoding="utf-8")
        return {
            "srt": srt_path,
            "vtt": vtt_path,
            "ass": ass_path,
            "transcript": txt_path,
            "words": words_path,
            "captions": captions_path,
        }
