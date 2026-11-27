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
    max_chars_per_line: int = 34,
    max_lines: int = 2,
) -> list[CaptionSegment]:
    """Group timed words into grammatical 1-2 line phrases for burned-in, SRT, and WebVTT."""
    segments: list[CaptionSegment] = []
    max_chunk_chars = max_chars_per_line * max_lines

    for b in beats:
        if not b.words or (b.narration and len(b.words) < len(b.narration.split())):
            txt = (b.narration or b.caption).strip()
            if txt:
                lines = _balance_two_lines(txt.split(), max_chars_per_line)
                segments.append(CaptionSegment(b.id, b.start_time, b.start_time + b.speech_duration, "\n".join(lines), lines, []))
            continue

        cur: list[list] = []
        for idx, w in enumerate(b.words):
            tok = str(w[0])
            cand = [*cur, w]
            cand_len = len(" ".join(str(x[0]) for x in cand))
            next_tok = str(b.words[idx + 1][0]).lower().strip(".,!?;:") if idx + 1 < len(b.words) else ""

            if cur and cand_len > max_chunk_chars:
                lines = _balance_two_lines([str(x[0]) for x in cur], max_chars_per_line)
                segments.append(CaptionSegment(b.id, float(cur[0][1]), float(cur[-1][2]), "\n".join(lines), lines, cur))
                cur = [w]
            else:
                cur = cand

            hard_break = tok.endswith(_CLAUSE_BREAKS) and len(cur) >= 2
            soft_break = next_tok in _SOFT_BREAK_WORDS and len(" ".join(str(x[0]) for x in cur)) >= int(max_chars_per_line * 0.75) and len(cur) >= 3
            if hard_break or soft_break:
                lines = _balance_two_lines([str(x[0]) for x in cur], max_chars_per_line)
                segments.append(CaptionSegment(b.id, float(cur[0][1]), float(cur[-1][2]), "\n".join(lines), lines, cur))
                cur = []

        if cur:
            lines = _balance_two_lines([str(x[0]) for x in cur], max_chars_per_line)
            segments.append(CaptionSegment(b.id, float(cur[0][1]), float(cur[-1][2]), "\n".join(lines), lines, cur))

    return segments


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

    def export_subtitles(self, out_dir: Path, stem: str) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        srt_path = out_dir / f"{stem}.srt"
        vtt_path = out_dir / f"{stem}.vtt"
        txt_path = out_dir / f"{stem}_transcript.txt"
        words_path = out_dir / f"{stem}_words.json"
        captions_path = out_dir / f"{stem}_captions.json"

        segments = build_semantic_captions(self.audio.beats)
        srt_lines: list[str] = []
        vtt_lines: list[str] = ["WEBVTT", ""]

        for idx, seg in enumerate(segments, 1):
            t0 = _fmt_srt_ts(seg.start_s)
            t1 = _fmt_srt_ts(seg.end_s)
            srt_lines.extend([str(idx), f"{t0} --> {t1}", seg.text, ""])
            vtt_lines.extend([f"{_fmt_vtt_ts(seg.start_s)} --> {_fmt_vtt_ts(seg.end_s)}", seg.text, ""])

        transcript_lines = [b.narration for b in self.audio.beats]
        all_words = [
            {"beat_id": b.id, "word": w[0], "start": w[1], "end": w[2], "emphasis": bool(w[3]) if len(w) > 3 else False}
            for b in self.audio.beats
            for w in b.words
        ]
        caption_meta = {
            "segments": [asdict(s) for s in segments],
            "beats": [
                {
                    "beat_id": b.id,
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
        return {"srt": srt_path, "vtt": vtt_path, "transcript": txt_path, "words": words_path, "captions": captions_path}
