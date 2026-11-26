"""Unified master timeline and subtitle/transcript exporters (SRT, WebVTT, JSON, TXT)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from studio.audio.tts import AudioTimeline


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

        srt_lines: list[str] = []
        vtt_lines: list[str] = ["WEBVTT", ""]
        transcript_lines: list[str] = []
        all_words: list[dict] = []

        for idx, b in enumerate(self.audio.beats, 1):
            t0 = _fmt_srt_ts(b.start_time)
            t1 = _fmt_srt_ts(b.start_time + b.speech_duration)
            text = b.narration or b.caption
            srt_lines.extend([str(idx), f"{t0} --> {t1}", text, ""])
            vtt_lines.extend([f"{_fmt_vtt_ts(b.start_time)} --> {_fmt_vtt_ts(b.start_time + b.speech_duration)}", text, ""])
            transcript_lines.append(b.narration)
            for w in b.words:
                all_words.append({"beat_id": b.id, "word": w[0], "start": w[1], "end": w[2], "emphasis": bool(w[3]) if len(w) > 3 else False})

        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")
        txt_path.write_text("\n\n".join(transcript_lines), encoding="utf-8")
        words_path.write_text(json.dumps(all_words, indent=2), encoding="utf-8")
        return {"srt": srt_path, "vtt": vtt_path, "transcript": txt_path, "words": words_path}
