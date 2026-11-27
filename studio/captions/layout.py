"""Grammatical & width-aware caption phrase chunking engine."""

from __future__ import annotations

import re
from typing import Iterable

from studio.audio.tts import SyncedBeat
from studio.captions.model import (
    CAPTION_PRESETS,
    CaptionCue,
    CaptionIR,
    CaptionStyle,
    CaptionWord,
    normalize_caption_style,
)
from studio.design.fonts import effective_px_scale, estimate_text_width_px

_CLAUSE_BREAKS = (".", ",", "!", "?", ";", ":", "—", ")")
_ORPHAN_TAILS = {
    "a", "an", "and", "or", "but", "nor", "so", "yet", "the", "of", "in", "on", "at",
    "to", "for", "from", "by", "with", "as", "if", "than", "that", "which", "when", "while", "where",
}
_SOFT_BREAK_BEFORE = {
    "and", "or", "but", "so", "because", "while", "where", "when", "then", "which", "that", "unless", "whereas",
}
_MATH_TOKEN_RE = re.compile(r"([=<>+\-*/^_]|lambda|alpha|beta|det|tr|re|im|\d)")


def _is_math_expr_boundary(tok_a: str, tok_b: str) -> bool:
    """Avoid splitting directly across short compact algebraic tokens like 'x = 0' or 'Re(λ) > 0'."""
    if not tok_a or not tok_b:
        return False
    if tok_b in ("=", "+", "-", "*", "/", "<", ">", "<=", ">=", "->"):
        return True
    if tok_a.endswith(("=", "+", "-", "<", ">", "(", "[", "{")) and len(tok_b) <= 8:
        return True
    return False


def _clean_word(raw: str) -> str:
    return re.sub(r"<bookmark\b[^>]*>|\*", "", str(raw)).strip()


def _speaker_display(speaker: str) -> str:
    sp = (speaker or "narrator").strip()
    if sp.lower() in ("narrator", "host"):
        return "Narrator" if sp.lower() == "narrator" else "Host"
    return sp.replace("_", " ").title()


def _balance_lines(
    tokens: list[str],
    *,
    max_chars_per_line: int,
    font_px: int,
    max_width_px: float,
    max_lines: int = 2,
) -> list[str]:
    """Balance tokens across 1..max_lines avoiding trailing orphan words and line overflow."""
    if not tokens:
        return []
    joined = " ".join(tokens)
    est_w = estimate_text_width_px(joined, font_px, bold=True)
    if (len(joined) <= max_chars_per_line and est_w <= max_width_px) or len(tokens) == 1 or max_lines <= 1:
        return [joined]

    best_split = max(1, len(tokens) // 2)
    best_score = float("inf")
    for idx in range(1, len(tokens)):
        l1 = " ".join(tokens[:idx])
        l2 = " ".join(tokens[idx:])
        w1 = estimate_text_width_px(l1, font_px, bold=True)
        w2 = estimate_text_width_px(l2, font_px, bold=True)
        penalty = 0.0
        if len(l1) > max_chars_per_line or w1 > max_width_px:
            penalty += 500.0 + max(0.0, w1 - max_width_px)
        if len(l2) > max_chars_per_line or w2 > max_width_px:
            penalty += 500.0 + max(0.0, w2 - max_width_px)
        tail_word = tokens[idx - 1].lower().strip(".,!?;:\"'")
        if tail_word in _ORPHAN_TAILS:
            penalty += 120.0
        if idx == 1 or idx == len(tokens) - 1:
            penalty += 60.0
        if _is_math_expr_boundary(tokens[idx - 1], tokens[idx]):
            penalty += 85.0
        score = penalty + abs(w1 - w2) * 0.25 + abs(len(l1) - len(l2)) * 2.0
        if score < best_score:
            best_score = score
            best_split = idx

    l1 = " ".join(tokens[:best_split])
    l2 = " ".join(tokens[best_split:])
    if max_lines >= 3 and (len(l2) > max_chars_per_line or estimate_text_width_px(l2, font_px, bold=True) > max_width_px):
        mid = max(1, len(tokens[best_split:]) // 2)
        return [l1, " ".join(tokens[best_split : best_split + mid]), " ".join(tokens[best_split + mid :])]
    return [l1, l2]


def chunk_caption_cues(
    beats: Iterable[SyncedBeat],
    *,
    max_chars_per_line: int = 44,
    max_lines: int = 2,
    font_px: int = 44,
    max_width_px: float = 1480.0,
) -> list[CaptionCue]:
    """Segment beats into grammatical, pixel-bounded CaptionCue items with zero orphan endings."""
    cues: list[CaptionCue] = []
    cue_counter = 0
    max_chunk_chars = max(16, int(max_chars_per_line * max_lines * 0.92))

    def _emit_chunk(beat: SyncedBeat, chunk_words: list[CaptionWord]) -> None:
        nonlocal cue_counter
        if not chunk_words:
            return
        cue_counter += 1
        lines = _balance_lines(
            [w.text for w in chunk_words],
            max_chars_per_line=max_chars_per_line,
            font_px=font_px,
            max_width_px=max_width_px,
            max_lines=max_lines,
        )
        provs = [w.timing_provenance for w in chunk_words]
        cue_prov = (
            "tts_word_boundary"
            if all(p == "tts_word_boundary" for p in provs)
            else ("forced_aligned" if "forced_aligned" in provs else ("estimated" if all(p == "estimated" for p in provs) else "interpolated"))
        )
        cues.append(
            CaptionCue(
                cue_id=f"cue_{cue_counter:04d}",
                beat_id=beat.id,
                speaker=getattr(beat, "speaker", "narrator") or "narrator",
                speaker_label=_speaker_display(getattr(beat, "speaker", "narrator")),
                start_s=float(chunk_words[0].start_s),
                end_s=max(float(chunk_words[0].start_s) + 0.15, float(chunk_words[-1].end_s)),
                lines=lines,
                words=chunk_words,
                original_text=" ".join(w.text for w in chunk_words),
                spoken_text=" ".join(w.spoken_token or w.text for w in chunk_words),
                timing_provenance=cue_prov,  # type: ignore[arg-type]
            )
        )

    for b in beats:
        timed_words: list[CaptionWord] = []
        b_provs = list(getattr(b, "word_provenance", []) or [])
        if b.words and len(b.words) >= len((b.narration or "").split()):
            for wi, w in enumerate(b.words):
                tok = _clean_word(w[0])
                if not tok:
                    continue
                prov = b_provs[wi] if wi < len(b_provs) else (str(w[4]) if len(w) > 4 else "tts_word_boundary")
                if prov not in ("forced_aligned", "tts_word_boundary", "interpolated", "estimated"):
                    prov = "tts_word_boundary"
                timed_words.append(
                    CaptionWord(
                        text=tok,
                        start_s=float(w[1]),
                        end_s=float(w[2]),
                        emphasis=bool(w[3]) if len(w) > 3 else False,
                        spoken_token=tok,
                        timing_provenance=prov,  # type: ignore[arg-type]
                    )
                )
        if not timed_words:
            raw_txt = _clean_word(b.narration or b.caption)
            toks = [t for t in raw_txt.split() if t]
            if not toks:
                continue
            dur = max(0.3, float(b.speech_duration or (b.end_time - b.start_time)))
            step = dur / len(toks)
            for i, tok in enumerate(toks):
                timed_words.append(
                    CaptionWord(
                        text=tok,
                        start_s=round(float(b.start_time) + i * step, 4),
                        end_s=round(float(b.start_time) + (i + 1) * step, 4),
                        emphasis=False,
                        spoken_token=tok,
                        timing_provenance="estimated",
                    )
                )

        cur: list[CaptionWord] = []
        n = len(timed_words)
        for idx, cw in enumerate(timed_words):
            cand = [*cur, cw]
            joined = " ".join(x.text for x in cand)
            est_w = estimate_text_width_px(joined, font_px, bold=True)
            over_chars = len(joined) > max_chunk_chars
            over_px = est_w > max_width_px * max_lines * 0.94

            if cur and (over_chars or over_px):
                # Do not leave trailing orphan conjunctions when previous token wasn't terminal
                if len(cur) >= 3 and cur[-1].text.lower().strip(".,!?") in _ORPHAN_TAILS:
                    carry = cur.pop()
                    _emit_chunk(b, cur)
                    cur = [carry, cw]
                else:
                    _emit_chunk(b, cur)
                    cur = [cw]
            else:
                cur = cand

            next_tok = timed_words[idx + 1].text.lower().strip(".,!?;:") if idx + 1 < n else ""
            raw_next = timed_words[idx + 1].text if idx + 1 < n else ""
            cur_joined = " ".join(x.text for x in cur)
            remaining = n - idx - 1
            avoid_orphan_cue = remaining == 1  # Never leave a lone trailing 1-word cue in the beat

            hard_break = cw.text.endswith(_CLAUSE_BREAKS) and len(cur) >= 2 and not avoid_orphan_cue
            soft_break = (
                next_tok in _SOFT_BREAK_BEFORE
                and len(cur_joined) >= int(max_chars_per_line * 0.72)
                and len(cur) >= 3
                and not avoid_orphan_cue
                and not _is_math_expr_boundary(cw.text, raw_next)
            )
            if (hard_break or soft_break) and cur:
                _emit_chunk(b, cur)
                cur = []

        if cur:
            _emit_chunk(b, cur)

    return cues


def build_caption_ir(
    beats: Iterable[SyncedBeat],
    *,
    style: str = "editorial",
    width_px: int = 1920,
    height_px: int = 1080,
    max_chars_per_line: int | None = None,
    max_lines: int | None = None,
) -> CaptionIR:
    """Build full CaptionIR bound to pixel geometry, safe zones, and chosen CaptionStyle."""
    norm_style: CaptionStyle = normalize_caption_style(style)
    preset = CAPTION_PRESETS[norm_style]
    is_vert = height_px > width_px
    aspect: str = "9:16" if is_vert else "16:9"
    px_scale = effective_px_scale(width_px, height_px)
    side_margin_px = int(width_px * (0.06 if is_vert else 0.05))
    max_width_px = float(width_px - 2 * side_margin_px) * 0.88
    chars = max_chars_per_line or (preset.portrait_max_chars if is_vert else preset.landscape_max_chars)
    lines = max_lines or preset.max_lines
    ratio = preset.reserve_band_ratio_portrait if is_vert else preset.reserve_band_ratio_landscape
    reserve_band_px = int(round(height_px * ratio))

    beat_list = list(beats)
    cues = chunk_caption_cues(
        beat_list,
        max_chars_per_line=chars,
        max_lines=lines,
        font_px=px_scale.caption_px,
        max_width_px=max_width_px,
    )
    sp_labels = {
        (getattr(b, "speaker", "narrator") or "narrator"): _speaker_display(getattr(b, "speaker", "narrator"))
        for b in beat_list
    }
    prov_counts: dict[str, int] = {}
    for c in cues:
        for w in c.words:
            prov_counts[w.timing_provenance] = prov_counts.get(w.timing_provenance, 0) + 1
    return CaptionIR(
        style=norm_style,
        aspect_label="9:16" if is_vert else "16:9",
        width_px=width_px,
        height_px=height_px,
        font_px=px_scale.caption_px,
        max_width_px=round(max_width_px, 1),
        reserve_band_px=reserve_band_px,
        speaker_labels=sp_labels,
        cues=cues,
        preset=preset,
        timing_provenance_counts=prov_counts,
    )
