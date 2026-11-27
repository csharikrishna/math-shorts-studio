"""Mathematical pronunciation normalization for natural TTS narration."""

from __future__ import annotations

import re

_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsin\(([^)]+)\)"), r"sine of \1"),
    (re.compile(r"\bcos\(([^)]+)\)"), r"cosine of \1"),
    (re.compile(r"\btan\(([^)]+)\)"), r"tangent of \1"),
    (re.compile(r"\b([a-zA-Z])'\(([^)]+)\)"), r"the derivative of \1 at \2"),
    (re.compile(r"\b([a-zA-Z])'"), r"the derivative of \1"),
    (re.compile(r"([a-zA-Z])\s*(?:→|->)\s*∞"), r"as \1 approaches infinity"),
    (re.compile(r"([a-zA-Z])\s*(?:→|->)\s*([0-9a-zA-Z-]+)"), r"as \1 approaches \2"),
    (re.compile(r"Σ|∑"), "the sum"),
    (re.compile(r"π"), "pi"),
    (re.compile(r"∞"), "infinity"),
    (re.compile(r"≈"), "approximately equals"),
    (re.compile(r"≠"), "does not equal"),
    (re.compile(r"≤"), "is less than or equal to"),
    (re.compile(r"≥"), "is greater than or equal to"),
    (re.compile(r"\b([A-Za-z0-9]+)\^2\b"), r"\1 squared"),
    (re.compile(r"\b([A-Za-z0-9]+)\^3\b"), r"\1 cubed"),
    (re.compile(r"([A-Za-z0-9])²"), r"\1 squared"),
    (re.compile(r"([A-Za-z0-9])³"), r"\1 cubed"),
    (re.compile(r"\b1/2\b"), "one half"),
    (re.compile(r"\b([A-Za-z])_(\d+)\b"), r"\1 \2"),
]


def normalize_for_speech(text: str) -> str:
    """Convert mathematical notation into natural spoken English while preserving <bookmark/> and *emphasis*."""
    out = text
    for pattern, repl in _REPLACEMENTS:
        out = pattern.sub(repl, out)
    return re.sub(r"\s{2,}", " ", out).strip()


def normalize_for_speech_with_map(text: str) -> dict[str, object]:
    """Normalize narration for TTS while returning original, spoken, and bookmark metadata."""
    from studio.audio.tts import BOOKMARK_RE

    marks = BOOKMARK_RE.findall(text)
    spoken = normalize_for_speech(text)
    return {"original": text, "spoken": spoken, "bookmarks": marks}
