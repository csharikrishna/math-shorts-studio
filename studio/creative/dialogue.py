"""Conversational and Socratic dialogue orchestration adapted from Podcastfy patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass

from studio.creative.models import NarrationMode, SpeakerRole


@dataclass(frozen=True)
class SpeakerIdentity:
    role: SpeakerRole
    label: str
    kokoro_voice: str
    edge_voice: str
    rate: str = "-4%"


@dataclass(frozen=True)
class DialogueProfile:
    mode: NarrationMode
    description: str
    speakers: tuple[SpeakerIdentity, ...]
    prompt_rules: str
    max_turn_words: int = 38


DIALOGUE_PROFILES: dict[NarrationMode, DialogueProfile] = {
    "single_educator": DialogueProfile(
        mode="single_educator",
        description="Single warm, curious educator walking through concrete intuition to mechanism.",
        speakers=(SpeakerIdentity("narrator", "Educator", "am_michael", "en-US-AndrewNeural", "-4%"),),
        prompt_rules=(
            "Single speaker (`speaker='narrator'`). Speak directly to a curious listener without greetings "
            "or dramatic cliches. Explain WHY every step follows from the visual state."
        ),
        max_turn_words=45,
    ),
    "conversational_explainer": DialogueProfile(
        mode="conversational_explainer",
        description="Accessible narrator explaining with concrete relatable everyday framing.",
        speakers=(SpeakerIdentity("educator", "Explainer", "af_bella", "en-US-EmmaNeural", "-3%"),),
        prompt_rules=(
            "Single speaker (`speaker='educator'`). Crisp conversational phrasing, active verbs, zero textbook stiffness."
        ),
        max_turn_words=40,
    ),
    "socratic": DialogueProfile(
        mode="socratic",
        description="Teacher and thoughtful learner resolving genuine misconceptions step by step.",
        speakers=(
            SpeakerIdentity("educator", "Teacher", "am_michael", "en-US-AndrewNeural", "-4%"),
            SpeakerIdentity("learner", "Learner", "af_bella", "en-US-AvaNeural", "-2%"),
        ),
        prompt_rules=(
            "Alternating `educator` and `learner` turns. Learner raises specific mathematical doubts or wrong guesses "
            "(NEVER empty 'Great question!' or 'Totally!'). Educator answers through visual actions and concrete numbers."
        ),
        max_turn_words=34,
    ),
    "discussion": DialogueProfile(
        mode="discussion",
        description="Two analytical peers examining multiple interpretations and edge cases.",
        speakers=(
            SpeakerIdentity("host_a", "Host A", "am_michael", "en-US-AndrewNeural", "-4%"),
            SpeakerIdentity("host_b", "Host B", "bm_george", "en-GB-RyanNeural", "-3%"),
        ),
        prompt_rules=(
            "Two-person analytical discussion (`host_a` and `host_b`). Zero small-talk greetings, zero artificial agreement; "
            "each turn advances the mathematical derivation, tests an assumption, or contrasts perspectives."
        ),
        max_turn_words=36,
    ),
    "documentary": DialogueProfile(
        mode="documentary",
        description="Measured documentary narrator suited for conceptual long-form arcs.",
        speakers=(SpeakerIdentity("narrator", "Narrator", "bm_george", "en-US-AndrewNeural", "-5%"),),
        prompt_rules=(
            "Single documentary narrator (`speaker='narrator'`). Clear rhetorical cadence with earned pauses between scenes; "
            "never melodramatic or vague."
        ),
        max_turn_words=48,
    ),
    "technical_instructor": DialogueProfile(
        mode="technical_instructor",
        description="Precise technical instructor highlighting invariants, assumptions, and derivations.",
        speakers=(SpeakerIdentity("educator", "Instructor", "am_adam", "en-US-BrianNeural", "-3%"),),
        prompt_rules=(
            "Single speaker (`speaker='educator'`). State assumptions explicitly, connect each formula symbol to visual objects."
        ),
        max_turn_words=44,
    ),
}

BANNED_CLICHES: tuple[str, ...] = (
    "welcome to a mathematical journey",
    "welcome to this episode",
    "welcome to the channel",
    "unlocking profound",
    "mathematical relationships that last forever",
    "game-changer",
    "revolutionize",
    "unlock the secret",
    "great question",
    "that's a fantastic point",
    "i couldn't agree more",
    "stay tuned",
    "after a quick break",
    "let's dive right in",
    "without further ado",
    "treacherous mountain range",
)

_SCRATCHPAD_RE = re.compile(r"```(?:scratchpad|plaintext)?.*?```|\[(?:pause|chuckle|laughs?|sighs?|music).*?\]", re.I | re.S)


def detect_banned_cliches(text: str) -> list[str]:
    low = text.lower()
    return [phrase for phrase in BANNED_CLICHES if phrase in low]


def clean_spoken_text(text: str) -> str:
    """Strip scratchpad/stage tags cleanly while keeping *emphasis* and <bookmark/> cues."""
    out = _SCRATCHPAD_RE.sub(" ", text)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def format_dialogue_instructions(mode: NarrationMode) -> str:
    prof = DIALOGUE_PROFILES.get(mode, DIALOGUE_PROFILES["single_educator"])
    sp_desc = ", ".join(f"{s.role} ({s.label})" for s in prof.speakers)
    return (
        f"NARRATION MODE: {prof.mode} — {prof.description}\n"
        f"ALLOWED SPEAKERS: {sp_desc}\n"
        f"DIALOGUE RULES: {prof.prompt_rules}\n"
        f"ANTI-FILLER POLICY: Never use greetings, host banter, empty agreement, or phrases like: "
        + "; ".join(f'"{c}"' for c in BANNED_CLICHES[:8])
    )


def merge_consecutive_speaker_turns(turns: list[tuple[SpeakerRole, str]]) -> list[tuple[SpeakerRole, str]]:
    """Reconcile consecutive identical speaker turns cleanly (adapted from Podcastfy alternation guard)."""
    merged: list[tuple[SpeakerRole, str]] = []
    for role, text in turns:
        cleaned = clean_spoken_text(text)
        if not cleaned:
            continue
        if merged and merged[-1][0] == role:
            prev_role, prev_text = merged[-1]
            merged[-1] = (prev_role, f"{prev_text} {cleaned}".strip())
        else:
            merged.append((role, cleaned))
    return merged
