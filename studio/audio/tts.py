"""Beat-synced neural TTS generator with silence trimming and exact timing manifests."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import wave
import edge_tts
import numpy as np

SAMPLE_RATE = 48000


def find_ffmpeg() -> str:
    """Locate ffmpeg executable via PATH or standard Windows Python script locations."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        Path.home() / "AppData/Local/Programs/Python/Python312/Scripts/ffmpeg.exe",
        Path.home() / "AppData/Local/Programs/Python/Python313/Scripts/ffmpeg.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("ffmpeg executable not found in PATH.")


@dataclass(frozen=True)
class BeatSpec:
    """Single narrative beat specification (no hardcoded duration)."""

    id: str
    narration: str
    caption: str
    pause_after: float = 0.25


@dataclass(frozen=True)
class SyncedBeat:
    """Resolved beat with exact measured speech timestamps."""

    id: str
    narration: str
    caption: str
    audio_path: str
    start_time: float
    end_time: float
    speech_duration: float
    total_duration: float
    # Absolute-time word timings [[text, start_s, end_s], ...] from edge-tts WordBoundary events
    words: list = field(default_factory=list)
    # Bookmark name -> absolute time (s), resolved from word boundaries
    marks: dict = field(default_factory=dict)


@dataclass
class AudioTimeline:
    """Full synchronized voiceover manifest for a scene."""

    voice: str
    rate: str
    master_voice_wav: str
    total_duration: float
    beats: list[SyncedBeat]

    def by_id(self, beat_id: str) -> SyncedBeat:
        for b in self.beats:
            if b.id == beat_id:
                return b
        raise KeyError(f"Beat '{beat_id}' not found in timeline.")

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "voice": self.voice,
            "rate": self.rate,
            "master_voice_wav": self.master_voice_wav,
            "total_duration": self.total_duration,
            "beats": [asdict(b) for b in self.beats],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> AudioTimeline:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            voice=raw["voice"],
            rate=raw["rate"],
            master_voice_wav=raw["master_voice_wav"],
            total_duration=raw["total_duration"],
            beats=[SyncedBeat(**b) for b in raw["beats"]],
        )


WORD_TICKS_PER_SECOND = 10_000_000  # edge-tts offsets/durations are in 100 ns units
SILENCE_THRESHOLD_DB = -45.0
LEAD_PAD_S = 0.04
TAIL_PAD_S = 0.05


async def _synth_edge_tts(text: str, voice: str, rate: str, out_mp3: Path, pitch: str = "+0Hz") -> list[list]:
    """Stream edge-tts audio while capturing WordBoundary timestamps (seconds, untrimmed)."""
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch, boundary="WordBoundary")
    words: list[list] = []
    with open(out_mp3, "wb") as fh:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                fh.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                t0 = chunk["offset"] / WORD_TICKS_PER_SECOND
                t1 = t0 + chunk["duration"] / WORD_TICKS_PER_SECOND
                words.append([chunk["text"], t0, t1])
    return words


def _decode_and_trim_wav(ffmpeg_bin: str, in_mp3: Path, out_wav: Path) -> tuple[float, float]:
    """Decode to 48kHz mono WAV and trim dead silence; returns (duration, seconds trimmed from the head)."""
    proc = subprocess.run(
        [ffmpeg_bin, "-y", "-i", str(in_mp3), "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        check=True,
        capture_output=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    thresh = 32767.0 * 10.0 ** (SILENCE_THRESHOLD_DB / 20.0)
    loud = np.flatnonzero(np.abs(pcm.astype(np.int32)) > thresh)
    if loud.size:
        i0 = max(0, int(loud[0]) - int(LEAD_PAD_S * SAMPLE_RATE))
        i1 = min(len(pcm), int(loud[-1]) + int(TAIL_PAD_S * SAMPLE_RATE))
    else:
        i0, i1 = 0, len(pcm)
    trimmed = pcm[i0:i1]
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(trimmed.tobytes())
    return len(trimmed) / float(SAMPLE_RATE), i0 / float(SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Script markup: *emphasis* and manim-voiceover style <bookmark mark='name'/>
# ---------------------------------------------------------------------------
BOOKMARK_RE = re.compile(r"<bookmark\s+mark\s*=\s*['\"](\w+)['\"]\s*/>")


@dataclass(frozen=True)
class ScriptToken:
    text: str  # display text, punctuation kept
    char_start: int  # offset inside the clean spoken text
    char_end: int
    emphasis: bool


def parse_script(narration: str) -> tuple[str, list[ScriptToken], dict[str, int]]:
    """Split narration into spoken text, display tokens, and bookmark -> token index."""
    spaced = BOOKMARK_RE.sub(lambda mt: f" \x00{mt.group(1)}\x00 ", narration)
    tokens: list[ScriptToken] = []
    marks: dict[str, int] = {}
    in_emph = False
    cursor = 0
    for raw in spaced.split():
        if raw.startswith("\x00") and raw.endswith("\x00"):
            marks[raw.strip("\x00")] = len(tokens)
            continue
        emph = in_emph or raw.startswith("*")
        if raw.count("*") % 2 == 1:
            in_emph = not in_emph
        text = raw.replace("*", "")
        if not text:
            continue
        if tokens:
            cursor += 1  # joining space
        tokens.append(ScriptToken(text, cursor, cursor + len(text), emph))
        cursor += len(text)
    clean = " ".join(t.text for t in tokens)
    return clean, tokens, marks


def align_words(
    clean: str, tokens: list[ScriptToken], boundaries: list[list], speech_dur: float
) -> list[tuple[float, float]]:
    """Map TTS word boundaries onto script tokens (MoneyPrinterTurbo-style script alignment).

    Each boundary word is located in the clean text at/after a moving cursor; a token takes the
    earliest start / latest end of the boundary words that land in it. Tokens with no boundary
    (symbols, unmatched words) are interpolated by character position, the manim-voiceover
    fallback. Times are relative to the trimmed clip.
    """
    lower = clean.lower()
    times: list[list[float] | None] = [None] * len(tokens)
    cursor = 0
    for word, t0, t1 in boundaries:
        needle = str(word).lower().strip()
        if not needle:
            continue
        pos = lower.find(needle, cursor)
        if pos < 0 or pos - cursor > 40:
            continue
        cursor = pos + len(needle)
        for i, tok in enumerate(tokens):
            if tok.char_start <= pos < tok.char_end + 1:
                if times[i] is None:
                    times[i] = [float(t0), float(t1)]
                else:
                    times[i][0] = min(times[i][0], float(t0))
                    times[i][1] = max(times[i][1], float(t1))
                break

    total_chars = max(1, len(clean))

    def by_chars(pos: int) -> float:
        return speech_dur * pos / total_chars

    known = [(tokens[i].char_start, times[i][0]) for i in range(len(tokens)) if times[i] is not None]
    out: list[tuple[float, float]] = []
    for i, tok in enumerate(tokens):
        if times[i] is not None:
            out.append((times[i][0], times[i][1]))
            continue
        if known:
            xs = np.array([k[0] for k in known] + [total_chars], dtype=float)
            ys = np.array([k[1] for k in known] + [speech_dur], dtype=float)
            if xs[0] > 0:
                xs = np.insert(xs, 0, 0.0)
                ys = np.insert(ys, 0, 0.0)
            start = float(np.interp(tok.char_start, xs, ys))
            end = float(np.interp(tok.char_end, xs, ys))
        else:
            start, end = by_chars(tok.char_start), by_chars(tok.char_end)
        out.append((start, max(start, end)))
    # Enforce monotonic starts (TTS boundaries can jitter around punctuation)
    for i in range(1, len(out)):
        if out[i][0] < out[i - 1][0]:
            out[i] = (out[i - 1][0], max(out[i - 1][0], out[i][1]))
    return out


# ---------------------------------------------------------------------------
# Optional local Kokoro-82M backend (falls back to edge-tts when unavailable)
# ---------------------------------------------------------------------------
KOKORO_SAMPLE_RATE = 24000
_KOKORO_PIPELINES: dict[str, object] = {}


def kokoro_available() -> bool:
    try:
        import kokoro  # noqa: F401
    except Exception:
        return False
    return True


def _synth_kokoro(text: str, voice: str, speed: float, out_wav: Path) -> list[list]:
    """Synthesize with Kokoro; per-token start_ts/end_ts become word boundaries (seconds)."""
    from kokoro import KPipeline  # type: ignore[import-not-found]

    lang = voice[0] if voice and voice[0].isalpha() else "a"
    pipe = _KOKORO_PIPELINES.get(lang)
    if pipe is None:
        pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
        _KOKORO_PIPELINES[lang] = pipe
    chunks: list[np.ndarray] = []
    words: list[list] = []
    offset = 0.0
    for result in pipe(text, voice=voice, speed=speed):  # type: ignore[operator]
        audio = result.audio
        audio = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
        for tok in result.tokens or []:
            if getattr(tok, "start_ts", None) is not None and getattr(tok, "end_ts", None) is not None:
                words.append([tok.text, offset + float(tok.start_ts), offset + float(tok.end_ts)])
        offset += len(audio) / float(KOKORO_SAMPLE_RATE)
        chunks.append(audio)
    pcm = np.clip(np.concatenate(chunks) if chunks else np.zeros(1), -1.0, 1.0)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(KOKORO_SAMPLE_RATE)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return words


def _rate_to_speed(rate: str) -> float:
    try:
        return 1.0 + float(rate.strip().rstrip("%")) / 100.0
    except ValueError:
        return 1.0


# ---------------------------------------------------------------------------
# Timeline assembly
# ---------------------------------------------------------------------------
def build_voice_timeline(
    beats: list[BeatSpec],
    build_dir: Path,
    voice: str = "en-US-AndrewNeural",
    rate: str = "+12%",
    lead_in: float = 0.25,
    pitch: str = "+0Hz",
    engine: str = "edge",
    kokoro_voice: str = "am_michael",
    allow_edge_fallback: bool = True,
) -> AudioTimeline:
    """Synthesize every beat independently, measure exact durations, and assemble master voiceover."""
    build_dir = Path(build_dir)
    segments_dir = build_dir / "voice_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = find_ffmpeg()

    if engine == "kokoro" and not kokoro_available():
        if not allow_edge_fallback:
            raise RuntimeError("kokoro is not installed and allow_edge_fallback=False.")
        print("      ! kokoro not installed (uv sync --extra kokoro); falling back to edge-tts")
        engine = "edge"

    async def _run_all(engine: str, active_voice: str) -> list[tuple[BeatSpec, Path, float, list[list]]]:
        results: list[tuple[BeatSpec, Path, float, list[list]]] = []
        for beat in beats:
            from studio.audio.pronounce import normalize_for_speech_with_map

            mapped = normalize_for_speech_with_map(beat.narration)
            spoken = str(mapped["spoken"])
            key = hashlib.sha1(f"v4|{engine}|{active_voice}|{rate}|{pitch}|{spoken}".encode("utf-8")).hexdigest()[:12]
            raw_path = segments_dir / f"{beat.id}_{key}.{'raw.wav' if engine == 'kokoro' else 'mp3'}"
            wav_path = segments_dir / f"{beat.id}_{key}.wav"
            words_path = segments_dir / f"{beat.id}_{key}.words.json"
            if not (wav_path.exists() and words_path.exists()):
                if engine == "kokoro":
                    raw_words = _synth_kokoro(spoken, active_voice, _rate_to_speed(rate), raw_path)
                else:
                    raw_words = await _synth_edge_tts(spoken, voice, rate, raw_path, pitch=pitch)
                dur, head_trim = _decode_and_trim_wav(ffmpeg_bin, raw_path, wav_path)
                # Re-base word times onto the trimmed clip so captions lock to the audible syllables
                words = [
                    [w, round(max(0.0, a - head_trim), 4), round(min(dur, max(0.0, b - head_trim)), 4)]
                    for w, a, b in raw_words
                ]
                words_path.write_text(json.dumps(words), encoding="utf-8")
            else:
                with wave.open(str(wav_path), "rb") as wf:
                    dur = wf.getnframes() / float(wf.getframerate())
                words = json.loads(words_path.read_text(encoding="utf-8"))
            results.append((beat, wav_path, dur, words))
        return results

    active_voice = kokoro_voice if engine == "kokoro" else voice
    try:
        synthesized = asyncio.run(_run_all(engine, active_voice))
    except Exception as exc:  # noqa: BLE001 - one engine for the whole Short, never mixed voices
        if engine != "kokoro" or not allow_edge_fallback:
            raise
        print(f"      ! kokoro synthesis failed ({type(exc).__name__}: {exc}); re-voicing with edge-tts")
        engine, active_voice = "edge", voice
        synthesized = asyncio.run(_run_all(engine, active_voice))

    # Assemble master 48kHz mono WAV with exact beat offsets
    cursor = lead_in
    synced_beats: list[SyncedBeat] = []
    chunks: list[np.ndarray] = [np.zeros(int(lead_in * SAMPLE_RATE), dtype=np.int16)]

    for beat, wav_path, speech_dur, boundaries in synthesized:
        from studio.audio.pronounce import normalize_for_speech_with_map

        with wave.open(str(wav_path), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        chunks.append(pcm)
        pause_samples = int(max(0.0, beat.pause_after) * SAMPLE_RATE)
        if pause_samples > 0:
            chunks.append(np.zeros(pause_samples, dtype=np.int16))

        clean, tokens, mark_idx = parse_script(beat.narration)
        mapped = normalize_for_speech_with_map(beat.narration)
        spoken_text = str(mapped["spoken"])
        sp_words = spoken_text.split()
        if sp_words and sp_words != [t.text for t in tokens]:
            # Align TTS boundaries against normalized spoken tokens, then map back to display tokens
            sp_tokens: list[ScriptToken] = []
            scur = 0
            for sw in sp_words:
                if sp_tokens:
                    scur += 1
                sp_tokens.append(ScriptToken(sw, scur, scur + len(sw), False))
                scur += len(sw)
            sp_times = align_words(spoken_text, sp_tokens, boundaries, speech_dur)
            token_times = []
            for entry in mapped["token_map"]:  # type: ignore[attr-defined]
                r0, r1 = entry["spoken_range"]  # type: ignore[index]
                r0 = min(r0, max(0, len(sp_times) - 1))
                r1 = min(max(r0 + 1, r1), len(sp_times))
                token_times.append((sp_times[r0][0], sp_times[r1 - 1][1]))
        else:
            token_times = align_words(clean, tokens, boundaries, speech_dur)
        words = [
            [tok.text, round(cursor + a, 4), round(cursor + b, 4), tok.emphasis]
            for tok, (a, b) in zip(tokens, token_times)
        ]
        marks = {
            name: round(cursor + (token_times[i][0] if i < len(token_times) else speech_dur), 4)
            for name, i in mark_idx.items()
        }

        total_beat_dur = speech_dur + beat.pause_after
        synced_beats.append(
            SyncedBeat(
                id=beat.id,
                narration=clean,
                caption=beat.caption,
                audio_path=str(wav_path),
                start_time=round(cursor, 4),
                end_time=round(cursor + total_beat_dur, 4),
                speech_duration=round(speech_dur, 4),
                total_duration=round(total_beat_dur, 4),
                words=words,
                marks=marks,
            )
        )
        cursor += total_beat_dur

    master_pcm = np.concatenate(chunks)
    master_wav = build_dir / "voiceover_master.wav"
    with wave.open(str(master_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(master_pcm.tobytes())

    timeline = AudioTimeline(
        voice=active_voice,
        rate=rate,
        master_voice_wav=str(master_wav),
        total_duration=round(cursor, 4),
        beats=synced_beats,
    )
    timeline.save(build_dir / "timeline.json")
    return timeline
