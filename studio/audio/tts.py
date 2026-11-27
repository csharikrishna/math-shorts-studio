"""Beat-synced neural TTS generator with silence trimming and exact timing manifests."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
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
    speaker: str = "narrator"


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
    speaker: str = "narrator"
    word_provenance: list[str] = field(default_factory=list)


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


def _trim_pcm_silence(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Trim leading/trailing silence below SILENCE_THRESHOLD_DB from mono int16 PCM."""
    thresh = 32767.0 * 10.0 ** (SILENCE_THRESHOLD_DB / 20.0)
    loud = np.flatnonzero(np.abs(pcm.astype(np.int32)) > thresh)
    if loud.size:
        i0 = max(0, int(loud[0]) - int(LEAD_PAD_S * sr))
        i1 = min(len(pcm), int(loud[-1]) + int(TAIL_PAD_S * sr))
    else:
        i0, i1 = 0, len(pcm)
    return pcm[i0:i1], i0 / float(sr)


def _write_trimmed_mono_wav(pcm: np.ndarray, out_wav: Path, sr: int = SAMPLE_RATE) -> tuple[float, float]:
    """Trim silence and write a mono 16-bit PCM WAV; returns (duration_s, head_trim_s)."""
    trimmed, head_trim = _trim_pcm_silence(pcm, sr=sr)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(trimmed.tobytes())
    return len(trimmed) / float(sr), head_trim


def _resample_mono_to_target_i16(pcm_f32: np.ndarray, src_sr: int, dst_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample normalized float mono audio to target sample rate int16 in memory (zero subprocess)."""
    x = np.asarray(pcm_f32, dtype=np.float32)
    if x.size == 0:
        return np.zeros(1, dtype=np.int16)
    if src_sr == dst_sr:
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    if dst_sr == 2 * src_sr and x.size >= 4:
        # 4-tap Catmull-Rom half-band interpolation preserves speech bandwidth cleanly without FFmpeg overhead
        xp = np.pad(x, (1, 2), mode="edge")
        mid = -0.0625 * xp[:-3] + 0.5625 * xp[1:-2] + 0.5625 * xp[2:-1] - 0.0625 * xp[3:]
        out = np.empty(x.size * 2, dtype=np.float32)
        out[0::2] = x
        out[1::2] = mid[: x.size]
        return (np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16)
    out_len = max(1, int(round(x.size * (dst_sr / float(src_sr)))))
    src_pos = np.linspace(0.0, float(x.size - 1), out_len, dtype=np.float64)
    up = np.interp(src_pos, np.arange(x.size, dtype=np.float64), x.astype(np.float64))
    return (np.clip(up, -1.0, 1.0) * 32767.0).astype(np.int16)


def _decode_and_trim_wav(ffmpeg_bin: str, in_mp3: Path, out_wav: Path) -> tuple[float, float]:
    """Decode to 48kHz mono WAV and trim dead silence; returns (duration, seconds trimmed from the head)."""
    proc = subprocess.run(
        [ffmpeg_bin, "-y", "-i", str(in_mp3), "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        check=True,
        capture_output=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return _write_trimmed_mono_wav(pcm, out_wav, sr=SAMPLE_RATE)


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


def _vad_active_span(pcm: np.ndarray, sr: int, fallback_dur: float) -> tuple[float, float]:
    """Lightweight local RMS voice-activity onset/offset detector on mono 16-bit PCM."""
    if pcm.size < max(1, int(0.04 * sr)):
        return 0.0, fallback_dur
    win = max(1, int(0.02 * sr))
    n_frames = pcm.size // win
    if n_frames <= 0:
        return 0.0, fallback_dur
    frames = np.abs(pcm[: n_frames * win].reshape(n_frames, win).astype(np.float32))
    rms = np.sqrt(np.mean(frames * frames, axis=1)) / 32768.0
    thresh = 10.0 ** (-38.0 / 20.0)
    active = np.flatnonzero(rms > thresh)
    if active.size == 0:
        return 0.0, fallback_dur
    return max(0.0, float(active[0] * win) / sr), min(fallback_dur, float((active[-1] + 1) * win) / sr)


def align_words_with_provenance(
    clean: str,
    tokens: list[ScriptToken],
    boundaries: list[list],
    speech_dur: float,
    *,
    pcm_mono: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Map TTS word boundaries onto script tokens while tracking per-word provenance."""
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
    matched_ratio = sum(1 for t in times if t is not None) / max(1, len(tokens))
    if matched_ratio < 0.25 and pcm_mono is not None:
        v0, v1 = _vad_active_span(pcm_mono, sr, speech_dur)
        active_span = max(0.1, v1 - v0)
        fallback_prov = "forced_aligned"
    else:
        v0, active_span = 0.0, speech_dur
        fallback_prov = "interpolated" if boundaries else "estimated"

    def by_chars(pos: int) -> float:
        return v0 + active_span * pos / total_chars

    known = [(tokens[i].char_start, times[i][0]) for i in range(len(tokens)) if times[i] is not None]
    out: list[tuple[float, float]] = []
    provs: list[str] = []
    for i, tok in enumerate(tokens):
        if times[i] is not None:
            out.append((times[i][0], times[i][1]))
            provs.append("tts_word_boundary")
            continue
        if known and matched_ratio >= 0.25:
            xs = np.array([k[0] for k in known] + [total_chars], dtype=float)
            ys = np.array([k[1] for k in known] + [speech_dur], dtype=float)
            if xs[0] > 0:
                xs = np.insert(xs, 0, 0.0)
                ys = np.insert(ys, 0, 0.0)
            start = float(np.interp(tok.char_start, xs, ys))
            end = float(np.interp(tok.char_end, xs, ys))
            provs.append("interpolated")
        else:
            start, end = by_chars(tok.char_start), by_chars(tok.char_end)
            provs.append(fallback_prov)
        out.append((start, max(start, end)))
    for i in range(1, len(out)):
        if out[i][0] < out[i - 1][0]:
            out[i] = (out[i - 1][0], max(out[i - 1][0], out[i][1]))
    return out, provs


def align_words(
    clean: str, tokens: list[ScriptToken], boundaries: list[list], speech_dur: float
) -> list[tuple[float, float]]:
    """Map TTS word boundaries onto script tokens (compat wrapper returning timings list)."""
    out, _provs = align_words_with_provenance(clean, tokens, boundaries, speech_dur)
    return out


# ---------------------------------------------------------------------------
# Optional local Kokoro-82M backend (falls back to edge-tts when unavailable)
# ---------------------------------------------------------------------------
KOKORO_SAMPLE_RATE = 24000
_KOKORO_PIPELINES: dict[str, object] = {}
_KOKORO_LOCK = threading.Lock()


def kokoro_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("kokoro") is not None


def _synth_kokoro_trimmed(text: str, voice: str, speed: float, out_wav: Path) -> tuple[list[list], float, float]:
    """Synthesize with Kokoro, resample 24->48kHz & trim silence in memory (zero FFmpeg spawn)."""
    from contextlib import nullcontext
    from kokoro import KPipeline  # type: ignore[import-not-found]

    try:
        import torch

        inf_ctx = torch.inference_mode()
    except Exception:
        inf_ctx = nullcontext()

    lang = voice[0] if voice and voice[0].isalpha() else "a"
    chunks: list[np.ndarray] = []
    words: list[list] = []
    offset = 0.0
    with _KOKORO_LOCK:
        pipe = _KOKORO_PIPELINES.get(lang)
        if pipe is None:
            pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
            _KOKORO_PIPELINES[lang] = pipe
        with inf_ctx:
            for result in pipe(text, voice=voice, speed=speed):  # type: ignore[operator]
                audio = result.audio
                audio = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
                for tok in result.tokens or []:
                    if getattr(tok, "start_ts", None) is not None and getattr(tok, "end_ts", None) is not None:
                        words.append([tok.text, offset + float(tok.start_ts), offset + float(tok.end_ts)])
                offset += len(audio) / float(KOKORO_SAMPLE_RATE)
                chunks.append(audio)
    pcm_f32 = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    pcm_48k = _resample_mono_to_target_i16(pcm_f32, KOKORO_SAMPLE_RATE, SAMPLE_RATE)
    dur, head_trim = _write_trimmed_mono_wav(pcm_48k, out_wav, sr=SAMPLE_RATE)
    return words, dur, head_trim


def _synth_kokoro(text: str, voice: str, speed: float, out_wav: Path) -> list[list]:
    """Legacy wrapper writing trimmed 48kHz Kokoro WAV directly."""
    words, _dur, _trim = _synth_kokoro_trimmed(text, voice, speed, out_wav)
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
    speaker_voices: dict[str, tuple[str, str]] | None = None,
) -> AudioTimeline:
    """Synthesize every beat (with cross-format shared cache & concurrent edge-tts), then assemble master WAV."""
    build_dir = Path(build_dir)
    segments_dir = build_dir / "voice_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    shared_dir: Path | None = None
    if os.environ.get("STUDIO_SHARED_CACHE", "1") != "0":
        shared_dir = Path(__file__).resolve().parent.parent.parent / "build" / "shared_cache" / "voice_segments"
        shared_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = find_ffmpeg()

    if engine == "kokoro" and not kokoro_available():
        if not allow_edge_fallback:
            raise RuntimeError("kokoro is not installed and allow_edge_fallback=False.")
        print("      ! kokoro not installed (uv sync --extra kokoro); falling back to edge-tts")
        engine = "edge"

    from studio.audio.pronounce import normalize_for_speech_with_map

    norm_map = {beat.id: normalize_for_speech_with_map(beat.narration) for beat in beats}

    async def _run_one(
        beat: BeatSpec,
        engine: str,
        active_voice: str,
        edge_sem: asyncio.Semaphore,
    ) -> tuple[BeatSpec, Path, float, list[list]]:
        mapped = norm_map[beat.id]
        spoken = str(mapped["spoken"])
        beat_voice = active_voice
        if speaker_voices and beat.speaker in speaker_voices:
            k_voice, e_voice = speaker_voices[beat.speaker]
            beat_voice = k_voice if engine == "kokoro" else e_voice
        eff_pitch = "" if engine == "kokoro" else pitch
        key = hashlib.sha1(f"v4|{engine}|{beat_voice}|{rate}|{eff_pitch}|{spoken}".encode("utf-8")).hexdigest()[:12]
        raw_path = segments_dir / f"{beat.id}_{key}.mp3"
        wav_path = segments_dir / f"{beat.id}_{key}.wav"
        words_path = segments_dir / f"{beat.id}_{key}.words.json"

        shared_wav = (shared_dir / f"{key}.wav") if shared_dir is not None else None
        shared_words = (shared_dir / f"{key}.words.json") if shared_dir is not None else None
        if not (wav_path.exists() and words_path.exists()) and shared_wav and shared_words and shared_wav.exists() and shared_words.exists():
            shutil.copy2(shared_wav, wav_path)
            shutil.copy2(shared_words, words_path)

        if not (wav_path.exists() and words_path.exists()):
            if engine == "kokoro":
                raw_words, dur, head_trim = _synth_kokoro_trimmed(spoken, beat_voice, _rate_to_speed(rate), wav_path)
            else:
                async with edge_sem:
                    raw_words = await _synth_edge_tts(spoken, beat_voice, rate, raw_path, pitch=pitch)
                dur, head_trim = _decode_and_trim_wav(ffmpeg_bin, raw_path, wav_path)
            words = [
                [w, round(max(0.0, a - head_trim), 4), round(min(dur, max(0.0, b - head_trim)), 4)]
                for w, a, b in raw_words
            ]
            words_path.write_text(json.dumps(words), encoding="utf-8")
            if shared_wav is not None and shared_words is not None:
                try:
                    shutil.copy2(wav_path, shared_wav)
                    shutil.copy2(words_path, shared_words)
                except OSError:
                    pass
        else:
            with wave.open(str(wav_path), "rb") as wf:
                dur = wf.getnframes() / float(wf.getframerate())
            words = json.loads(words_path.read_text(encoding="utf-8"))
        return (beat, wav_path, dur, words)

    async def _run_all(engine: str, active_voice: str) -> list[tuple[BeatSpec, Path, float, list[list]]]:
        edge_sem = asyncio.Semaphore(int(os.environ.get("STUDIO_EDGE_CONCURRENCY", "4")))
        if engine == "edge":
            return list(await asyncio.gather(*(_run_one(b, engine, active_voice, edge_sem) for b in beats)))
        results: list[tuple[BeatSpec, Path, float, list[list]]] = []
        for beat in beats:
            results.append(await _run_one(beat, engine, active_voice, edge_sem))
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
        with wave.open(str(wav_path), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        chunks.append(pcm)
        pause_samples = int(max(0.0, beat.pause_after) * SAMPLE_RATE)
        if pause_samples > 0:
            chunks.append(np.zeros(pause_samples, dtype=np.int16))

        clean, tokens, mark_idx = parse_script(beat.narration)
        mapped = norm_map[beat.id]
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
            sp_times, sp_provs = align_words_with_provenance(
                spoken_text, sp_tokens, boundaries, speech_dur, pcm_mono=pcm, sr=SAMPLE_RATE
            )
            token_times = []
            token_provs: list[str] = []
            for entry in mapped["token_map"]:  # type: ignore[attr-defined]
                r0, r1 = entry["spoken_range"]  # type: ignore[index]
                r0 = min(r0, max(0, len(sp_times) - 1))
                r1 = min(max(r0 + 1, r1), len(sp_times))
                token_times.append((sp_times[r0][0], sp_times[r1 - 1][1]))
                seg_provs = sp_provs[r0:r1]
                token_provs.append("tts_word_boundary" if all(p == "tts_word_boundary" for p in seg_provs) else (seg_provs[0] if seg_provs else "interpolated"))
        else:
            token_times, token_provs = align_words_with_provenance(
                clean, tokens, boundaries, speech_dur, pcm_mono=pcm, sr=SAMPLE_RATE
            )
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
                speaker=getattr(beat, "speaker", "narrator"),
                word_provenance=token_provs,
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
