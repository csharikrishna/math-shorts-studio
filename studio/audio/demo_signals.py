"""Deterministic synthesized mathematical audio demonstrations (clean 48kHz stereo WAV)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional
import wave
import numpy as np

from studio.ai.linked_models import LinkedMathModelSpec, evaluate_signal_spectral_pipeline


def _resample_and_pad(
    samples: list[float],
    *,
    sample_rate: int = 48000,
    duration_s: float = 2.4,
    base_pitch_hz: float = 220.0,
) -> np.ndarray:
    """Sonify a discrete mathematical signal trajectory into a clean anti-aliased audio clip."""
    n_out = max(sample_rate // 2, int(round(duration_s * sample_rate)))
    arr = np.asarray(samples, dtype=float)
    if arr.size < 4:
        t = np.linspace(0.0, duration_s, n_out, endpoint=False)
        sig = np.sin(2.0 * math.pi * base_pitch_hz * t)
    else:
        # Blend normalized envelope/harmonics at audible pitch scaled from math trajectory
        t_audio = np.linspace(0.0, duration_s, n_out, endpoint=False)
        src_x = np.linspace(0.0, 1.0, arr.size)
        dst_x = np.linspace(0.0, 1.0, n_out)
        interp = np.interp(dst_x, src_x, arr)
        mx = max(1e-6, float(np.max(np.abs(interp))))
        norm_mod = interp / mx
        # Synthesize clean fundamental + harmonic timbre modulated faithfully by the mathematical signal
        sig = 0.55 * norm_mod * np.sin(2.0 * math.pi * base_pitch_hz * t_audio) + 0.45 * np.sin(
            2.0 * math.pi * (base_pitch_hz * 0.5 + 0.5 * base_pitch_hz * norm_mod) * t_audio
        )
    # Apply 60ms smooth cosine fade-in / fade-out (zero boundary clicks)
    fade_n = min(n_out // 6, int(0.06 * sample_rate))
    if fade_n > 1:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, fade_n)))
        sig[:fade_n] *= ramp
        sig[-fade_n:] *= ramp[::-1]
    peak = max(1e-9, float(np.max(np.abs(sig))))
    sig = sig * (0.251 / peak)  # Clean -12 dBFS target peak
    return sig


def synthesize_math_audio_demo(
    spec: LinkedMathModelSpec,
    out_wav: Path,
    *,
    variant: str = "raw_vs_filtered",
    base_pitch_hz: float = 220.0,
    duration_s: float = 2.4,
    sample_rate: int = 48000,
) -> Path:
    """Render a real 48kHz stereo WAV sound demonstration directly from a LinkedMathModelSpec."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    pipe = evaluate_signal_spectral_pipeline(spec) if spec.kind == "signal_spectral_pipeline" else None
    raw_samples = list(pipe.get("raw_samples", [])) if pipe else []
    filt_samples = list(pipe.get("filtered_samples", [])) if pipe else []

    if variant == "raw":
        mono = _resample_and_pad(raw_samples, sample_rate=sample_rate, duration_s=duration_s, base_pitch_hz=base_pitch_hz)
    elif variant == "filtered":
        mono = _resample_and_pad(filt_samples or raw_samples, sample_rate=sample_rate, duration_s=duration_s, base_pitch_hz=base_pitch_hz)
    else:
        half = max(0.8, duration_s * 0.48)
        s_raw = _resample_and_pad(raw_samples, sample_rate=sample_rate, duration_s=half, base_pitch_hz=base_pitch_hz)
        gap = np.zeros(int(0.12 * sample_rate), dtype=float)
        s_flt = _resample_and_pad(filt_samples or raw_samples, sample_rate=sample_rate, duration_s=half, base_pitch_hz=base_pitch_hz)
        mono = np.concatenate([s_raw, gap, s_flt])

    pcm = np.clip(mono * 32767.0, -32767.0, 32767.0).astype(np.int16)
    stereo = np.column_stack([pcm, pcm]).ravel()
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())
    return out_wav


def mix_audio_demos_over_track(
    base_wav: Path,
    demo_events: Iterable[tuple[float, Path, float]],
    out_wav: Path,
) -> Path:
    """Overlay (start_s, demo_wav_path, gain) cleanly into a 48kHz WAV timeline without ffmpeg graph bloat."""
    base_wav = Path(base_wav)
    out_wav = Path(out_wav)
    events = [(float(st), Path(dp), float(g)) for st, dp, g in demo_events if Path(dp).exists()]
    if not events:
        return base_wav
    with wave.open(str(base_wav), "rb") as rf:
        n_ch = rf.getnchannels()
        sr = rf.getframerate()
        raw = np.frombuffer(rf.readframes(rf.getnframes()), dtype=np.int16).astype(np.float32)
    if n_ch == 2:
        buf = raw.reshape(-1, 2)
    else:
        buf = np.column_stack([raw, raw])

    for st_s, dpath, gain in events:
        with wave.open(str(dpath), "rb") as df:
            d_ch = df.getnchannels()
            d_raw = np.frombuffer(df.readframes(df.getnframes()), dtype=np.int16).astype(np.float32)
        dbuf = d_raw.reshape(-1, 2) if d_ch == 2 else np.column_stack([d_raw, d_raw])
        idx0 = max(0, int(round(st_s * sr)))
        if idx0 >= buf.shape[0]:
            continue
        n_copy = min(dbuf.shape[0], buf.shape[0] - idx0)
        buf[idx0 : idx0 + n_copy] = np.clip(buf[idx0 : idx0 + n_copy] + dbuf[:n_copy] * gain, -32767.0, 32767.0)

    out_pcm = buf.astype(np.int16).ravel()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out_pcm.tobytes())
    return out_wav


def apply_plan_audio_demos(plan: object, audio_timeline: object, build_dir: Path) -> Path:
    """Synthesize and mix any SceneV3.demo_audio clips tied to plan.math_models into master voice WAV."""
    base_wav = Path(getattr(audio_timeline, "master_voice_wav", ""))
    math_models = {m.model_id: m for m in (getattr(plan, "math_models", []) or [])}
    scenes = getattr(plan, "scenes", []) or []
    if not math_models or not base_wav.exists():
        return base_wav
    beats_by_id = {b.id: b for b in (getattr(audio_timeline, "beats", []) or [])}
    events: list[tuple[float, Path, float]] = []
    demo_dir = Path(build_dir) / "audio_demos"
    for sc in scenes:
        da = getattr(sc, "demo_audio", None)
        if da is None or da.model_id not in math_models:
            continue
        sb = beats_by_id.get(sc.scene_id)
        if sb is None:
            continue
        demo_wav = demo_dir / f"{sc.scene_id}_{da.model_id}.wav"
        synthesize_math_audio_demo(
            math_models[da.model_id],
            demo_wav,
            variant=da.variant,
            base_pitch_hz=float(da.base_pitch_hz or 220.0),
            duration_s=float(da.duration_s or 2.0),
        )
        marks = getattr(sb, "marks", {}) or {}
        if da.anchor_mark and da.anchor_mark in marks:
            t_at = float(marks[da.anchor_mark])
        else:
            t_at = max(float(sb.start_time), float(sb.end_time) - float(da.duration_s or 2.0) - 0.1)
        events.append((t_at, demo_wav, 0.72))
    if not events:
        return base_wav
    return mix_audio_demos_over_track(base_wav, events, Path(build_dir) / "master_voice_with_demos.wav")

