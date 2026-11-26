"""Procedural studio-grade sound effects synthesizer (zero external audio dependencies)."""

from pathlib import Path
import wave
import numpy as np

SAMPLE_RATE = 48000


def _svf_bandpass(signal: np.ndarray, freq_curve: np.ndarray, q: float = 2.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply a time-varying state-variable bandpass filter for smooth resonant sweeps."""
    out = np.zeros_like(signal)
    low = 0.0
    band = 0.0
    for i in range(len(signal)):
        f = 2.0 * np.sin(np.pi * min(freq_curve[i], sr * 0.4) / sr)
        high = signal[i] - low - band / q
        band = band + f * high
        low = low + f * band
        out[i] = band
    return out


def _stereo_room(left: np.ndarray, right: np.ndarray, wet: float = 0.22, tail_s: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Small decorrelated stereo room: prime-spaced comb taps per side widen the image."""
    pad = int(tail_s * SAMPLE_RATE)
    l_out = np.concatenate([left, np.zeros(pad)])
    r_out = np.concatenate([right, np.zeros(pad)])
    dry_l, dry_r = l_out.copy(), r_out.copy()
    taps_l = [(0.0231, 0.55), (0.0397, 0.42), (0.0613, 0.30), (0.0887, 0.20), (0.1301, 0.12)]
    taps_r = [(0.0263, 0.55), (0.0431, 0.42), (0.0671, 0.30), (0.0953, 0.20), (0.1409, 0.12)]
    for taps, src, dst in ((taps_l, dry_r, l_out), (taps_r, dry_l, r_out)):
        for delay_s, gain in taps:
            d = int(delay_s * SAMPLE_RATE)
            dst[d:] += wet * gain * src[: len(src) - d]
    return l_out, r_out


def _pan(mono: np.ndarray, position: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-power pan; position in [-1 (left), +1 (right)]."""
    angle = (position + 1.0) * np.pi / 4.0
    return mono * np.cos(angle), mono * np.sin(angle)


def _write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.stack([left, right], axis=1)
    peak = np.max(np.abs(stereo))
    if peak > 0.95:
        stereo = stereo * (0.92 / peak)
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def _synth_whoosh_soft() -> tuple[np.ndarray, np.ndarray]:
    dur = 0.42
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    rng = np.random.default_rng(101)
    noise = rng.normal(0.0, 1.0, n)
    freq_sweep = 240.0 + 1350.0 * np.sin(np.pi * (t / dur)) ** 1.4
    filtered = _svf_bandpass(noise, freq_sweep, q=1.8)
    env = np.sin(np.pi * (t / dur)) ** 1.6
    # Add subtle warm sub-body
    sub = 0.18 * np.sin(2.0 * np.pi * (95.0 + 70.0 * (t / dur)) * t) * env
    mono = filtered * env * 0.45 + sub
    # Slight stereo Haas widening
    delay = int(0.008 * SAMPLE_RATE)
    right = np.roll(mono, delay)
    right[:delay] = 0.0
    return mono, right


def _synth_pop_marimba() -> tuple[np.ndarray, np.ndarray]:
    dur = 0.22
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    f0 = 440.0
    pitch = f0 + 260.0 * np.exp(-t * 85.0)
    phase = 2.0 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    body = (
        0.70 * np.sin(phase) * np.exp(-t * 22.0)
        + 0.22 * np.sin(phase * 2.0) * np.exp(-t * 36.0)
        + 0.10 * np.sin(phase * 3.98) * np.exp(-t * 55.0)
    )
    rng = np.random.default_rng(202)
    click = rng.normal(0.0, 0.25, n) * np.exp(-t * 260.0)
    mono = (body + click) * 0.72
    return _stereo_room(*_pan(mono, -0.18), wet=0.18, tail_s=0.35)


def _synth_tick_soft() -> tuple[np.ndarray, np.ndarray]:
    dur = 0.09
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    f0 = 980.0 + 600.0 * np.exp(-t * 140.0)
    phase = 2.0 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    mono = np.sin(phase) * np.exp(-t * 55.0) * 0.55
    return _stereo_room(*_pan(mono, 0.22), wet=0.20, tail_s=0.30)


def _synth_chime_reveal() -> tuple[np.ndarray, np.ndarray]:
    dur = 1.35
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    freqs = [587.33, 880.00, 1174.66, 1760.00]  # D5, A5, D6, A6
    amps = [0.38, 0.28, 0.20, 0.12]
    decays = [2.6, 3.2, 4.0, 5.2]
    attack = 1.0 - np.exp(-t * 120.0)
    left = np.zeros(n)
    right = np.zeros(n)
    for f, a, d in zip(freqs, amps, decays):
        left += a * np.sin(2.0 * np.pi * f * t) * np.exp(-t * d)
        right += a * np.sin(2.0 * np.pi * (f * 1.003) * t) * np.exp(-t * d)
    return _stereo_room(left * attack * 0.65, right * attack * 0.65, wet=0.26, tail_s=0.5)


def _synth_chord_epiphany() -> tuple[np.ndarray, np.ndarray]:
    dur = 2.4
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    # Warm Cmaj9 chord: C4, G4, D5, E5, B5
    notes = [261.63, 392.00, 587.33, 659.25, 987.77]
    amps = [0.28, 0.24, 0.18, 0.18, 0.12]
    attack = 1.0 - np.exp(-t * 35.0)
    left = np.zeros(n)
    right = np.zeros(n)
    for idx, (f, a) in enumerate(zip(notes, amps)):
        detune = 1.0 + (0.0025 if idx % 2 == 0 else -0.0025)
        decay = np.exp(-t * (1.2 + idx * 0.25))
        left += a * np.sin(2.0 * np.pi * f * t) * decay
        right += a * np.sin(2.0 * np.pi * (f * detune) * t) * decay
    # Warm low octave anchor
    sub = 0.22 * np.sin(2.0 * np.pi * 130.81 * t) * np.exp(-t * 1.4)
    return _stereo_room((left + sub) * attack * 0.72, (right + sub) * attack * 0.72, wet=0.24, tail_s=0.7)


def _synth_riser_soft() -> tuple[np.ndarray, np.ndarray]:
    """Airy filtered-noise swell + rising shimmer that peaks at its end (beat transitions)."""
    dur = 0.6
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    u = t / dur
    rng = np.random.default_rng(303)
    sweep = 600.0 + 5200.0 * u**2
    noise_l = _svf_bandpass(rng.normal(0.0, 1.0, n), sweep, q=2.4)
    noise_r = _svf_bandpass(rng.normal(0.0, 1.0, n), sweep * 1.04, q=2.4)
    env = u**2.2 * (1.0 - np.exp(-(1.0 - u) * 60.0))
    shimmer_f = 880.0 * 2.0 ** u
    phase = 2.0 * np.pi * np.cumsum(shimmer_f) / SAMPLE_RATE
    shimmer = 0.08 * np.sin(phase) * env
    # Slow auto-pan sweep left -> right for motion across the stereo field
    pan_l = np.cos(u * np.pi / 2.0) * 0.5 + 0.5
    pan_r = np.sin(u * np.pi / 2.0) * 0.5 + 0.5
    left = (noise_l * 0.30 * env + shimmer) * pan_l
    right = (noise_r * 0.30 * env + shimmer) * pan_r
    return left, right


_GENERATORS = {
    "whoosh_soft": _synth_whoosh_soft,
    "pop_marimba": _synth_pop_marimba,
    "tick_soft": _synth_tick_soft,
    "chime_reveal": _synth_chime_reveal,
    "chord_epiphany": _synth_chord_epiphany,
    "riser_soft": _synth_riser_soft,
}

SFX_VERSION = "v2"  # bump to invalidate cached renders when synthesis changes


class SfxLibrary:
    """Caches and returns file paths for studio-synthesized sound effects."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_all(self) -> dict[str, Path]:
        return {name: self.get(name) for name in _GENERATORS}

    def get(self, name: str) -> Path:
        if name not in _GENERATORS:
            raise ValueError(f"Unknown SFX '{name}'. Available: {list(_GENERATORS)}")
        target = self.cache_dir / f"{name}_{SFX_VERSION}.wav"
        if not target.exists():
            left, right = _GENERATORS[name]()
            _write_stereo_wav(target, left, right)
        return target
