"""Procedural ambient background music generator scaled to exact timeline length."""

from pathlib import Path
import wave
import numpy as np

SAMPLE_RATE = 48000


def generate_ambient_bgm(output_path: Path, duration: float, bpm: float = 84.0) -> Path:
    """Synthesize a warm, stereo ambient pad progression matching exact duration."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_dur = max(4.0, duration + 0.5)
    n = int(SAMPLE_RATE * total_dur)
    t = np.linspace(0.0, total_dur, n, endpoint=False)

    # Emotional cinematic math chord progression (Am9 -> Fmaj7 -> Cmaj9 -> G6)
    chords = [
        [110.00, 164.81, 246.94, 329.63],  # Am9
        [87.31, 130.81, 220.00, 329.63],   # Fmaj7
        [130.81, 196.00, 293.66, 392.00],  # Cmaj9
        [98.00, 146.83, 246.94, 329.63],   # G6
    ]

    bar_seconds = (60.0 / bpm) * 4.0 * 2.0  # 2 bars per chord (~5.71s at 84 BPM)
    left = np.zeros(n)
    right = np.zeros(n)

    for chord_idx in range(int(np.ceil(total_dur / bar_seconds))):
        start_s = chord_idx * bar_seconds
        end_s = min(total_dur, (chord_idx + 1) * bar_seconds)
        i0 = int(start_s * SAMPLE_RATE)
        i1 = int(end_s * SAMPLE_RATE)
        if i1 <= i0:
            continue
        tc = t[i0:i1] - start_s
        local_dur = end_s - start_s
        freqs = chords[chord_idx % len(chords)]

        # Soft crossfaded pad envelope
        fade = min(0.8, local_dur * 0.25)
        env = np.ones_like(tc)
        fade_n = int(fade * SAMPLE_RATE)
        if fade_n > 0 and fade_n * 2 < len(env):
            env[:fade_n] = np.sin(np.linspace(0, np.pi / 2, fade_n)) ** 2
            env[-fade_n:] = np.sin(np.linspace(np.pi / 2, 0, fade_n)) ** 2

        pad_l = np.zeros_like(tc)
        pad_r = np.zeros_like(tc)
        for j, f in enumerate(freqs):
            detune = 1.0 + (0.002 if j % 2 == 0 else -0.002)
            # Warm fundamental + subtle 2nd harmonic
            pad_l += 0.24 * np.sin(2.0 * np.pi * f * tc) + 0.05 * np.sin(4.0 * np.pi * f * tc)
            pad_r += 0.24 * np.sin(2.0 * np.pi * (f * detune) * tc) + 0.05 * np.sin(4.0 * np.pi * (f * detune) * tc)

        left[i0:i1] += pad_l * env
        right[i0:i1] += pad_r * env

        # Ping-pong pluck arpeggio on 8th notes (chord tones up an octave), alternating L/R
        eighth = 60.0 / bpm / 2.0
        arp_notes = [f * 2.0 for f in freqs] + [freqs[1] * 4.0, freqs[2] * 2.0]
        n_steps = int(local_dur / eighth)
        pluck_len = int(min(eighth * 3.0, 0.9) * SAMPLE_RATE)
        tp = np.arange(pluck_len) / SAMPLE_RATE
        for step in range(n_steps):
            s0 = i0 + int(step * eighth * SAMPLE_RATE)
            s1 = min(i1, s0 + pluck_len)
            if s1 <= s0:
                continue
            f = arp_notes[(step * 3) % len(arp_notes)]
            tone = (np.sin(2.0 * np.pi * f * tp) + 0.25 * np.sin(4.0 * np.pi * f * tp)) * np.exp(-tp * 7.0)
            tone *= 0.045 * (1.0 - np.exp(-tp * 400.0))
            seg = tone[: s1 - s0]
            pos = 0.35 if step % 2 == 0 else -0.35
            left[s0:s1] += seg * np.cos((pos + 1.0) * np.pi / 4.0) * 1.41
            right[s0:s1] += seg * np.sin((pos + 1.0) * np.pi / 4.0) * 1.41

    # Slow stereo chorus/LFO: decorrelated amplitude drift widens the pad image
    lfo_l = 1.0 + 0.10 * np.sin(2.0 * np.pi * 0.11 * t)
    lfo_r = 1.0 + 0.10 * np.sin(2.0 * np.pi * 0.11 * t + np.pi * 0.6)
    left *= lfo_l
    right *= lfo_r

    # Airy filtered-noise bed (one-pole high-passed, very low level) for "room" presence
    rng = np.random.default_rng(84)
    for chan in (left, right):
        noise = rng.normal(0.0, 1.0, n)
        hp = noise - np.convolve(noise, np.ones(24) / 24.0, mode="same")
        chan += 0.012 * hp * (0.7 + 0.3 * np.sin(2.0 * np.pi * 0.05 * t))

    # Global smooth fade-in / fade-out
    g_fade = int(min(1.2, total_dur * 0.15) * SAMPLE_RATE)
    if g_fade > 0:
        left[:g_fade] *= np.linspace(0.0, 1.0, g_fade)
        right[:g_fade] *= np.linspace(0.0, 1.0, g_fade)
        left[-g_fade:] *= np.linspace(1.0, 0.0, g_fade)
        right[-g_fade:] *= np.linspace(1.0, 0.0, g_fade)

    peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-6)
    stereo = np.stack([left, right], axis=1) * (0.28 / peak)
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return output_path
