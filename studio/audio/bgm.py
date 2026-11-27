"""Procedural ambient background music generator: smooth overlapping pads + soothing felt/bell motifs."""

from __future__ import annotations

from pathlib import Path
import wave
import numpy as np

SAMPLE_RATE = 48000


def _soft_stereo_reverb(
    left: np.ndarray,
    right: np.ndarray,
    *,
    wet: float = 0.34,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply multi-tap low-passed stereo diffusion for warm, soothing bell and pad bloom."""
    n = len(left)
    out_l = left.copy()
    out_r = right.copy()

    # Warm damped copy so echoes soften smoothly without metallic brightness
    kernel = np.array([0.2, 0.3, 0.3, 0.2], dtype=np.float32)
    warm_l = np.convolve(left, kernel, mode="same")
    warm_r = np.convolve(right, kernel, mode="same")

    taps_l = ((0.037, 0.42), (0.067, 0.31), (0.113, 0.22), (0.173, 0.14), (0.241, 0.08))
    taps_r = ((0.043, 0.42), (0.079, 0.31), (0.127, 0.22), (0.191, 0.14), (0.263, 0.08))
    for taps, src_a, src_b, dst in (
        (taps_l, warm_r, warm_l, out_l),
        (taps_r, warm_l, warm_r, out_r),
    ):
        for delay_s, gain in taps:
            d = int(delay_s * SAMPLE_RATE)
            if 0 < d < n:
                dst[d:] += wet * gain * (0.72 * src_a[: n - d] + 0.28 * src_b[: n - d])
    return out_l, out_r


def _add_bell_note(
    left: np.ndarray,
    right: np.ndarray,
    *,
    onset_s: float,
    freq: float,
    amp: float,
    decay_s: float,
    pan: float,
    bright: float = 0.14,
) -> None:
    """Synthesize a gentle felt-mallet bell / celesta / soft wooden pluck at onset_s."""
    sr = SAMPLE_RATE
    note_dur = min(decay_s * 3.4, max(0.35, (len(left) / sr) - onset_s))
    i0 = int(onset_s * sr)
    cnt = int(note_dur * sr)
    if i0 < 0 or cnt <= 64 or i0 >= len(left):
        return
    i1 = min(len(left), i0 + cnt)
    seg_n = i1 - i0
    tc = np.arange(seg_n, dtype=np.float32) / float(sr)

    # Smooth mallet attack (~8 ms raised cosine) eliminating sharp transient clicks
    att_n = min(seg_n, int(0.009 * sr))
    env = np.exp(-tc / max(0.08, decay_s)).astype(np.float32)
    if att_n > 1:
        env[:att_n] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(att_n, dtype=np.float32) / float(att_n)))

    # Tail taper to guarantee zero discontinuity at segment edge
    tail_n = min(seg_n, int(0.03 * sr))
    if tail_n > 1:
        env[-tail_n:] *= np.linspace(1.0, 0.0, tail_n, dtype=np.float32)

    two_pi_t = (2.0 * np.pi * tc).astype(np.float32)
    harm2_env = np.exp(-tc / max(0.05, decay_s * 0.55)).astype(np.float32)
    harm3_env = np.exp(-tc / max(0.03, decay_s * 0.28)).astype(np.float32)

    tone = (
        np.sin(two_pi_t * freq) * env
        + (bright * np.sin(two_pi_t * (freq * 2.0)) * env * harm2_env)
        + (bright * 0.35 * np.sin(two_pi_t * (freq * 3.0)) * env * harm3_env)
    ) * float(amp)

    angle = (float(np.clip(pan, -0.85, 0.85)) + 1.0) * (np.pi * 0.25)
    left[i0:i1] += tone * float(np.cos(angle))
    right[i0:i1] += tone * float(np.sin(angle))


def generate_ambient_bgm(output_path: Path, duration: float, bpm: float = 72.0) -> Path:
    """Synthesize a silky, soothing stereo ambient score with warm pads, soft bells, and gentle plucks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_dur = max(4.0, float(duration) + 0.6)
    n = int(SAMPLE_RATE * total_dur)
    t = np.arange(n, dtype=np.float32) / float(SAMPLE_RATE)

    # Warm, meditative Am9 -> Fmaj9 -> Cmaj9 -> G6 progression with gentle bell voice-leading
    # Frequencies chosen cleanly below & around middle C + sparkling 5th-octave pentatonic bells
    progression = [
        {
            "pad": [110.00, 164.81, 220.00, 246.94, 329.63],   # A2, E3, A3, B3, E4
            "bells": [523.25, 659.25, 783.99, 659.25],         # C5, E5, G5, E5
            "pulses": [261.63, 329.63, 392.00, 329.63],        # C4, E4, G4, E4
        },
        {
            "pad": [87.31, 130.81, 174.61, 220.00, 329.63],    # F2, C3, F3, A3, E4
            "bells": [440.00, 523.25, 659.25, 587.33],         # A4, C5, E5, D5
            "pulses": [220.00, 261.63, 329.63, 293.66],        # A3, C4, E4, D4
        },
        {
            "pad": [130.81, 196.00, 246.94, 293.66, 392.00],   # C3, G3, B3, D4, G4
            "bells": [587.33, 659.25, 783.99, 987.77],         # D5, E5, G5, B5
            "pulses": [293.66, 329.63, 392.00, 329.63],        # D4, E4, G4, E4
        },
        {
            "pad": [98.00, 146.83, 196.00, 246.94, 329.63],    # G2, D3, G3, B3, E4
            "bells": [493.88, 587.33, 659.25, 587.33],         # B4, D5, E5, D5
            "pulses": [246.94, 293.66, 329.63, 293.66],        # B3, D4, E4, D4
        },
    ]

    beat_s = 60.0 / max(45.0, float(bpm))
    bar_s = beat_s * 8.0  # 2 bars per chord (~6.67s at 72 BPM)
    overlap_s = min(1.8, bar_s * 0.28)  # seamless raised-cosine crossfade across chord changes

    pad_l = np.zeros(n, dtype=np.float32)
    pad_r = np.zeros(n, dtype=np.float32)
    bell_l = np.zeros(n, dtype=np.float32)
    bell_r = np.zeros(n, dtype=np.float32)

    num_chords = int(np.ceil(total_dur / bar_s))
    for c_idx in range(num_chords):
        chord = progression[c_idx % len(progression)]
        c_start = max(0.0, c_idx * bar_s - (overlap_s * 0.5 if c_idx > 0 else 0.0))
        c_end = min(total_dur, (c_idx + 1) * bar_s + overlap_s * 0.5)
        i0 = int(c_start * SAMPLE_RATE)
        i1 = int(c_end * SAMPLE_RATE)
        if i1 <= i0 + 16:
            continue

        tc = t[i0:i1] - float(c_start)
        loc_n = len(tc)
        env = np.ones(loc_n, dtype=np.float32)
        fade_in_n = min(loc_n // 2, int(overlap_s * SAMPLE_RATE))
        fade_out_n = min(loc_n // 2, int(overlap_s * SAMPLE_RATE))
        if fade_in_n > 1:
            env[:fade_in_n] = np.sin(np.linspace(0.0, np.pi * 0.5, fade_in_n, dtype=np.float32)) ** 2
        if fade_out_n > 1:
            env[-fade_out_n:] = np.sin(np.linspace(np.pi * 0.5, 0.0, fade_out_n, dtype=np.float32)) ** 2

        acc_l = np.zeros(loc_n, dtype=np.float32)
        acc_r = np.zeros(loc_n, dtype=np.float32)
        for v_idx, freq in enumerate(chord["pad"]):
            # Warm subtle analog chorus detuning (+/- 0.11%) with zero harsh upper overtones
            detune = 1.0 + (0.0011 if v_idx % 2 == 0 else -0.0011)
            weight = 0.24 if v_idx == 0 else (0.19 if v_idx <= 2 else 0.13)
            w_t = (2.0 * np.pi * tc).astype(np.float32)
            acc_l += weight * np.sin(w_t * float(freq)) + (weight * 0.12) * np.sin(w_t * float(freq * 2.0))
            acc_r += weight * np.sin(w_t * float(freq * detune)) + (weight * 0.12) * np.sin(w_t * float(freq * detune * 2.0))

        pad_l[i0:i1] += acc_l * env
        pad_r[i0:i1] += acc_r * env

        # Gentle bell & soft muted wood/text pulse events paced across the 8 beats of this chord
        chord_origin = c_idx * bar_s
        bell_slots = (0.0, 2.5 * beat_s, 4.5 * beat_s, 6.5 * beat_s)
        pulse_slots = (1.0 * beat_s, 3.0 * beat_s, 5.0 * beat_s, 7.0 * beat_s)
        pans = (-0.36, 0.30, -0.22, 0.42)

        for k, rel_t in enumerate(bell_slots):
            ons = chord_origin + rel_t
            if ons < total_dur - 0.5:
                _add_bell_note(
                    bell_l,
                    bell_r,
                    onset_s=ons,
                    freq=chord["bells"][k % len(chord["bells"])],
                    amp=0.14 if k % 2 == 0 else 0.10,
                    decay_s=1.85 if k == 0 else 1.45,
                    pan=pans[k],
                    bright=0.16,
                )

        for k, rel_t in enumerate(pulse_slots):
            ons = chord_origin + rel_t
            if ons < total_dur - 0.3:
                _add_bell_note(
                    bell_l,
                    bell_r,
                    onset_s=ons,
                    freq=chord["pulses"][k % len(chord["pulses"])],
                    amp=0.075,
                    decay_s=0.38,
                    pan=-pans[k] * 0.7,
                    bright=0.06,
                )

    # Diffuse bell/pulse layer through soft stereo room reverb, then blend seamlessly onto pad
    bell_l, bell_r = _soft_stereo_reverb(bell_l, bell_r, wet=0.36)

    # Very slow, serene breath LFO (0.06 Hz) across channels
    lfo_l = 1.0 + 0.06 * np.sin(2.0 * np.pi * 0.06 * t)
    lfo_r = 1.0 + 0.06 * np.sin(2.0 * np.pi * 0.06 * t + np.pi * 0.55)
    left = (pad_l * 0.72 + bell_l * 0.65) * lfo_l
    right = (pad_r * 0.72 + bell_r * 0.65) * lfo_r

    # Global smooth fade-in / fade-out (no harsh start/stop)
    g_in = int(min(1.8, total_dur * 0.16) * SAMPLE_RATE)
    g_out = int(min(2.2, total_dur * 0.20) * SAMPLE_RATE)
    if g_in > 1:
        in_curve = np.sin(np.linspace(0.0, np.pi * 0.5, g_in, dtype=np.float32)) ** 2
        left[:g_in] *= in_curve
        right[:g_in] *= in_curve
    if g_out > 1:
        out_curve = np.sin(np.linspace(np.pi * 0.5, 0.0, g_out, dtype=np.float32)) ** 2
        left[-g_out:] *= out_curve
        right[-g_out:] *= out_curve

    peak = float(max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-6))
    stereo = np.stack([left, right], axis=1) * (0.32 / peak)
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return output_path
