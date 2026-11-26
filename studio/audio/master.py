"""FFmpeg audio-video mastering pipeline: sidechain ducking, mixing, and broadcast normalization."""

from pathlib import Path
import subprocess
import os
from studio.audio.tts import find_ffmpeg


# ---------------------------------------------------------------------------
# Encoder selection
# ---------------------------------------------------------------------------
# ManimGX renders frames on the GPU (wgpu/Rust), but the final H.264 encode
# is done by FFmpeg, which defaults to libx264 (CPU).  Using NVENC offloads
# the encode to the GTX 1650's NVENC hardware:
#   - 5-10× faster mastering step
#   - Identical visual quality at the same CRF-equivalent QP
#   - Frees the CPU for TTS / audio synthesis
#
# File size is controlled by CRF / QP, NOT by whether the GPU is used.
# A high CRF (quality-agnostic) value is what bloats size; we default to
# a sensible CRF=23 for libx264 and QP=26 for NVENC (visually equivalent
# for clean vector animation content like math graphs).
#
# Set STUDIO_ENCODER=cpu in env to force libx264, e.g. for CI.
# ---------------------------------------------------------------------------

def _pick_encoder() -> tuple[str, list[str]]:
    """Return (encoder_name, codec_flags). Prefers NVENC, falls back to libx264."""
    prefer = os.environ.get("STUDIO_ENCODER", "nvenc").lower()
    if prefer == "cpu":
        return "libx264", _libx264_flags()
    # Quick NVENC probe: NVENC requires frames >= 145x49, so probe at 256x256
    ffmpeg_bin = find_ffmpeg()
    probe = subprocess.run(
        [ffmpeg_bin, "-y", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
         "-c:v", "h264_nvenc", "-f", "null", "-"],
        capture_output=True,
    )
    if probe.returncode == 0:
        return "h264_nvenc", _nvenc_flags()
    return "libx264", _libx264_flags()


def _libx264_flags() -> list[str]:
    """CPU libx264: CRF 23 at faster preset — sharpest compression (~3 MB for 20s 1080p60)."""
    return ["-c:v", "libx264", "-preset", "faster", "-crf", "23", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level:v", "4.2"]


def _nvenc_flags() -> list[str]:
    """NVENC H.264: CQ 31 + Spatial AQ + 2.5M cap — GPU-accelerated (~7s) and compact (~3-5 MB)."""
    return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "31",
            "-spatial-aq", "1", "-b:v", "0", "-maxrate", "2.5M", "-bufsize", "5M",
            "-pix_fmt", "yuv420p", "-profile:v", "high"]


def master_video(
    raw_video: Path,
    voice_wav: Path,
    bgm_wav: Path,
    output_mp4: Path,
    sfx_gain: float = 0.85,
    voice_gain: float = 1.15,
    bgm_gain: float = 0.24,
    look: str = "none",
    lut_path: Path | None = None,
) -> Path:
    """Mix GPU-rendered video+SFX with ducked BGM and voiceover into an H.264/AAC MP4."""
    ffmpeg_bin = find_ffmpeg()
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    encoder_name, default_codec_flags = _pick_encoder()

    # Filtergraph:
    # 1. Split voice so it drives sidechaincompress on BGM (auto-ducking when narrator speaks).
    # 2. Mix SFX stream (0:a), boosted voice, and ducked BGM.
    # 3. Two-pass EBU R128 loudness normalization (-14.0 LUFS, -1.0 dBTP, 48 kHz).
    audio_mix = (
        f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={voice_gain},asplit=2[voice_mix][voice_sc];"
        f"[2:a]aresample=48000,aformat=channel_layouts=stereo,volume={bgm_gain}[bgm_pre];"
        f"[bgm_pre]extrastereo=m=1.15[bgm_wide];"
        f"[bgm_wide][voice_sc]sidechaincompress=threshold=0.03:ratio=4.5:attack=40:release=320[bgm_ducked];"
        f"[0:a]aresample=48000,aformat=channel_layouts=stereo,volume={sfx_gain},extrastereo=m=1.25[sfx];"
        f"[sfx][voice_mix][bgm_ducked]amix=inputs=3:duration=first:normalize=0,alimiter=limit=0.92"
    )

    # Pass 1: measure integrated loudness, true peak, LRA, threshold, target offset
    import json as _json
    import re as _re
    pass1 = subprocess.run(
        [
            ffmpeg_bin, "-y",
            "-i", str(raw_video), "-i", str(voice_wav), "-i", str(bgm_wav),
            "-filter_complex", f"{audio_mix},loudnorm=I=-14.0:TP=-1.0:LRA=7.0:print_format=json[m_a]",
            "-map", "[m_a]", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    ln_filter = "loudnorm=I=-14.0:TP=-1.0:LRA=7.0"
    m_json = _re.search(r"\{\s*\"input_i\"\s*:.*?\}", pass1.stderr, _re.DOTALL)
    if pass1.returncode == 0 and m_json:
        stats = _json.loads(m_json.group(0))
        ln_filter = (
            f"loudnorm=I=-14.0:TP=-1.0:LRA=7.0:"
            f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
            f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}:linear=true"
        )

    filter_complex = f"{audio_mix},{ln_filter}[master_a]"

    video_chain = cinematic_video_filter(look, lut_path)
    if video_chain:
        filter_complex += f";[0:v]{video_chain}[master_v]"
        video_map = "[master_v]"
        video_codec = default_codec_flags
    else:
        # No visual filter: stream-copy ManimGX's already-compact CRF 23 GPU render (< 0.2s)
        video_map = "0:v"
        video_codec = ["-c:v", "copy"]
        encoder_name = "copy"

    print(f"      encoder: {encoder_name}")

    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-i", str(raw_video),
            "-i", str(voice_wav),
            "-i", str(bgm_wav),
            "-filter_complex", filter_complex,
            "-map", video_map,
            "-map", "[master_a]",
            *video_codec,
            "-c:a", "aac",
            "-b:a", "160k",
            "-movflags", "+faststart",
            str(output_mp4),
        ],
        check=True,
        capture_output=True,
    )
    # Report final file size
    size_mb = Path(output_mp4).stat().st_size / 1_048_576
    print(f"      output: {output_mp4.name}  ({size_mb:.1f} MB)")
    return output_mp4


# Grade presets: fast YUV-native adjustments (no RGB roundtrip, no random film-grain bloat).
GRADES = {
    "cinematic": "eq=contrast=1.06:saturation=1.08:brightness=-0.02:gamma_b=1.04:gamma_r=0.98",
    "warm": "eq=contrast=1.04:saturation=1.06:gamma_r=1.05:gamma_b=0.95",
    "cool": "eq=contrast=1.04:saturation=1.04:gamma_b=1.05:gamma_r=0.96",
}
LOOKS = ("none", *GRADES)


def cinematic_video_filter(look: str, lut_path: Path | None = None, grain: bool = False) -> str:
    """FFmpeg video chain: YUV-native grade (or .cube LUT), with optional subtle grain."""
    if look == "none" and lut_path is None:
        return ""
    if look not in LOOKS:
        raise ValueError(f"Unknown look '{look}'. Options: {list(LOOKS)}")
    steps: list[str] = []
    if lut_path is not None:
        lut = str(Path(lut_path).resolve()).replace("\\", "/").replace(":", r"\:")
        steps.append(f"lut3d=file='{lut}'")
    if look != "none":
        steps.append(GRADES[look])
        if grain:
            steps.append("noise=c0s=3:c0f=t+u")
    steps.append("format=yuv420p")
    return ",".join(steps)
