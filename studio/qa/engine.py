"""Multi-layer automated quality evaluation: SymPy math QA, FFmpeg audio QA, and PIL visual QA."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
import sympy as sp

from studio.audio.tts import find_ffmpeg


@dataclass
class QACheck:
    name: str
    passed: bool
    detail: str
    severity: str = "error"  # error | warning


@dataclass
class QualityReport:
    checks: list[QACheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed or c.severity != "error" for c in self.checks)

    def summary(self) -> str:
        ok = sum(1 for c in self.checks if c.passed)
        return f"{ok}/{len(self.checks)} checks passed (overall={'PASS' if self.passed else 'FAIL'})"


def verify_symbolic_derivative(expr_str: str, deriv_str: str, var: str = "x") -> QACheck:
    """Verify symbolically with SymPy that d/d(var) expr == deriv."""
    try:
        sym = sp.Symbol(var)
        e1 = sp.sympify(expr_str)
        e2 = sp.sympify(deriv_str)
        diff = sp.simplify(sp.diff(e1, sym) - e2)
        ok = diff == 0
        return QACheck("sympy_derivative", ok, f"d/d{var}({expr_str}) - ({deriv_str}) = {diff}")
    except Exception as exc:
        return QACheck("sympy_derivative", False, f"SymPy error: {exc}")


def verify_audio_loudness(video_path: Path, min_lufs: float = -16.5, max_lufs: float = -13.0) -> QACheck:
    """Measure actual integrated loudness via FFmpeg ebur128."""
    ff = find_ffmpeg()
    proc = subprocess.run(
        [ff, "-hide_banner", "-nostats", "-i", str(video_path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    matches = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", proc.stderr)
    if not matches:
        return QACheck("audio_loudness", False, "Could not parse EBU R128 loudness.")
    lufs = float(matches[-1])
    ok = min_lufs <= lufs <= max_lufs
    return QACheck("audio_loudness", ok, f"Integrated loudness = {lufs:.1f} LUFS (target {min_lufs}..{max_lufs})")


def verify_storyboard_image(image_path: Path) -> QACheck:
    """Verify storyboard/frame is non-blank and has healthy contrast."""
    if not image_path.exists():
        return QACheck("visual_storyboard", False, f"Missing {image_path}")
    arr = np.asarray(Image.open(image_path).convert("L"), dtype=float)
    std = float(np.std(arr))
    ok = std > 8.0
    return QACheck("visual_storyboard", ok, f"{image_path.name} luminance std={std:.1f}")
