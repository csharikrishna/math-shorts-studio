"""Multi-layer automated quality evaluation: safe AST-to-SymPy math QA, FFmpeg audio QA, and PIL visual QA."""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import sympy as sp

from studio.audio.tts import find_ffmpeg

QAStatus = Literal["passed", "failed", "not_evaluated", "human_review_required"]

_SYMPY_FUNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
}
_SYMPY_CONSTS = {"pi": sp.pi, "e": sp.E, "E": sp.E}
_MAX_AST_NODES = 128


def safe_sympy_from_expr(source: str, allowed_vars: tuple[str, ...] = ("x",)) -> sp.Expr:
    """Safely convert a restricted math string into a SymPy Expr via Python AST (never sympify/eval)."""
    tree = ast.parse(source.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError(f"Expression exceeds {_MAX_AST_NODES} AST nodes.")

    def _conv(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return _conv(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return sp.nsimplify(node.value) if isinstance(node.value, int) else sp.Float(node.value)
        if isinstance(node, ast.Name):
            if node.id in allowed_vars:
                return sp.Symbol(node.id, real=True)
            if node.id in _SYMPY_CONSTS:
                return _SYMPY_CONSTS[node.id]
            raise ValueError(f"Disallowed symbol: {node.id!r}")
        if isinstance(node, ast.UnaryOp):
            val = _conv(node.operand)
            if isinstance(node.op, ast.UAdd):
                return val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError(f"Disallowed unary op: {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            left, right = _conv(node.left), _conv(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError(f"Disallowed binary op: {type(node.op).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _SYMPY_FUNCS.get(node.func.id)
            if fn is None or node.keywords or len(node.args) != 1:
                raise ValueError(f"Disallowed function call: {node.func.id!r}")
            return fn(_conv(node.args[0]))
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    return _conv(tree)


@dataclass
class QACheck:
    name: str
    status: QAStatus
    detail: str
    severity: str = "error"  # error | warning

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass
class QualityReport:
    checks: list[QACheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if not self.checks:
            return False  # Empty report must never silently pass
        evaluated = [c for c in self.checks if c.status != "not_evaluated"]
        if not evaluated:
            return False
        return all(c.status == "passed" or c.severity != "error" for c in evaluated)

    def summary(self) -> str:
        ok = sum(1 for c in self.checks if c.status == "passed")
        return f"{ok}/{len(self.checks)} checks passed (overall={'PASS' if self.passed else 'FAIL'})"


def verify_symbolic_derivative(expr_str: str, deriv_str: str, var: str = "x") -> QACheck:
    """Verify symbolically with SymPy (via safe AST) that d/d(var) expr == deriv."""
    try:
        sym = sp.Symbol(var, real=True)
        e1 = safe_sympy_from_expr(expr_str, (var,))
        e2 = safe_sympy_from_expr(deriv_str, (var,))
        diff = sp.simplify(sp.diff(e1, sym) - e2)
        ok = diff == 0
        return QACheck("sympy_derivative", "passed" if ok else "failed", f"d/d{var}({expr_str}) - ({deriv_str}) = {diff}")
    except Exception as exc:
        return QACheck("sympy_derivative", "failed", f"Math QA error: {exc}")


def verify_audio_loudness(video_path: Path, min_lufs: float = -16.5, max_lufs: float = -13.0) -> QACheck:
    """Measure actual integrated loudness and true peak via FFmpeg ebur128."""
    ff = find_ffmpeg()
    try:
        proc = subprocess.run(
            [ff, "-hide_banner", "-nostats", "-i", str(video_path), "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return QACheck("audio_loudness", "failed", f"FFmpeg ebur128 failed: {exc}")
    if proc.returncode != 0:
        return QACheck("audio_loudness", "failed", f"FFmpeg ebur128 exited {proc.returncode}")
    matches = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", proc.stderr)
    peak_matches = re.findall(r"Peak:\s*(-?\d+(?:\.\d+)?)\s*dBFS", proc.stderr)
    if not matches:
        return QACheck("audio_loudness", "failed", "Could not parse EBU R128 loudness.")
    lufs = float(matches[-1])
    peak = float(peak_matches[-1]) if peak_matches else 0.0
    ok = (min_lufs <= lufs <= max_lufs) and (peak <= -1.0)
    return QACheck(
        "audio_loudness",
        "passed" if ok else "failed",
        f"Integrated={lufs:.1f} LUFS (target {min_lufs}..{max_lufs}), TruePeak={peak:.1f} dBFS (<= -1.0)",
    )


def verify_storyboard_image(image_path: Path) -> QACheck:
    """Verify storyboard/frame is non-blank and has healthy contrast."""
    if not image_path.exists():
        return QACheck("visual_storyboard", "not_evaluated", f"Missing {image_path}", severity="warning")
    arr = np.asarray(Image.open(image_path).convert("L"), dtype=float)
    std = float(np.std(arr))
    ok = std > 8.0
    return QACheck("visual_storyboard", "passed" if ok else "failed", f"{image_path.name} luminance std={std:.1f}")
