"""Offline OFL font staging and symbolic math text formatting for HyperFrames."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from studio.ai.env import PROJECT_ROOT
from studio.design.fonts import FONT_DIR, font_face_css

KATEX_DIR = PROJECT_ROOT / "assets" / "vendor" / "katex"


def verify_vendored_katex() -> list[str]:
    """Verify vendored offline KaTeX bundle matches KATEX_MANIFEST.json SHA-256 digests."""
    manifest_p = KATEX_DIR / "KATEX_MANIFEST.json"
    if not manifest_p.exists():
        return [f"Missing {manifest_p}"]
    data = json.loads(manifest_p.read_text(encoding="utf-8"))
    errs: list[str] = []
    for rel, expected in (data.get("files") or {}).items():
        fp = KATEX_DIR / rel
        if not fp.exists():
            errs.append(f"Missing vendored KaTeX file: {rel}")
            continue
        got = hashlib.sha256(fp.read_bytes()).hexdigest()
        if got != expected:
            errs.append(f"SHA mismatch on {rel}: {got[:12]} != {expected[:12]}")
    return errs


def stage_katex_to_site(site_dir: Path) -> bool:
    """Copy vendored offline KaTeX JS/CSS/woff2 assets into site_dir/vendor/katex."""
    if not (KATEX_DIR / "katex.min.js").exists():
        return False
    dst = site_dir / "vendor" / "katex"
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("katex.min.js", "katex.min.css", "LICENSE"):
        src_f = KATEX_DIR / fname
        if src_f.exists():
            tgt_f = dst / fname
            if not tgt_f.exists() or tgt_f.stat().st_size != src_f.stat().st_size:
                shutil.copy2(src_f, tgt_f)
    src_fonts = KATEX_DIR / "fonts"
    if src_fonts.exists():
        dst_fonts = dst / "fonts"
        dst_fonts.mkdir(parents=True, exist_ok=True)
        for wf in src_fonts.glob("*.woff2"):
            tgt_f = dst_fonts / wf.name
            if not tgt_f.exists() or tgt_f.stat().st_size != wf.stat().st_size:
                shutil.copy2(wf, tgt_f)
    return True


def stage_fonts_to_site(site_dir: Path) -> None:
    """Copy pinned offline OFL font binaries into site_dir/fonts for loopback Chromium loading."""
    fonts_dst = site_dir / "fonts"
    if not FONT_DIR.exists():
        return
    fonts_dst.mkdir(parents=True, exist_ok=True)
    for fitem in FONT_DIR.iterdir():
        if fitem.is_file() and fitem.suffix.lower() in (".ttf", ".otf"):
            tgt_f = fonts_dst / fitem.name
            if not tgt_f.exists() or tgt_f.stat().st_size != fitem.stat().st_size:
                shutil.copy2(fitem, tgt_f)


def build_font_css(url_prefix: str = "/fonts/") -> str:
    return font_face_css(url_prefix=url_prefix, embed_base64=False)


def _pretty_math_label(tex: str) -> str:
    """Convert basic LaTeX to crisp typographic Unicode strings offline."""
    out = re.sub(r"\\(?:mathbf|mathrm|text|vec|operatorname|mathcal)\{([^{}]*)\}", r"\1", tex)
    out = re.sub(r"\\(?:tfrac|dfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", out)
    out = re.sub(r"\\overset\{([^{}]+)\}\{([^{}]+)\}", r"\2 (\1)", out)
    repls = [
        (r"\;", " "),
        (r"\,", " "),
        (r"\quad", "  "),
        (r"\qquad", "   "),
        (r"\!", ""),
        (r"\left", ""),
        (r"\right", ""),
        (r"\det", "det"),
        (r"\tr", "tr"),
        (r"\Re", "Re"),
        (r"\Im", "Im"),
        (r"\pm", "±"),
        (r"\mp", "∓"),
        (r"\pi", "π"),
        (r"\theta", "θ"),
        (r"\lambda", "λ"),
        (r"\alpha", "α"),
        (r"\beta", "β"),
        (r"\sigma", "σ"),
        (r"\mu", "μ"),
        (r"\Delta", "Δ"),
        (r"\delta", "δ"),
        (r"\sum", "Σ"),
        (r"\int", "∫"),
        (r"\cdot", "·"),
        (r"\times", "×"),
        (r"\omega", "ω"),
        (r"\phi", "φ"),
        (r"\psi", "ψ"),
        (r"\approx", "≈"),
        (r"\le", "≤"),
        (r"\ge", "≥"),
        (r"\neq", "≠"),
        (r"\iff", "⇔"),
        (r"\Leftrightarrow", "⇔"),
        (r"\Rightarrow", "⇒"),
        (r"\rightarrow", "→"),
        (r"\to", "→"),
        (r"\infty", "∞"),
        (r"\in", "∈"),
        (r"\sqrt", "√"),
    ]
    for k, v in repls:
        out = out.replace(k, v)
    sub_map = str.maketrans("0123456789ijkn", "₀₁₂₃₄₅₆₇₈₉ᵢⱼₖₙ")
    sup_map = str.maketrans("0123456789+-in", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁱⁿ")
    out = re.sub(r"_\{([0-9ijkn]+)\}", lambda m: m.group(1).translate(sub_map), out)
    out = re.sub(r"_([0-9ijkn])", lambda m: m.group(1).translate(sub_map), out)
    out = re.sub(r"\^\{([0-9+\-in]+)\}", lambda m: m.group(1).translate(sup_map), out)
    out = re.sub(r"\^([0-9])", lambda m: m.group(1).translate(sup_map), out)
    out = re.sub(r"[{}]", "", out)
    out = re.sub(r"\\([A-Za-z]+)", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()
