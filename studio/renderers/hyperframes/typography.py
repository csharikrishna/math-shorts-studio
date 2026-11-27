"""Offline OFL font staging and symbolic math text formatting for HyperFrames."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from studio.design.fonts import FONT_DIR, font_face_css


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
        (r"\approx", "≈"),
        (r"\le", "≤"),
        (r"\ge", "≥"),
        (r"\neq", "≠"),
        (r"\iff", "⇔"),
        (r"\Leftrightarrow", "⇔"),
        (r"\Rightarrow", "⇒"),
        (r"\rightarrow", "→"),
        (r"\to", "→"),
        (r"\in", "∈"),
        (r"\infty", "∞"),
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
