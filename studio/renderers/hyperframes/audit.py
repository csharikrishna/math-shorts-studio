"""In-browser DOM bounding-box and safe-zone layout audit helper for HyperFrames."""

from __future__ import annotations

AUDIT_JS_SNIPPET = """
  window.__HF_AUDIT_LAYOUT = () => {
    const issues = [];
    const capEl = document.getElementById("caption-band");
    const capRect = capEl ? capEl.getBoundingClientRect() : { top: DATA.height - 20 };
    const hdrEl = document.getElementById("header-band");
    const hdrBottom = hdrEl ? hdrEl.getBoundingClientRect().bottom : Math.max(12, DATA.top_band * 0.4);
    const elems = Array.from(document.querySelectorAll("[data-hf-elem]"));
    for (const node of elems) {
      const r = node.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const eid = node.getAttribute("data-hf-elem");
      if (r.left < -1 || r.right > DATA.width + 1) issues.push(`${eid}: horizontal overflow`);
      if (DATA.caption_style !== "off" && r.bottom > capRect.top + 4 && !node.classList.contains("hf-eq-card")) {
        issues.push(`${eid}: overlaps caption band`);
      }
      if (hdrEl && r.top < hdrBottom - 2 && !node.classList.contains("hf-eq-card")) {
        issues.push(`${eid}: overlaps header band`);
      }
    }
    return issues;
  };
"""
