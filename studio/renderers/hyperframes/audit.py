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
    const rects = [];
    for (const node of elems) {
      const r = node.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const eid = node.getAttribute("data-hf-elem");
      const role = node.getAttribute("data-hf-role") || "";
      if (r.left < -1 || r.right > DATA.width + 1) issues.push(`${eid}: horizontal overflow`);
      if (r.top < 4 && !node.classList.contains("hf-eq-card")) issues.push(`${eid}: clipped at top margin`);
      if (DATA.caption_style !== "off" && r.bottom > capRect.top + 4 && !node.classList.contains("hf-eq-card")) {
        issues.push(`${eid}: overlaps caption band`);
      }
      if (hdrEl && r.top < hdrBottom - 2 && !node.classList.contains("hf-eq-card")) {
        issues.push(`${eid}: overlaps header band`);
      }
      if (role === "panel" || role === "bar_chart" || node.classList.contains("hf-matrix") || node.classList.contains("hf-table")) {
        rects.push({ eid, r });
      }
    }
    for (let i = 0; i < rects.length; i++) {
      for (let j = i + 1; j < rects.length; j++) {
        const a = rects[i].r, b = rects[j].r;
        const ix = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
        const iy = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
        if (ix > 10 && iy > 10) {
          issues.push(`${rects[i].eid} overlaps ${rects[j].eid} (${Math.round(ix)}x${Math.round(iy)}px)`);
        }
      }
    }
    return issues;
  };
"""
