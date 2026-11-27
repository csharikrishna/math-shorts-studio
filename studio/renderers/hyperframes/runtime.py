"""Deterministic JavaScript timeline & vector/DOM runtime for HyperFrames."""

from __future__ import annotations

from studio.renderers.hyperframes.audit import AUDIT_JS_SNIPPET


def build_runtime_js(payload_json: str) -> str:
    """Return deterministic offline SVG/HTML Motion runtime script block."""
    return f"""
(() => {{
  const DATA = {payload_json};
  const SVG_NS = "http://www.w3.org/2000/svg";
  const layers = document.getElementById("svg-layers");
  const overlay = document.getElementById("html-overlay");
  const spBadge = document.getElementById("speaker-badge");
  const capText = document.getElementById("caption-text");

  function clamp01(t) {{ return t < 0 ? 0 : (t > 1 ? 1 : t); }}
  function easeOutExpo(t) {{ t = clamp01(t); return t >= 1 ? 1 : 1 - Math.pow(2, -10 * t); }}
  function easeInOutCubic(t) {{ t = clamp01(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }}
  function springEval(t, enterS, zeta, stiffness) {{
    if (t <= 0) return 0;
    const wn = Math.sqrt(Math.max(10, stiffness));
    const tau = t * (3.5 / Math.max(0.2, enterS));
    if (zeta < 0.99) {{
      const wd = wn * Math.sqrt(1 - zeta * zeta);
      const env = Math.exp(-zeta * wn * tau);
      return 1 - env * (Math.cos(wd * tau) + (zeta * wn / wd) * Math.sin(wd * tau));
    }}
    return easeOutExpo(t / Math.max(0.15, enterS));
  }}

  let __vtimeMs = 0;
  if (window.performance && window.performance.now) {{
    window.performance.now = () => __vtimeMs;
  }}

  function svgEl(tag, attrs) {{
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    return el;
  }}

  let prevSig = "";

  function activeBeatAt(t) {{
    let activeBeat = DATA.beats[DATA.beats.length - 1] || null;
    let activeIdx = Math.max(0, DATA.beats.length - 1);
    for (let i = 0; i < DATA.beats.length; i++) {{
      if (t >= DATA.beats[i].start && t < DATA.beats[i].end) {{
        activeBeat = DATA.beats[i];
        activeIdx = i;
        break;
      }}
    }}
    return {{ activeBeat, activeIdx }};
  }}

  function activeCueAt(t) {{
    if (!DATA.cues || !DATA.cues.length) return null;
    let chosen = null;
    for (const c of DATA.cues) {{
      if (t + 1e-4 >= c.start && t <= c.end + 0.12) {{
        chosen = c;
      }}
    }}
    return chosen;
  }}

  function computeRenderSig(t) {{
    const enterDur = DATA.motion.enter_s || 0.45;
    const {{ activeBeat, activeIdx }} = activeBeatAt(t);
    const activeCue = activeCueAt(t);
    let capSig = String(activeIdx);
    if (DATA.caption_style !== "off" && activeCue) {{
      const wStates = (activeCue.words || []).map(w => (t >= w.s && t <= w.e ? 1 : (t > w.e ? 2 : 0))).join("");
      capSig += `:${{activeCue.cue_id}}:${{wStates}}`;
    }} else if (activeBeat && activeBeat.words && activeBeat.words.length > 0 && DATA.caption_style !== "off") {{
      let activeWIdx = activeBeat.words.findIndex(w => t >= w.s && t <= w.e);
      if (activeWIdx < 0) {{
        activeWIdx = Math.max(0, activeBeat.words.filter(w => t > w.e).length - 1);
      }}
      const startIdx = Math.floor(activeWIdx / 6) * 6;
      const chunk = activeBeat.words.slice(startIdx, startIdx + 6);
      const wStates = chunk.map(w => (t >= w.s && t <= w.e ? 1 : (t > w.e ? 2 : 0))).join("");
      capSig += `:${{startIdx}}:${{wStates}}`;
    }}
    const parts = [capSig];
    for (const el of DATA.elements) {{
      if (t < el.show_at || t >= el.hide_at) continue;
      const p = clamp01(easeOutExpo((t - el.show_at) / enterDur));
      const qP = Math.round(p * 2000);
      let extra = "";
      if (el.polyline && el.polyline.length > 1) {{
        extra = `:${{Math.max(2, Math.floor(el.polyline.length * p))}}`;
      }} else if (el.kind === "point" && el.orbit && el.orbit.length > 1) {{
        const oIdx = Math.min(el.orbit.length - 1, Math.floor((el.orbit.length - 1) * easeInOutCubic((t - el.show_at) / Math.max(0.5, el.hide_at - el.show_at))));
        const sQ = Math.round(springEval(t - el.show_at, enterDur, 0.72, 180) * 500);
        extra = `:${{oIdx}}:${{sQ}}`;
      }} else if (el.kind === "point") {{
        extra = `:${{Math.round(springEval(t - el.show_at, enterDur, 0.72, 180) * 500)}}`;
      }}
      parts.push(`${{el.id}}@${{qP}}${{extra}}`);
    }}
    return parts.join("|");
  }}

  function renderAtTime(t) {{
    __vtimeMs = t * 1000.0;
    layers.innerHTML = "";
    overlay.innerHTML = "";

    const {{ activeBeat }} = activeBeatAt(t);
    const activeCue = activeCueAt(t);
    if (DATA.caption_style !== "off" && activeBeat) {{
      const showSp = !!(DATA.caption_preset && DATA.caption_preset.show_speaker_badge && activeBeat.speaker);
      if (showSp && activeBeat.speaker) {{
        spBadge.style.display = "inline-block";
        spBadge.textContent = activeBeat.speaker.replace("_", " ");
      }} else {{
        spBadge.style.display = "none";
      }}
      capText.innerHTML = "";
      if (activeCue && activeCue.lines && activeCue.lines.length > 0) {{
        const cueWords = activeCue.words || [];
        let wPtr = 0;
        for (const lineStr of activeCue.lines) {{
          const lineDiv = document.createElement("span");
          lineDiv.className = "cap-line";
          const lineToks = lineStr.split(/\\s+/).filter(Boolean);
          for (let k = 0; k < lineToks.length; k++) {{
            const tok = lineToks[k];
            const wObj = cueWords[wPtr] || null;
            if (wObj) wPtr += 1;
            const span = document.createElement("span");
            span.textContent = tok + (k + 1 < lineToks.length ? " " : "");
            if (wObj) {{
              if (t >= wObj.s && t <= wObj.e) span.className = "word-active";
              else if (t > wObj.e) span.className = "word-past";
              else span.className = "word-future";
            }} else {{
              span.className = "word-past";
            }}
            lineDiv.appendChild(span);
          }}
          capText.appendChild(lineDiv);
        }}
      }} else {{
        capText.textContent = activeBeat.caption;
      }}
    }} else if (DATA.caption_style === "off") {{
      spBadge.style.display = "none";
      capText.innerHTML = "";
    }}

    const enterDur = DATA.motion.enter_s || 0.45;
    for (const el of DATA.elements) {{
      if (t < el.show_at || t >= el.hide_at) continue;
      const p = clamp01(easeOutExpo((t - el.show_at) / enterDur));
      const [bx, by, bw, bh] = el.box;
      let tx = 0, ty = 0, rot = 0, scF = 1;
      if (el.transforms && el.transforms.length) {{
        for (const tr of el.transforms) {{
          if (t < tr.t0) continue;
          const tp = clamp01(easeInOutCubic((t - tr.t0) / Math.max(0.15, tr.t1 - tr.t0)));
          tx += (tr.dx_px || 0) * tp;
          ty += (tr.dy_px || 0) * tp;
          rot += -(tr.angle_deg || 0) * tp;
          if (tr.op === "scale" || tr.op === "zoom") {{
            scF *= 1 + ((tr.scale || 1) - 1) * tp;
          }}
        }}
      }}
      const cxEl = bx + bw * 0.5;
      const cyEl = by + bh * 0.5;
      const hasTx = Math.abs(tx) > 0.05 || Math.abs(ty) > 0.05 || Math.abs(rot) > 0.05 || Math.abs(scF - 1) > 0.002;
      const svgTarget = hasTx
        ? svgEl("g", {{ transform: `translate(${{tx.toFixed(2)}},${{ty.toFixed(2)}}) translate(${{cxEl.toFixed(1)}},${{cyEl.toFixed(1)}}) rotate(${{rot.toFixed(2)}}) scale(${{scF.toFixed(4)}}) translate(${{(-cxEl).toFixed(1)}},${{(-cyEl).toFixed(1)}})` }})
        : layers;
      if (hasTx) layers.appendChild(svgTarget);

      if (el.kind === "axes" && el.axes) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id }});
        g.appendChild(svgEl("rect", {{ x: bx, y: by, width: bw, height: bh, fill: DATA.look.surface, "fill-opacity": 0.22, stroke: DATA.look.grid_line, "stroke-width": 1, rx: 10 }}));
        g.appendChild(svgEl("line", {{ x1: bx, y1: el.axes.oy, x2: bx + bw, y2: el.axes.oy, stroke: DATA.look.ink_muted, "stroke-width": 2 }}));
        g.appendChild(svgEl("line", {{ x1: el.axes.ox, y1: by + bh, x2: el.axes.ox, y2: by, stroke: DATA.look.ink_muted, "stroke-width": 2 }}));
        svgTarget.appendChild(g);
      }} else if ((el.kind === "function_plot" || el.kind === "parametric_curve" || el.kind === "waveform" || el.kind === "data_series") && el.polyline && el.polyline.length > 1) {{
        const nVis = Math.max(2, Math.floor(el.polyline.length * p));
        const ptsStr = el.polyline.slice(0, nVis).map(pt => `${{pt[0]}},${{pt[1]}}`).join(" ");
        svgTarget.appendChild(svgEl("polyline", {{
          points: ptsStr, fill: "none", stroke: el.color, "stroke-width": 4, "stroke-linecap": "round", "stroke-linejoin": "round",
          "data-hf-elem": el.id
        }}));
      }} else if ((el.kind === "polygon" || el.kind === "probability_region") && el.polygon && el.polygon.length >= 3) {{
        const ptsStr = el.polygon.map(pt => `${{pt[0]}},${{pt[1]}}`).join(" ");
        svgTarget.appendChild(svgEl("polygon", {{
          points: ptsStr, fill: el.color, "fill-opacity": 0.30 * p, stroke: el.color, "stroke-width": 3, opacity: p, "data-hf-elem": el.id
        }}));
      }} else if (el.kind === "circle" && el.circle) {{
        svgTarget.appendChild(svgEl("circle", {{
          cx: el.circle.cx, cy: el.circle.cy, r: el.circle.r * p, fill: "none", stroke: el.color, "stroke-width": 3.2, opacity: p, "data-hf-elem": el.id
        }}));
      }} else if (el.kind === "point" && el.point) {{
        const pt = el.orbit && el.orbit.length > 1
          ? el.orbit[Math.min(el.orbit.length - 1, Math.floor((el.orbit.length - 1) * easeInOutCubic((t - el.show_at) / Math.max(0.5, el.hide_at - el.show_at))))]
          : el.point;
        svgTarget.appendChild(svgEl("circle", {{ cx: pt[0], cy: pt[1], r: 8 * springEval(t - el.show_at, enterDur, 0.72, 180), fill: el.color, "data-hf-elem": el.id }}));
      }} else if ((el.kind === "vector" || el.kind === "line") && el.segment) {{
        const x2 = el.segment.x1 + (el.segment.x2 - el.segment.x1) * p;
        const y2 = el.segment.y1 + (el.segment.y2 - el.segment.y1) * p;
        const attrs = {{ x1: el.segment.x1, y1: el.segment.y1, x2: x2, y2: y2, stroke: el.color, "stroke-width": 3.5, "data-hf-elem": el.id }};
        if (el.segment.arrow) attrs["marker-end"] = "url(#hf-arrow)";
        svgTarget.appendChild(svgEl("line", attrs));
      }} else if (el.kind === "bar_chart" && el.bars) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id }});
        const n = el.bars.length;
        const colW = bw / Math.max(1, n);
        const baseY = by + bh * 0.85;
        const maxH = bh * 0.68;
        el.bars.forEach((b, i) => {{
          const barH = Math.max(2, b.frac * maxH * p);
          const x = bx + i * colW + colW * 0.18;
          g.appendChild(svgEl("rect", {{ x: x, y: baseY - barH, width: Math.max(6, colW * 0.64), height: barH, fill: el.color, rx: 4 }}));
          if (i % Math.max(1, Math.ceil(n / 8)) === 0) {{
            const txt = svgEl("text", {{ x: x + colW * 0.32, y: baseY + 24, fill: DATA.look.ink_muted, "font-size": 20, "text-anchor": "middle" }});
            txt.textContent = b.label;
            g.appendChild(txt);
          }}
        }});
        svgTarget.appendChild(g);
      }} else if (el.kind === "graph_network" && el.network) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id }});
        const ns = el.network.nodes || [];
        for (const edge of (el.network.edges || [])) {{
          const si = parseInt(edge[0], 10), di = parseInt(edge[1], 10);
          if (ns[si] && ns[di]) {{
            g.appendChild(svgEl("line", {{ x1: ns[si][0], y1: ns[si][1], x2: ns[di][0], y2: ns[di][1], stroke: DATA.look.accent_secondary, "stroke-width": 2.5 }}));
          }}
        }}
        ns.forEach((pt, idx) => {{
          g.appendChild(svgEl("circle", {{ cx: pt[0], cy: pt[1], r: 24, fill: DATA.look.surface, stroke: el.color, "stroke-width": 3 }}));
          const lbl = svgEl("text", {{ x: pt[0], y: pt[1] + 6, fill: DATA.look.ink_primary, "font-size": 18, "font-weight": "bold", "text-anchor": "middle" }});
          lbl.textContent = (el.network.labels && el.network.labels[idx]) ? el.network.labels[idx] : `S${{idx}}`;
          g.appendChild(lbl);
        }});
        svgTarget.appendChild(g);
      }} else if (el.kind === "particle_system" && el.particles) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id }});
        for (const pt of el.particles) {{
          g.appendChild(svgEl("circle", {{ cx: pt[0], cy: pt[1], r: 4.5, fill: el.color }}));
        }}
        svgTarget.appendChild(g);
      }} else if (el.kind === "number_line" && el.number_line) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id }});
        const cy = by + bh * 0.65;
        g.appendChild(svgEl("line", {{ x1: bx + 20, y1: cy, x2: bx + bw - 20, y2: cy, stroke: el.color, "stroke-width": 3 }}));
        const x0 = el.number_line.xr[0], x1 = el.number_line.xr[1];
        for (let i = 0; i <= 4; i++) {{
          const tx = bx + 20 + ((bw - 40) * i) / 4;
          g.appendChild(svgEl("line", {{ x1: tx, y1: cy - 8, x2: tx, y2: cy + 8, stroke: el.color, "stroke-width": 2 }}));
          const lbl = svgEl("text", {{ x: tx, y: cy + 28, fill: DATA.look.ink_muted, "font-size": 20, "text-anchor": "middle" }});
          lbl.textContent = (x0 + ((x1 - x0) * i) / 4).toFixed(1).replace(/\\.0$/, "");
          g.appendChild(lbl);
        }}
        svgTarget.appendChild(g);
      }} else if (el.kind === "equation") {{
        const card = document.createElement("div");
        card.className = "hf-eq-card";
        card.setAttribute("data-hf-elem", el.id);
        card.style.opacity = String(p);
        const inner = document.createElement("div");
        inner.className = el.card ? "hf-eq-inner" : "hf-eq-bare";
        if (el.card) inner.style.borderColor = el.color;
        else inner.style.color = el.color;
        inner.textContent = el.eq_text || el.label;
        card.appendChild(inner);
        overlay.appendChild(card);
      }} else if (el.kind === "matrix" && el.matrix) {{
        const mat = document.createElement("div");
        mat.className = "hf-matrix";
        mat.setAttribute("data-hf-elem", el.id);
        mat.style.opacity = String(p);
        const cols = el.matrix[0] ? el.matrix[0].length : 2;
        mat.style.gridTemplateColumns = `repeat(${{cols}}, auto)`;
        mat.style.left = `${{Math.round(bx + bw * 0.28)}}px`;
        mat.style.top = `${{Math.round(by + bh * 0.30)}}px`;
        for (const r of el.matrix) {{
          for (const c of r) {{
            const cell = document.createElement("span");
            cell.textContent = String(c);
            mat.appendChild(cell);
          }}
        }}
        overlay.appendChild(mat);
      }} else if (el.kind === "table") {{
        const tbl = document.createElement("table");
        tbl.className = "hf-table";
        tbl.setAttribute("data-hf-elem", el.id);
        tbl.style.opacity = String(p);
        tbl.style.left = `${{Math.round(bx + 20)}}px`;
        tbl.style.top = `${{Math.round(by + 30)}}px`;
        if (el.headers && el.headers.length) {{
          const tr = document.createElement("tr");
          for (const h of el.headers) {{ const th = document.createElement("th"); th.textContent = h; tr.appendChild(th); }}
          tbl.appendChild(tr);
        }}
        for (const row of (el.rows || [])) {{
          const tr = document.createElement("tr");
          for (const v of row) {{ const td = document.createElement("td"); td.textContent = v; tr.appendChild(td); }}
          tbl.appendChild(tr);
        }}
        overlay.appendChild(tbl);
      }} else if (el.kind === "annotation" && el.text_pos) {{
        const txt = svgEl("text", {{ x: el.text_pos[0], y: el.text_pos[1], fill: el.color, "font-size": DATA.typo.body_px, "font-weight": "700", "text-anchor": "middle", opacity: p, "data-hf-elem": el.id }});
        txt.textContent = el.text || el.label;
        svgTarget.appendChild(txt);
      }}
    }}
  }}

  window.__HF_SEEK = (t) => {{
    const ts = Number(t) || 0;
    prevSig = computeRenderSig(ts);
    renderAtTime(ts);
    return true;
  }};
{AUDIT_JS_SNIPPET}
  window.__HF_STEP = (t, doAudit) => {{
    const ts = Number(t) || 0;
    const sig = computeRenderSig(ts);
    const changed = sig !== prevSig;
    if (changed) {{
      prevSig = sig;
      renderAtTime(ts);
    }}
    const issues = doAudit ? window.__HF_AUDIT_LAYOUT() : [];
    return {{ changed, issues }};
  }};

  renderAtTime(0);
}})();
"""
