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
  const svgStage = document.getElementById("svg-stage");

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

  function sweepUFor(el, t) {{
    if (!el.sweep_frames_px || !el.sweep_frames_px.length) return -1;
    const segs = el.sweep_segments || [];
    if (segs.length) {{
      let curU = segs[0].u0 !== undefined ? segs[0].u0 : (el.selected_u || 0);
      for (const s of segs) {{
        if (t < s.t0) break;
        if (t >= s.t0 && t <= s.t1) {{
          const p = clamp01((t - s.t0) / Math.max(0.15, s.t1 - s.t0));
          return clamp01(s.u0 + (s.u1 - s.u0) * p);
        }}
        curU = s.u1;
      }}
      return clamp01(curU);
    }}
    if (el.sweep_t0 !== undefined && el.sweep_t1 !== undefined) {{
      return clamp01((t - el.sweep_t0) / Math.max(0.25, el.sweep_t1 - el.sweep_t0));
    }}
    return el.selected_u !== undefined ? clamp01(el.selected_u) : 0;
  }}

  function sweepIdxFor(el, t) {{
    const u = sweepUFor(el, t);
    if (u < 0 || !el.sweep_frames_px || !el.sweep_frames_px.length) return -1;
    return Math.min(el.sweep_frames_px.length - 1, Math.round(u * (el.sweep_frames_px.length - 1)));
  }}

  function cameraViewBoxAt(t) {{
    const zEvs = DATA.zoom_events || [];
    let vx = 0, vy = 0, vw = DATA.width, vh = DATA.height;
    for (const z of zEvs) {{
      if (t >= z.t0 && t <= z.hold_end) {{
        const p = clamp01(easeInOutCubic((t - z.t0) / Math.max(0.25, z.t1 - z.t0)));
        const sc = 1 + (z.scale - 1) * p;
        const nw = DATA.width / sc;
        const nh = DATA.height / sc;
        const cx = DATA.width * 0.5 + (z.cx - DATA.width * 0.5) * p;
        const cy = DATA.height * 0.5 + (z.cy - DATA.height * 0.5) * p;
        vx = Math.max(0, Math.min(DATA.width - nw, cx - nw * 0.5));
        vy = Math.max(0, Math.min(DATA.height - nh, cy - nh * 0.5));
        vw = nw;
        vh = nh;
      }}
    }}
    return `${{vx.toFixed(1)}} ${{vy.toFixed(1)}} ${{vw.toFixed(1)}} ${{vh.toFixed(1)}}`;
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
        const swi = sweepIdxFor(el, t);
        extra = `:${{Math.max(2, Math.floor(el.polyline.length * p))}}:${{swi}}`;
      }} else if (el.kind === "point" && el.sweep_frames_px && el.sweep_frames_px.length) {{
        extra = `:sw${{sweepIdxFor(el, t)}}`;
      }} else if ((el.kind === "vector" || el.kind === "annotation" || el.kind === "mechanical_body" || el.kind === "bar_chart") && el.sweep_frames_px && el.sweep_frames_px.length) {{
        extra = `:sw${{sweepIdxFor(el, t)}}`;
      }} else if (el.kind === "point" && el.orbit && el.orbit.length > 1) {{
        const oIdx = Math.min(el.orbit.length - 1, Math.floor((el.orbit.length - 1) * easeInOutCubic((t - el.show_at) / Math.max(0.5, el.hide_at - el.show_at))));
        const sQ = Math.round(springEval(t - el.show_at, enterDur, 0.72, 180) * 500);
        extra = `:${{oIdx}}:${{sQ}}`;
      }} else if (el.kind === "point") {{
        extra = `:${{Math.round(springEval(t - el.show_at, enterDur, 0.72, 180) * 500)}}`;
      }}
      parts.push(`${{el.id}}@${{qP}}${{extra}}`);
    }}
    parts.push(cameraViewBoxAt(t));
    return parts.join("|");
  }}

  function renderAtTime(t) {{
    __vtimeMs = t * 1000.0;
    layers.innerHTML = "";
    overlay.innerHTML = "";
    if (svgStage) svgStage.setAttribute("viewBox", cameraViewBoxAt(t));

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
              if (t >= wObj.s && t <= wObj.e) span.className = wObj.emph ? "word-active-emph" : "word-active";
              else if (t > wObj.e) span.className = wObj.emph ? "word-past-emph" : "word-past";
              else span.className = "word-future";
            }} else {{
              span.className = "word-past";
            }}
            lineDiv.appendChild(span);
          }}
          capText.appendChild(lineDiv);
        }}
      }} else {{
        const prevCue = DATA.cues ? DATA.cues.find(c => c.beat_id === activeBeat.id && t > c.end && (t - c.end) <= 0.35) : null;
        if (prevCue && prevCue.lines && prevCue.lines.length > 0) {{
          for (const lineStr of prevCue.lines) {{
            const lineDiv = document.createElement("span");
            lineDiv.className = "cap-line word-past";
            lineDiv.textContent = lineStr;
            capText.appendChild(lineDiv);
          }}
        }} else {{
          capText.style.display = "none";
        }}
      }}
      if (capText.childNodes.length > 0) {{
        capText.style.display = "";
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
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id, "data-hf-role": "panel" }});
        g.appendChild(svgEl("rect", {{ x: bx, y: by, width: bw, height: bh, fill: DATA.look.surface, "fill-opacity": 0.24, stroke: DATA.look.grid_line, "stroke-width": 1.2, rx: 10 }}));
        const oxClamped = Math.max(bx, Math.min(bx + bw, el.axes.ox));
        const oyClamped = Math.max(by, Math.min(by + bh, el.axes.oy));
        const xr = el.axes.xr || [0, 1], yr = el.axes.yr || [0, 1];
        const xSpan = (xr[1] - xr[0]) || 1, ySpan = (yr[1] - yr[0]) || 1;
        const fmtTick = (v) => {{
          if (Math.abs(v) < 1e-6) return "0";
          const av = Math.abs(v);
          return av >= 10 || Math.abs(v - Math.round(v)) < 1e-3 ? String(Math.round(v * 10) / 10) : v.toFixed(1);
        }};
        const nTicks = 4;
        for (let ti = 0; ti <= nTicks; ti++) {{
          const tx = bx + (bw * ti) / nTicks;
          const xVal = xr[0] + (xSpan * ti) / nTicks;
          if (ti > 0 && ti < nTicks) {{
            g.appendChild(svgEl("line", {{ x1: tx, y1: by, x2: tx, y2: by + bh, stroke: DATA.look.grid_line, "stroke-width": 1, "stroke-opacity": 0.45, "stroke-dasharray": "3 4" }}));
          }}
          g.appendChild(svgEl("line", {{ x1: tx, y1: oyClamped - 4, x2: tx, y2: oyClamped + 4, stroke: DATA.look.ink_muted, "stroke-width": 1.5 }}));
          const xt = svgEl("text", {{ x: tx, y: by + bh + 20, fill: DATA.look.ink_muted, "font-size": 17, "text-anchor": "middle", "font-variant-numeric": "tabular-nums" }});
          xt.textContent = fmtTick(xVal);
          g.appendChild(xt);
          const ty = by + bh - (bh * ti) / nTicks;
          const yVal = yr[0] + (ySpan * ti) / nTicks;
          if (ti > 0 && ti < nTicks) {{
            g.appendChild(svgEl("line", {{ x1: bx, y1: ty, x2: bx + bw, y2: ty, stroke: DATA.look.grid_line, "stroke-width": 1, "stroke-opacity": 0.45, "stroke-dasharray": "3 4" }}));
          }}
          g.appendChild(svgEl("line", {{ x1: oxClamped - 4, y1: ty, x2: oxClamped + 4, y2: ty, stroke: DATA.look.ink_muted, "stroke-width": 1.5 }}));
          const yt = svgEl("text", {{ x: bx - 8, y: ty + 5, fill: DATA.look.ink_muted, "font-size": 17, "text-anchor": "end", "font-variant-numeric": "tabular-nums" }});
          yt.textContent = fmtTick(yVal);
          g.appendChild(yt);
        }}
        g.appendChild(svgEl("line", {{ x1: bx, y1: oyClamped, x2: bx + bw, y2: oyClamped, stroke: DATA.look.ink_muted, "stroke-width": 2.2 }}));
        g.appendChild(svgEl("line", {{ x1: oxClamped, y1: by + bh, x2: oxClamped, y2: by, stroke: DATA.look.ink_muted, "stroke-width": 2.2 }}));
        if (el.x_label) {{
          const xl = svgEl("text", {{ x: bx + bw, y: by + bh + 42, fill: DATA.look.ink_primary, "font-size": Math.max(17, DATA.typo.body_px - 6), "text-anchor": "end", "font-weight": "600" }});
          xl.textContent = el.x_label;
          g.appendChild(xl);
        }}
        if (el.y_label) {{
          const yl = svgEl("text", {{ x: bx, y: Math.max(18, by - 10), fill: DATA.look.ink_primary, "font-size": Math.max(17, DATA.typo.body_px - 6), "text-anchor": "start", "font-weight": "600" }});
          yl.textContent = el.y_label;
          g.appendChild(yl);
        }}
        if (el.legend_items && el.legend_items.length) {{
          el.legend_items.forEach((li, lIdx) => {{
            const lx = bx + bw - 8;
            const ly = Math.max(22, by + 20 + lIdx * 26);
            g.appendChild(svgEl("circle", {{ cx: lx - 8, cy: ly - 5, r: 5.5, fill: li.color }}));
            const lt = svgEl("text", {{ x: lx - 18, y: ly, fill: DATA.look.ink_primary, "font-size": Math.max(16, DATA.typo.body_px - 7), "text-anchor": "end", "font-weight": "600" }});
            lt.textContent = li.label;
            g.appendChild(lt);
          }});
        }}
        svgTarget.appendChild(g);
      }} else if ((el.kind === "function_plot" || el.kind === "parametric_curve" || el.kind === "waveform" || el.kind === "data_series") && el.polyline && el.polyline.length > 1) {{
        const swi = sweepIdxFor(el, t);
        const activeLine = (swi >= 0 && el.sweep_frames_px[swi] && el.sweep_frames_px[swi].polyline_px && el.sweep_frames_px[swi].polyline_px.length > 1)
          ? el.sweep_frames_px[swi].polyline_px
          : el.polyline;
        const nVis = swi >= 0 ? activeLine.length : Math.max(2, Math.floor(activeLine.length * p));
        const ptsStr = activeLine.slice(0, nVis).map(pt => `${{pt[0]}},${{pt[1]}}`).join(" ");
        svgTarget.appendChild(svgEl("polyline", {{
          points: ptsStr, fill: "none", stroke: el.color, "stroke-width": 4.0, "stroke-linecap": "round", "stroke-linejoin": "round",
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
        const swi = sweepIdxFor(el, t);
        const pt = (swi >= 0 && el.sweep_frames_px[swi] && el.sweep_frames_px[swi].point_px && el.sweep_frames_px[swi].point_px.length >= 2)
          ? el.sweep_frames_px[swi].point_px
          : (el.orbit && el.orbit.length > 1
            ? el.orbit[Math.min(el.orbit.length - 1, Math.floor((el.orbit.length - 1) * easeInOutCubic((t - el.show_at) / Math.max(0.5, el.hide_at - el.show_at))))]
            : el.point);
        svgTarget.appendChild(svgEl("circle", {{ cx: pt[0], cy: pt[1], r: 9 * springEval(t - el.show_at, enterDur, 0.72, 180), fill: el.color, stroke: DATA.look.ground, "stroke-width": 2, "data-hf-elem": el.id }}));
      }} else if ((el.kind === "vector" || el.kind === "line") && el.segment) {{
        const swi = sweepIdxFor(el, t);
        const segX1 = (swi >= 0 && el.sweep_frames_px[swi] && el.sweep_frames_px[swi].vec_start_px && el.sweep_frames_px[swi].vec_start_px.length >= 2) ? el.sweep_frames_px[swi].vec_start_px[0] : el.segment.x1;
        const segY1 = (swi >= 0 && el.sweep_frames_px[swi] && el.sweep_frames_px[swi].vec_start_px && el.sweep_frames_px[swi].vec_start_px.length >= 2) ? el.sweep_frames_px[swi].vec_start_px[1] : el.segment.y1;
        const tgtX2 = (swi >= 0 && el.sweep_frames_px[swi] && el.sweep_frames_px[swi].vec_end_px && el.sweep_frames_px[swi].vec_end_px.length >= 2) ? el.sweep_frames_px[swi].vec_end_px[0] : el.segment.x2;
        const tgtY2 = (swi >= 0 && el.sweep_frames_px[swi] && el.sweep_frames_px[swi].vec_end_px && el.sweep_frames_px[swi].vec_end_px.length >= 2) ? el.sweep_frames_px[swi].vec_end_px[1] : el.segment.y2;
        const x2 = swi >= 0 ? tgtX2 : (segX1 + (tgtX2 - segX1) * p);
        const y2 = swi >= 0 ? tgtY2 : (segY1 + (tgtY2 - segY1) * p);
        const attrs = {{ x1: segX1, y1: segY1, x2: x2, y2: y2, stroke: el.color, "stroke-width": 4.0, "data-hf-elem": el.id }};
        if (el.segment.arrow) attrs["marker-end"] = "url(#hf-arrow)";
        svgTarget.appendChild(svgEl("line", attrs));
      }} else if (el.kind === "bar_chart" && el.bars) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id, "data-hf-role": "bar_chart" }});
        const swiB = sweepIdxFor(el, t);
        const liveVals = (swiB >= 0 && el.sweep_frames_px[swiB] && el.sweep_frames_px[swiB].bar_values && el.sweep_frames_px[swiB].bar_values.length === el.bars.length)
          ? el.sweep_frames_px[swiB].bar_values
          : null;
        const maxLive = liveVals ? Math.max(...liveVals.map(v => Math.abs(v)), el.bars_max_val || 1e-6) : (el.bars_max_val || 1);
        const n = el.bars.length;
        const padX = Math.min(44, bw * 0.08);
        const plotW = Math.max(80, bw - 2 * padX);
        const colW = plotW / Math.max(1, n);
        const barW = Math.min(88, Math.max(14, colW * 0.56));
        const baseY = by + bh * 0.84;
        const maxH = bh * 0.62;
        // Baseline and subtle horizontal guide ticks
        g.appendChild(svgEl("line", {{ x1: bx + padX, y1: baseY, x2: bx + padX + plotW, y2: baseY, stroke: DATA.look.ink_muted, "stroke-width": 2 }}));
        [0.5, 1.0].forEach((f) => {{
          const gy = baseY - f * maxH;
          g.appendChild(svgEl("line", {{ x1: bx + padX, y1: gy, x2: bx + padX + plotW, y2: gy, stroke: DATA.look.grid, "stroke-width": 1.2, "stroke-dasharray": "4 4" }}));
        }});
        el.bars.forEach((b, i) => {{
          const rawVal = liveVals ? liveVals[i] : b.val;
          const fracUse = liveVals ? (Math.abs(liveVals[i]) / maxLive) : b.frac;
          const barH = Math.max(3, Math.min(maxH, fracUse * maxH * (swiB >= 0 ? 1 : p)));
          const colCx = bx + padX + (i + 0.5) * colW;
          const x = colCx - 0.5 * barW;
          g.appendChild(svgEl("rect", {{ x: x, y: baseY - barH, width: barW, height: barH, fill: el.color, rx: 5, opacity: 0.92 }}));
          if (n <= 8 && Number.isFinite(rawVal)) {{
            const vTxt = svgEl("text", {{ x: colCx, y: baseY - barH - 8, fill: DATA.look.ink_primary, "font-size": 17, "font-weight": "600", "text-anchor": "middle", "font-variant-numeric": "tabular-nums" }});
            vTxt.textContent = Math.abs(rawVal) >= 100 ? rawVal.toFixed(0) : rawVal.toFixed(2);
            g.appendChild(vTxt);
          }}
          if (i % Math.max(1, Math.ceil(n / 8)) === 0) {{
            const txt = svgEl("text", {{ x: colCx, y: baseY + 24, fill: DATA.look.ink_muted, "font-size": 18, "text-anchor": "middle" }});
            txt.textContent = b.label;
            g.appendChild(txt);
          }}
        }});
        svgTarget.appendChild(g);
      }} else if (el.kind === "vector_field" && el.field_arrows) {{
        const g = svgEl("g", {{ opacity: p * 0.85, "data-hf-elem": el.id }});
        for (const a of el.field_arrows) {{
          g.appendChild(svgEl("line", {{ x1: a[0], y1: a[1], x2: a[2], y2: a[3], stroke: el.color, "stroke-width": 2, "marker-end": "url(#hf-arrow)" }}));
        }}
        svgTarget.appendChild(g);
      }} else if (el.kind === "mechanical_body" && el.mech) {{
        const swiM = sweepIdxFor(el, t);
        const bodyPt = (swiM >= 0 && el.sweep_frames_px[swiM] && el.sweep_frames_px[swiM].body_px && el.sweep_frames_px[swiM].body_px.length >= 2)
          ? el.sweep_frames_px[swiM].body_px
          : el.mech.body_px;
        const ancPt = el.mech.anchor_px;
        const ck = el.mech.connector || "spring";
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id, "data-hf-role": "panel" }});
        const dx = bodyPt[0] - ancPt[0], dy = bodyPt[1] - ancPt[1];
        const distM = Math.hypot(dx, dy) || 1;
        // Fixed structural wall/ceiling anchor bracket
        if (ck === "rod") {{
          g.appendChild(svgEl("line", {{ x1: ancPt[0] - 42, y1: ancPt[1], x2: ancPt[0] + 42, y2: ancPt[1], stroke: DATA.look.ink_primary, "stroke-width": 4.5, "stroke-linecap": "round" }}));
        }} else {{
          g.appendChild(svgEl("line", {{ x1: ancPt[0], y1: ancPt[1] - 46, x2: ancPt[0], y2: ancPt[1] + 54, stroke: DATA.look.ink_primary, "stroke-width": 4.5, "stroke-linecap": "round" }}));
          for (let hi = -3; hi <= 3; hi++) {{
            const hy = ancPt[1] + hi * 14;
            g.appendChild(svgEl("line", {{ x1: ancPt[0] - 12, y1: hy + 8, x2: ancPt[0], y2: hy, stroke: DATA.look.ink_muted, "stroke-width": 1.6 }}));
          }}
        }}
        if (ck === "spring" || ck === "spring_damper") {{
          const ux = dx / distM, uy = dy / distM, nx = -uy, ny = ux;
          const nCoils = 10;
          const ptsS = [`${{ancPt[0]}},${{ancPt[1]}}`, `${{(ancPt[0] + dx * 0.08).toFixed(1)}},${{(ancPt[1] + dy * 0.08).toFixed(1)}}`];
          for (let ci = 0; ci < nCoils; ci++) {{
            const tc = 0.10 + 0.80 * (ci + 0.5) / nCoils;
            const sgn = ci % 2 === 0 ? 1 : -1;
            ptsS.push(`${{(ancPt[0] + dx * tc + nx * sgn * 18).toFixed(1)}},${{(ancPt[1] + dy * tc + ny * sgn * 18).toFixed(1)}}`);
          }}
          ptsS.push(`${{(ancPt[0] + dx * 0.92).toFixed(1)}},${{(ancPt[1] + dy * 0.92).toFixed(1)}}`, `${{bodyPt[0]}},${{bodyPt[1]}}`);
          g.appendChild(svgEl("polyline", {{ points: ptsS.join(" "), fill: "none", stroke: DATA.look.ink_muted, "stroke-width": 3.4, "stroke-linejoin": "round" }}));
        }}
        if (ck === "damper" || ck === "spring_damper") {{
          const offY = ck === "spring_damper" ? 30 : 0;
          const midX = ancPt[0] + dx * 0.50;
          g.appendChild(svgEl("line", {{ x1: ancPt[0], y1: ancPt[1] + offY, x2: midX, y2: ancPt[1] + offY + dy * 0.50, stroke: DATA.look.accent_secondary, "stroke-width": 3.2 }}));
          g.appendChild(svgEl("rect", {{ x: midX - 20, y: ancPt[1] + offY + dy * 0.50 - 11, width: 40, height: 22, rx: 4, fill: DATA.look.surface, stroke: DATA.look.accent_secondary, "stroke-width": 2.6 }}));
          g.appendChild(svgEl("line", {{ x1: midX, y1: ancPt[1] + offY + dy * 0.50, x2: bodyPt[0], y2: bodyPt[1] + offY, stroke: DATA.look.accent_secondary, "stroke-width": 3.0 }}));
        }}
        if (ck === "rod") {{
          g.appendChild(svgEl("line", {{ x1: ancPt[0], y1: ancPt[1], x2: bodyPt[0], y2: bodyPt[1], stroke: DATA.look.ink_primary, "stroke-width": 3.4 }}));
        }}
        g.appendChild(svgEl("circle", {{ cx: ancPt[0], cy: ancPt[1], r: 7, fill: DATA.look.ink_primary }}));
        if (ck === "rod") {{
          g.appendChild(svgEl("circle", {{ cx: bodyPt[0], cy: bodyPt[1], r: 30, fill: el.color, stroke: DATA.look.ground, "stroke-width": 3 }}));
        }} else {{
          g.appendChild(svgEl("rect", {{ x: bodyPt[0] - 36, y: bodyPt[1] - 32, width: 72, height: 64, rx: 10, fill: el.color, stroke: DATA.look.ground, "stroke-width": 3 }}));
        }}
        const massTxt = svgEl("text", {{ x: bodyPt[0], y: bodyPt[1] + 7, fill: DATA.look.ground, "font-size": 21, "font-weight": "700", "text-anchor": "middle" }});
        massTxt.textContent = (el.label || "m").slice(0, 8);
        g.appendChild(massTxt);
        if (swiM >= 0 && el.sweep_frames_px[swiM] && el.sweep_frames_px[swiM].label) {{
          const rd = svgEl("text", {{ x: bodyPt[0], y: bodyPt[1] + 54, fill: DATA.look.ink_primary, "font-size": 17, "font-weight": "600", "text-anchor": "middle", "font-variant-numeric": "tabular-nums" }});
          rd.textContent = el.sweep_frames_px[swiM].label.split("|").pop().trim();
          g.appendChild(rd);
        }}
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
      }} else if (el.kind === "fraction_bar" && el.frac_bar) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id, "data-hf-role": "panel" }});
        const {{ num, den, sub, label }} = el.frac_bar;
        const barW = Math.min(bw * 0.84, 720);
        const barH = Math.min(96, bh * 0.34);
        const x0 = bx + (bw - barW) * 0.5;
        const y0 = by + (bh - barH) * 0.5;
        const totalParts = den * sub;
        const filledParts = num * sub;
        const partW = barW / Math.max(1, totalParts);
        for (let i = 0; i < totalParts; i++) {{
          const isFilled = i < filledParts;
          g.appendChild(svgEl("rect", {{
            x: x0 + i * partW + 1.5, y: y0, width: Math.max(4, partW - 3), height: barH, rx: 4,
            fill: isFilled ? el.color : DATA.look.surface, stroke: DATA.look.ink_primary, "stroke-width": 2.2
          }}));
        }}
        const lbl = svgEl("text", {{ x: bx + bw * 0.5, y: y0 - 18, fill: DATA.look.ink_primary, "font-size": 28, "font-weight": "700", "text-anchor": "middle" }});
        lbl.textContent = label;
        g.appendChild(lbl);
        svgTarget.appendChild(g);
      }} else if (el.kind === "fraction_circle" && el.frac_circle) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id, "data-hf-role": "panel" }});
        const {{ num, den, sub, label }} = el.frac_circle;
        const total = den * sub;
        const filled = num * sub;
        const cx = bx + bw * 0.5, cy = by + bh * 0.52;
        const rad = Math.min(bw, bh) * 0.34;
        for (let i = 0; i < total; i++) {{
          const a0 = -Math.PI / 2 + (2 * Math.PI * i) / total;
          const a1 = -Math.PI / 2 + (2 * Math.PI * (i + 1)) / total;
          const x1 = cx + rad * Math.cos(a0), y1 = cy + rad * Math.sin(a0);
          const x2 = cx + rad * Math.cos(a1), y2 = cy + rad * Math.sin(a1);
          const largeArc = (a1 - a0) > Math.PI ? 1 : 0;
          const d = `M ${{cx}} ${{cy}} L ${{x1.toFixed(1)}} ${{y1.toFixed(1)}} A ${{rad.toFixed(1)}} ${{rad.toFixed(1)}} 0 ${{largeArc}} 1 ${{x2.toFixed(1)}} ${{y2.toFixed(1)}} Z`;
          g.appendChild(svgEl("path", {{ d: d, fill: i < filled ? el.color : DATA.look.surface, stroke: DATA.look.ink_primary, "stroke-width": 2.4 }}));
        }}
        const lbl = svgEl("text", {{ x: cx, y: cy - rad - 18, fill: DATA.look.ink_primary, "font-size": 28, "font-weight": "700", "text-anchor": "middle" }});
        lbl.textContent = label;
        g.appendChild(lbl);
        svgTarget.appendChild(g);
      }} else if (el.kind === "array_grid" && el.array_grid) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id, "data-hf-role": "panel" }});
        const {{ rows, cols, count, label }} = el.array_grid;
        const cell = Math.min(96, Math.min((bw * 0.78) / cols, (bh * 0.68) / rows));
        const gridW = cell * cols, gridH = cell * rows;
        const x0 = bx + (bw - gridW) * 0.5, y0 = by + (bh - gridH) * 0.52;
        let idx = 0;
        for (let r = 0; r < rows; r++) {{
          for (let c = 0; c < cols; c++) {{
            const active = idx < count;
            g.appendChild(svgEl("rect", {{
              x: x0 + c * cell + 3, y: y0 + r * cell + 3, width: Math.max(6, cell - 6), height: Math.max(6, cell - 6),
              rx: 6, fill: active ? el.color : DATA.look.surface, stroke: DATA.look.ink_primary, "stroke-width": 1.8, opacity: active ? 0.92 : 0.45
            }}));
            idx += 1;
          }}
        }}
        const lbl = svgEl("text", {{ x: bx + bw * 0.5, y: y0 - 16, fill: DATA.look.ink_primary, "font-size": 26, "font-weight": "700", "text-anchor": "middle" }});
        lbl.textContent = label;
        g.appendChild(lbl);
        svgTarget.appendChild(g);
      }} else if (el.kind === "counting_group" && el.counting_group) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id, "data-hf-role": "panel" }});
        const {{ count, group_size, label }} = el.counting_group;
        const cols = Math.min(group_size, Math.max(1, count));
        const rows = Math.max(1, Math.ceil(count / Math.max(1, group_size)));
        const step = Math.min(88, (bw * 0.8) / Math.max(1, cols), (bh * 0.68) / Math.max(1, rows));
        const rad = Math.max(10, step * 0.34);
        const x0 = bx + (bw - cols * step) * 0.5 + step * 0.5;
        const y0 = by + (bh - rows * step) * 0.52 + step * 0.5;
        for (let i = 0; i < count; i++) {{
          const r = Math.floor(i / group_size), c = i % group_size;
          g.appendChild(svgEl("circle", {{ cx: x0 + c * step, cy: y0 + r * step, r: rad, fill: el.color, stroke: DATA.look.ground, "stroke-width": 2.4 }}));
        }}
        const lbl = svgEl("text", {{ x: bx + bw * 0.5, y: y0 - step * 0.5 - 16, fill: DATA.look.ink_primary, "font-size": 26, "font-weight": "700", "text-anchor": "middle" }});
        lbl.textContent = label;
        g.appendChild(lbl);
        svgTarget.appendChild(g);
      }} else if (el.kind === "equation_balance" && el.eq_balance) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id, "data-hf-role": "panel" }});
        const cx = bx + bw * 0.5, baseY = by + bh * 0.74, beamY = by + bh * 0.42;
        const span = Math.min(bw * 0.34, 240);
        g.appendChild(svgEl("polygon", {{ points: `${{cx - 28}},${{baseY}} ${{cx + 28}},${{baseY}} ${{cx}},${{beamY + 8}}`, fill: DATA.look.accent_secondary }}));
        g.appendChild(svgEl("line", {{ x1: cx - span, y1: beamY, x2: cx + span, y2: beamY, stroke: DATA.look.ink_primary, "stroke-width": 5, "stroke-linecap": "round" }}));
        [[-span, el.eq_balance.lhs], [span, el.eq_balance.rhs]].forEach(([dx, txtStr]) => {{
          const px = cx + dx;
          g.appendChild(svgEl("rect", {{ x: px - 105, y: beamY - 64, width: 210, height: 52, rx: 10, fill: DATA.look.surface, stroke: el.color, "stroke-width": 2.6 }}));
          const tEl = svgEl("text", {{ x: px, y: beamY - 30, fill: DATA.look.ink_primary, "font-size": 24, "font-weight": "700", "text-anchor": "middle" }});
          tEl.textContent = txtStr;
          g.appendChild(tEl);
        }});
        svgTarget.appendChild(g);
      }} else if (el.kind === "concept_card" && el.concept_card) {{
        const g = svgEl("g", {{ opacity: p, "data-hf-elem": el.id, "data-hf-role": "panel" }});
        const cw = Math.min(bw - 24, 1180);
        const steps = (el.concept_card.steps || []).slice(0, 5);
        const rowH = 44;
        const ch = Math.min(bh - 20, 84 + steps.length * rowH);
        const cx0 = bx + (bw - cw) * 0.5;
        const cy0 = by + (bh - ch) * 0.5;
        g.appendChild(svgEl("rect", {{ x: cx0, y: cy0, width: cw, height: ch, rx: 16, fill: DATA.look.surface, stroke: el.color, "stroke-width": 3 }}));
        const hTxt = svgEl("text", {{ x: cx0 + 28, y: cy0 + 42, fill: el.color, "font-size": 26, "font-weight": "700" }});
        hTxt.textContent = (el.concept_card.title || "").slice(0, 72);
        g.appendChild(hTxt);
        steps.forEach((stStr, si) => {{
          const rTxt = svgEl("text", {{ x: cx0 + 32, y: cy0 + 82 + si * rowH, fill: DATA.look.ink_primary, "font-size": 22 }});
          rTxt.textContent = `${{si + 1}}. ${{String(stStr).slice(0, 82)}}`;
          g.appendChild(rTxt);
        }});
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
        if (window.katex && el.tex) {{
          try {{
            window.katex.render(String(el.tex), inner, {{ throwOnError: false, displayMode: false }});
          }} catch (_e) {{
            inner.textContent = el.eq_text || el.label;
          }}
        }} else {{
          inner.textContent = el.eq_text || el.label;
        }}
        card.appendChild(inner);
        overlay.appendChild(card);
        const availW = Math.max(200, (DATA.eq_box ? DATA.eq_box[2] : DATA.width * 0.85) - 32);
        const rectInner = inner.getBoundingClientRect();
        if (rectInner.width > availW) {{
          const sEq = Math.max(0.55, availW / rectInner.width);
          inner.style.transform = `scale(${{sEq.toFixed(3)}})`;
        }}
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
        tbl.style.left = `${{Math.round(bx + bw * 0.5)}}px`;
        tbl.style.top = `${{Math.round(by + bh * 0.48)}}px`;
        tbl.style.transform = "translate(-50%, -50%)";
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
        const swi = sweepIdxFor(el, t);
        const annLabel = (swi >= 0 && el.sweep_frames_px[swi] && el.sweep_frames_px[swi].label)
          ? el.sweep_frames_px[swi].label
          : (el.text || el.label);
        const isReadout = annLabel.includes("=") || annLabel.includes("|");
        const fsAnn = isReadout ? Math.max(20, DATA.typo.body_px - 3) : DATA.typo.body_px;
        if (isReadout) {{
          const rw = Math.min(DATA.width * 0.42, Math.max(220, annLabel.length * fsAnn * 0.56 + 28));
          const rh = fsAnn * 1.65;
          const gAnn = svgEl("g", {{ opacity: p, "data-hf-elem": el.id }});
          gAnn.appendChild(svgEl("rect", {{ x: el.text_pos[0] - rw * 0.5, y: el.text_pos[1] - rh * 0.68, width: rw, height: rh, rx: 8, fill: DATA.look.surface, stroke: DATA.look.surface_border, "stroke-width": 1.2 }}));
          const txt = svgEl("text", {{ x: el.text_pos[0], y: el.text_pos[1], fill: el.color, "font-size": fsAnn, "font-weight": "600", "text-anchor": "middle", "font-variant-numeric": "tabular-nums" }});
          txt.textContent = annLabel;
          gAnn.appendChild(txt);
          svgTarget.appendChild(gAnn);
        }} else {{
          const txt = svgEl("text", {{ x: el.text_pos[0], y: el.text_pos[1], fill: el.color, "font-size": fsAnn, "font-weight": "700", "text-anchor": "middle", opacity: p, "data-hf-elem": el.id }});
          txt.textContent = annLabel;
          svgTarget.appendChild(txt);
        }}
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
