"""Lightweight local scene-preview workflow reusing production schemas, storyboards, CaptionIR, SceneRuntimeState, and QA."""

from __future__ import annotations

import argparse
import html
import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from studio.ai.v3_schema import VideoPlanV3
from studio.ai.world_model import materialize_constructions
from studio.captions import CaptionIR, build_caption_ir
from studio.creative.script_ir import ScriptIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from studio.design.spec import NAMED_LOOKS, default_design_spec, resolve_look
from studio.qa.engine import audit_educational_quality
from studio.renderers import get_backend
from studio.renderers.hyperframes.compiler import compile_plan_to_html
from studio.renderers.scene_state import SceneRuntimeState, evaluate_plan_states, materialize_carry_objects
from studio.storyboard import storyboard_from_v3_plan
from studio.timeline import AudioTimeline


def _load_plan_bundle(source: Path) -> tuple[str, list[tuple[str, VideoPlanV3, AudioTimeline | None]], ScriptIR | None]:
    """Load a standalone VideoPlanV3 JSON or a long-form directory into inspectable plans."""
    src = Path(source)
    if src.is_file():
        plan = VideoPlanV3.model_validate_json(src.read_text(encoding="utf-8"))
        tl_p = src.parent / "timeline.json"
        tl = AudioTimeline.load(tl_p) if tl_p.exists() else None
        return plan.title or plan.topic, [("main", plan, tl)], None

    if not src.is_dir():
        raise FileNotFoundError(f"Preview source not found: {src}")

    script_ir = None
    for sname in ("script_ir_approved.json", "script_ir.json"):
        sp = src / sname
        if sp.exists():
            try:
                script_ir = ScriptIR.model_validate_json(sp.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    lf_plan_p = src / "longform_plan.json"
    chapters: list[tuple[str, VideoPlanV3, AudioTimeline | None]] = []
    if lf_plan_p.exists():
        lf_raw = json.loads(lf_plan_p.read_text(encoding="utf-8"))
        title = str(lf_raw.get("topic") or src.name)
        for ch in lf_raw.get("chapters", []):
            cid = str(ch.get("chapter_id") or "")
            cand_plan = [
                src / "logs" / f"ch_{cid}.json",
                src / "chapters" / cid / "plan.json",
                PROJECT_ROOT / "build" / f"v3_{src.name.replace('longform_', '')}_{cid}" / "plan.json",
            ]
            # Also match hashed chapter logs `logs/ch_<hash>_<cid>.json`
            cand_plan.extend(sorted((src / "logs").glob(f"ch_*_{cid}.json")) if (src / "logs").exists() else [])
            p_file = next((p for p in cand_plan if p.exists()), None)
            if p_file is None:
                continue
            cp = VideoPlanV3.model_validate_json(p_file.read_text(encoding="utf-8"))
            tl_cand = [
                PROJECT_ROOT / "build" / f"v3_{src.name.replace('longform_', '')}_{cid}" / "timeline.json",
                src / "chapters" / cid / "timeline.json",
            ]
            tl_f = next((t for t in tl_cand if t.exists()), None)
            tl = AudioTimeline.load(tl_f) if tl_f else None
            chapters.append((cid, cp, tl))
        if chapters:
            return title, chapters, script_ir

    plan_files = sorted(src.glob("*.json"))
    for pf in plan_files:
        try:
            vp = VideoPlanV3.model_validate_json(pf.read_text(encoding="utf-8"))
            tl_p = src / "timeline.json"
            return vp.title or vp.topic, [("main", vp, AudioTimeline.load(tl_p) if tl_p.exists() else None)], script_ir
        except Exception:
            continue
    raise FileNotFoundError(f"No valid VideoPlanV3 JSON found inside {src}")


def slice_plan_for_scene(plan: VideoPlanV3, target_scene_id: str) -> tuple[VideoPlanV3, dict[str, Any]]:
    """Extract a standalone single-scene VideoPlanV3 initialized with evaluated SceneRuntimeState."""
    mat = materialize_constructions(plan)
    states = evaluate_plan_states(mat)
    idx = next((i for i, sc in enumerate(mat.scenes) if sc.scene_id == target_scene_id), -1)
    if idx < 0:
        raise KeyError(f"Scene '{target_scene_id}' not found in plan '{plan.topic}'.")
    in_state = states[idx - 1] if idx > 0 else SceneRuntimeState()
    out_state = states[idx]
    target_sc = mat.scenes[idx].model_copy(deep=True)
    existing_ids = {o.id for o in target_sc.objects}
    needed_ids = existing_ids | {e.target_id for e in target_sc.events if e.target_id} | {
        e.source_id for e in target_sc.events if e.source_id
    }
    carried = [
        o for o in materialize_carry_objects(in_state, needed_ids) if o.id not in existing_ids
    ]
    target_sc.objects = carried + list(target_sc.objects)
    sliced = VideoPlanV3(
        schema_version=mat.schema_version,
        topic=f"{mat.topic} [{target_scene_id}]",
        title=target_sc.purpose or target_scene_id,
        audience=mat.audience,
        strategy=mat.strategy,
        target_duration_s=12.0,
        scenes=[target_sc],
    )
    return sliced, {
        "scene_id": target_scene_id,
        "carried_object_count": len(carried),
        "carried_object_ids": [o.id for o in carried],
        "removed_ids": sorted(in_state.removed_ids),
        "out_state": out_state.model_dump(mode="json"),
    }


def render_single_scene_preview(
    plan_or_path: VideoPlanV3 | Path,
    scene_id: str,
    *,
    out_dir: Path | None = None,
    format_name: str = "widescreen_hd",
    look_name: str = "light",
    caption_style: str = "editorial",
    backend: str = "hyperframes",
    with_tts: bool = False,
) -> dict[str, Any]:
    """Render an isolated single-scene preview preserving carried SceneRuntimeState."""
    t0 = time.perf_counter()
    if isinstance(plan_or_path, Path):
        _, bundles, _ = _load_plan_bundle(plan_or_path)
        plan = next((p for _, p, _ in bundles if any(s.scene_id == scene_id for s in p.scenes)), bundles[0][1])
    else:
        plan = plan_or_path

    sliced, state_info = slice_plan_for_scene(plan, scene_id)
    dest = out_dir or (PROJECT_ROOT / "build" / "scene_previews" / scene_id)
    dest.mkdir(parents=True, exist_ok=True)
    design = default_design_spec(format_name=format_name, look_name=resolve_look(look_name).name)
    html_str, dur_s = compile_plan_to_html(sliced, design, None, caption_style=caption_style)
    preview_html = dest / f"{scene_id}_{design.look.name}_{caption_style}.html"
    preview_html.write_text(html_str, encoding="utf-8")

    # Save sliced plan JSON + compiled canonical CaptionIR sidecar
    plan_p = dest / f"{scene_id}_sliced_plan.json"
    plan_p.write_text(sliced.model_dump_json(indent=2), encoding="utf-8")

    mp4_path = dest / f"{scene_id}_{format_name}_{design.look.name}_{caption_style}.mp4"
    res = get_backend(backend).render(
        sliced,
        slug=f"preview_{scene_id}_{design.look.name}",
        format_name=format_name,
        look_name=design.look.name,
        out_path=mp4_path,
        caption_style=caption_style,
        tts_engine="kokoro" if with_tts else "silent",
        allow_edge_fallback=False,
    )
    cap_ir_p = dest / f"{scene_id}_caption_ir.json"
    if not with_tts:
        from studio.audio.tts import SyncedBeat
        from studio.creative.dialogue import clean_spoken_text

        txt = clean_spoken_text(sliced.scenes[0].narration)
        sb = SyncedBeat(
            id=scene_id,
            narration=txt,
            caption=sliced.scenes[0].purpose,
            audio_path="",
            start_time=0.0,
            end_time=round(dur_s, 3),
            speech_duration=round(max(0.5, dur_s - 0.3), 3),
            total_duration=round(dur_s, 3),
            words=[],
            speaker=getattr(sliced.scenes[0], "speaker", "narrator"),
        )
        cir = build_caption_ir([sb], style=caption_style, width_px=design.safe_zone.width, height_px=design.safe_zone.height)
        cap_ir_p.write_text(cir.model_dump_json(indent=2), encoding="utf-8")

    cand_srt = res.video_path.with_suffix(".srt")
    rev = audit_educational_quality(
        topic=sliced.topic,
        title=sliced.title,
        plan=sliced,
        format_name=format_name,
        backend=backend,
        video_path=res.video_path,
        contact_sheet_out=dest / f"{scene_id}_qa_sheet.png",
        look_name=design.look.name,
        caption_ir_path=cap_ir_p if cap_ir_p.exists() else None,
        srt_path=cand_srt if cand_srt.exists() else None,
    )
    elapsed_s = round(time.perf_counter() - t0, 2)
    return {
        "scene_id": scene_id,
        "look_name": design.look.name,
        "caption_style": caption_style,
        "backend": backend,
        "preview_html": str(preview_html),
        "video_path": str(res.video_path),
        "contact_sheet": str(getattr(rev, "contact_sheet_path", "")),
        "production_state": str(getattr(rev, "production_state", "review_required")),
        "qa_score": float(getattr(rev, "overall_score", 0.0)),
        "qa_approved": bool(getattr(rev, "approved", False)),
        "state_info": state_info,
        "elapsed_s": elapsed_s,
    }


def build_scene_preview_bundle(
    source: Path,
    *,
    out_dir: Path | None = None,
    format_name: str = "widescreen_hd",
    look_name: str = "light",
    caption_style: str = "editorial",
) -> Path:
    """Build an offline HTML preview workbench from production schemas, storyboards, and QA."""
    title, bundles, script_ir = _load_plan_bundle(source)
    dest = out_dir or (PROJECT_ROOT / "build" / f"preview_{Path(source).stem}")
    dest.mkdir(parents=True, exist_ok=True)
    scenes_dir = dest / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    resolved_look = resolve_look(look_name).name
    records: list[dict[str, Any]] = []
    for ch_id, plan, tl in bundles:
        mat = materialize_constructions(plan)
        states = evaluate_plan_states(mat)
        sb = storyboard_from_v3_plan(mat, format_name=format_name, look_name=resolved_look, audio_timeline=tl)
        rev = audit_educational_quality(
            topic=mat.topic,
            title=mat.title,
            plan=mat,
            format_name=format_name,
            look_name=resolved_look,
            timeline=tl,
            script=script_ir,
        )
        # Render interactive HyperFrames preview HTML for each theme in LOOK_PRESETS
        for idx, (sc, shot) in enumerate(zip(mat.scenes, sb.shots)):
            sliced, st_info = slice_plan_for_scene(mat, sc.scene_id)
            theme_urls: dict[str, str] = {}
            for lk_key in ("light", "midnight_studio", "editorial_warm", "sunset_slate"):
                if lk_key not in NAMED_LOOKS:
                    continue
                d_spec = default_design_spec(format_name=format_name, look_name=lk_key)
                h_str, _ = compile_plan_to_html(sliced, d_spec, tl, caption_style=caption_style)
                rel = f"scenes/{ch_id}__{sc.scene_id}__{lk_key}.html"
                (dest / rel).write_text(h_str, encoding="utf-8")
                theme_urls[lk_key] = rel

            in_s = states[idx - 1] if idx > 0 else SceneRuntimeState()
            out_s = states[idx]
            records.append(
                {
                    "chapter_id": ch_id,
                    "scene_id": sc.scene_id,
                    "purpose": sc.purpose,
                    "speaker": getattr(sc, "speaker", "narrator"),
                    "start_s": shot.start_s,
                    "end_s": shot.end_s,
                    "duration_s": shot.duration_s,
                    "composition": shot.composition,
                    "narration": sc.narration,
                    "claims": [c.model_dump() for c in sc.claims],
                    "constructions": [c.model_dump() for c in sc.constructions],
                    "objects": [o.model_dump() for o in sc.objects],
                    "events": [e.model_dump() for e in sc.events],
                    "state_in": in_s.model_dump(mode="json"),
                    "state_out": out_s.model_dump(mode="json"),
                    "carried_ids": st_info["carried_object_ids"],
                    "qa_state": str(getattr(rev, "production_state", "draft")),
                    "qa_score": float(getattr(rev, "overall_score", 0.0)),
                    "qa_categories": dict(getattr(rev, "category_statuses", {})),
                    "qa_notes": [n.model_dump() for n in getattr(rev, "notes", []) if n.shot_id in (sc.scene_id, "plan", "global")],
                    "theme_urls": theme_urls,
                }
            )

    manifest = {
        "title": title,
        "format_name": format_name,
        "default_look": resolved_look,
        "caption_style": caption_style,
        "script_provenance": {
            "approved_revision": script_ir.approved_revision if script_ir else 0,
            "locked": bool(script_ir.locked) if script_ir else False,
            "revisions": len(script_ir.revisions) if script_ir else 0,
        },
        "scene_count": len(records),
        "scenes": records,
    }
    (dest / "preview_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=lambda x: sorted(x) if isinstance(x, set) else str(x)), encoding="utf-8")

    safe_title = html.escape(title)
    manifest_json = json.dumps(manifest, ensure_ascii=False)
    index_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Scene Preview — {safe_title}</title>
<style>
:root {{ --bg: #0f172a; --panel: #182235; --ink: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8; --border: #2c3b55; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--ink); font-family: system-ui, -apple-system, sans-serif; display: grid; grid-template-columns: 320px 1fr 380px; height: 100vh; overflow: hidden; }}
aside, section {{ overflow-y: auto; padding: 14px; border-right: 1px solid var(--border); }}
main {{ display: flex; flex-direction: column; overflow: hidden; border-right: 1px solid var(--border); }}
.hdr {{ padding: 12px 16px; background: var(--panel); border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; gap: 10px; }}
.scene-btn {{ width: 100%; text-align: left; background: var(--panel); border: 1px solid var(--border); color: var(--ink); padding: 10px; border-radius: 8px; margin-bottom: 8px; cursor: pointer; }}
.scene-btn.active {{ border-color: var(--accent); background: #1e2d47; }}
.meta {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
.stage-wrap {{ flex: 1; display: flex; align-items: center; justify-content: center; padding: 12px; background: #090d16; }}
iframe {{ width: 100%; height: 100%; border: 1px solid var(--border); border-radius: 10px; background: #fff; }}
pre {{ background: #0b111e; border: 1px solid var(--border); border-radius: 8px; padding: 10px; font-size: 12px; overflow-x: auto; margin-top: 8px; white-space: pre-wrap; }}
select {{ background: var(--bg); color: var(--ink); border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; background: #1e3a5f; color: #7dd3fc; margin-right: 6px; }}
</style>
</head>
<body>
<aside>
  <h3 style="margin-bottom:10px;">Storyboard Scenes ({len(records)})</h3>
  <div id="scene-list"></div>
</aside>
<main>
  <div class="hdr">
    <div><strong id="cur-title"></strong> <span class="meta" id="cur-time"></span></div>
    <div>
      <label style="font-size:12px;color:var(--muted);">Theme:
        <select id="theme-sel">
          <option value="light">light</option>
          <option value="midnight_studio">midnight_studio</option>
          <option value="editorial_warm">editorial_warm</option>
          <option value="sunset_slate">sunset_slate</option>
        </select>
      </label>
    </div>
  </div>
  <div class="stage-wrap"><iframe id="stage-frame"></iframe></div>
</main>
<section>
  <div style="margin-bottom:10px;"><span class="badge" id="qa-badge"></span><span class="badge" id="comp-badge"></span></div>
  <h4>Narration & Bookmarks</h4>
  <pre id="narration-box"></pre>
  <h4 style="margin-top:12px;">Evaluated SceneRuntimeState</h4>
  <pre id="state-box"></pre>
  <h4 style="margin-top:12px;">Claims & QA Findings</h4>
  <pre id="qa-box"></pre>
</section>
<script>
const MANIFEST = {manifest_json};
let curIdx = 0;
const themeSel = document.getElementById("theme-sel");
themeSel.value = MANIFEST.default_look || "light";
const listEl = document.getElementById("scene-list");
function selectScene(i) {{
  curIdx = i;
  const sc = MANIFEST.scenes[i];
  document.querySelectorAll(".scene-btn").forEach((b, k) => b.classList.toggle("active", k === i));
  document.getElementById("cur-title").textContent = `${{sc.chapter_id}} / ${{sc.scene_id}} — ${{sc.purpose}}`;
  document.getElementById("cur-time").textContent = `${{sc.start_s.toFixed(1)}}s–${{sc.end_s.toFixed(1)}}s (${{sc.duration_s.toFixed(1)}}s)`;
  document.getElementById("qa-badge").textContent = `QA: ${{sc.qa_state}} (${{sc.qa_score}}/100)`;
  document.getElementById("comp-badge").textContent = sc.composition;
  document.getElementById("narration-box").textContent = `[${{sc.speaker}}] ${{sc.narration}}`;
  document.getElementById("state-box").textContent = JSON.stringify({{ carried_ids: sc.carried_ids, state_out: sc.state_out }}, null, 2);
  document.getElementById("qa-box").textContent = JSON.stringify({{ claims: sc.claims, notes: sc.qa_notes }}, null, 2);
  const lk = themeSel.value;
  document.getElementById("stage-frame").src = sc.theme_urls[lk] || Object.values(sc.theme_urls)[0] || "";
}}
MANIFEST.scenes.forEach((sc, idx) => {{
  const btn = document.createElement("button");
  btn.className = "scene-btn";
  btn.innerHTML = `<strong>${{sc.scene_id}}</strong><div class="meta">${{sc.chapter_id}} · ${{sc.duration_s.toFixed(1)}}s · ${{sc.purpose}}</div>`;
  btn.onclick = () => selectScene(idx);
  listEl.appendChild(btn);
}});
themeSel.onchange = () => selectScene(curIdx);
if (MANIFEST.scenes.length) selectScene(0);
</script>
</body>
</html>
"""
    index_path = dest / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    return index_path


def start_preview_server(root_dir: Path, *, port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start a strictly loopback-only (127.0.0.1) preview HTTP server."""

    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(root_dir), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            return None

    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), _Handler)
    actual_port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, name=f"preview-{actual_port}", daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{actual_port}/index.html"


def main() -> int:
    parser = argparse.ArgumentParser(description="Math Shorts Studio V3.8 local scene preview workbench")
    parser.add_argument("--source", required=True, help="Path to VideoPlanV3 JSON or longform build directory")
    parser.add_argument("--scene", default="", help="Optional single scene_id to render as isolated MP4 preview")
    parser.add_argument("--look", default="light", help="Visual look preset (light, midnight_studio, editorial_warm, sunset_slate)")
    parser.add_argument("--caption-style", default="editorial", help="Caption preset (editorial, kinetic, minimal, off)")
    parser.add_argument("--format", default="widescreen_hd", dest="format_name", help="Canvas format")
    parser.add_argument("--backend", default="hyperframes", help="Renderer backend for single-scene preview")
    parser.add_argument("--with-tts", action="store_true", help="Synthesize Kokoro audio in single-scene preview")
    parser.add_argument("--serve", action="store_true", help="Start loopback 127.0.0.1 HTTP server after generating bundle")
    parser.add_argument("--port", type=int, default=8765, help="Loopback server port")
    args = parser.parse_args()

    src = Path(args.source)
    index_p = build_scene_preview_bundle(
        src,
        format_name=args.format_name,
        look_name=args.look,
        caption_style=args.caption_style,
    )
    print(f"Preview bundle: {index_p}")

    if args.scene:
        res = render_single_scene_preview(
            src,
            args.scene,
            format_name=args.format_name,
            look_name=args.look,
            caption_style=args.caption_style,
            backend=args.backend,
            with_tts=args.with_tts,
        )
        print(json.dumps(res, indent=2))

    if args.serve:
        srv, url = start_preview_server(index_p.parent, port=args.port)
        print(f"Serving preview at {url} (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            srv.shutdown()
            srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
