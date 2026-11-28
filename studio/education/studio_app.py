"""Local K-12 AI Educational Studio web interface (strictly loopback 127.0.0.1)."""

from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from studio.education.planner import generate_k12_lesson_plan
from studio.education.profiles import CBSE_MATH_INDEX, LessonRequest
from studio.providers import get_provider
from studio.qa.engine import audit_educational_quality

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def list_output_library() -> list[dict[str, Any]]:
    """Scan output/ and build/ for completed, review-required, and draft videos + manifests."""
    items: list[dict[str, Any]] = []
    if OUTPUT_DIR.exists():
        for mp4 in sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:40]:
            man_p = mp4.with_suffix(".manifest.json")
            man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
            items.append(
                {
                    "name": mp4.name,
                    "path": str(mp4),
                    "size_mb": round(mp4.stat().st_size / (1024 * 1024), 2),
                    "topic": man.get("topic") or mp4.stem,
                    "production_state": man.get("production_state", "rendered"),
                    "qa_score": man.get("qa_score"),
                    "format": man.get("format", "widescreen_hd"),
                    "theme": man.get("theme", "light"),
                    "has_srt": mp4.with_suffix(".srt").exists(),
                }
            )
    return items


STUDIO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Math Shorts Studio — K-12 Educational Studio</title>
<style>
  :root { --bg:#f8fafc; --panel:#ffffff; --ink:#0f172a; --muted:#475569; --accent:#2563eb; --border:#cbd5e1; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family: system-ui, -apple-system, sans-serif; background:var(--bg); color:var(--ink); display:grid; grid-template-columns:360px 1fr 360px; height:100vh; overflow:hidden; }
  aside, main, section { overflow-y:auto; padding:18px; border-right:1px solid var(--border); }
  h1 { font-size:18px; margin-bottom:12px; }
  h2 { font-size:15px; margin:14px 0 8px; color:var(--accent); }
  label { display:block; font-size:12px; font-weight:600; color:var(--muted); margin:10px 0 4px; }
  input, select, textarea { width:100%; padding:8px 10px; border:1px solid var(--border); border-radius:8px; font-size:13px; background:#fff; }
  button { margin-top:12px; padding:10px 14px; border:none; border-radius:8px; background:var(--accent); color:#fff; font-weight:600; cursor:pointer; width:100%; }
  button.secondary { background:#0f172a; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px; margin-bottom:12px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; background:#dbeafe; color:#1e40af; margin-right:6px; }
  pre { background:#f1f5f9; padding:10px; border-radius:8px; font-size:12px; white-space:pre-wrap; margin-top:6px; }
  .lib-item { padding:10px; border:1px solid var(--border); border-radius:8px; margin-bottom:8px; background:#fff; }
</style>
</head>
<body>
<aside>
  <h1>K-12 Lesson Creator</h1>
  <label>Class / Grade (1–12)</label>
  <select id="grade"></select>
  <label>Curriculum / Board</label>
  <select id="board">
    <option value="CBSE/NCERT">CBSE / NCERT (India)</option>
    <option value="curriculum_neutral">Curriculum-Neutral</option>
    <option value="ICSE">ICSE</option>
    <option value="Common Core">Common Core</option>
  </select>
  <label>Subject</label>
  <select id="subject"><option value="Mathematics">Mathematics</option></select>
  <label>Topic</label>
  <input id="topic" value="Adding fractions with different denominators" />
  <label>Teaching Style</label>
  <select id="teaching_style">
    <option value="concrete_to_abstract">Visual &amp; Intuitive (Concrete to Abstract)</option>
    <option value="problem_and_solution">Step-by-Step Problem &amp; Solution</option>
    <option value="visual_proof">Visual Proof</option>
    <option value="story_based">Story-Based</option>
    <option value="question_and_answer">Question &amp; Answer</option>
    <option value="examination_revision">Exam Revision</option>
  </select>
  <label>Duration (minutes)</label>
  <select id="duration">
    <option value="1.0">Quick Demo (~1 min)</option>
    <option value="4.5">Standard Lesson (~4.5 min)</option>
    <option value="10.0">Detailed Lesson (~10 min)</option>
  </select>
  <label>Theme &amp; Format</label>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
    <select id="theme"><option value="light">Educational Light</option><option value="midnight_studio">Educational Dark</option></select>
    <select id="format_name"><option value="widescreen_hd">16:9 Landscape</option><option value="shorts_hd">9:16 Portrait</option></select>
  </div>
  <label>Teacher Notes (optional — sent to AI provider if used)</label>
  <textarea id="custom_notes" rows="3" placeholder="Optional chapter notes or focus points..."></textarea>
  <button onclick="generateLesson(false)">Generate Lesson Preview (Fast)</button>
  <button class="secondary" onclick="generateLesson(true)">Generate + Render Complete MP4</button>
  <div id="status-box" class="card" style="margin-top:12px;display:none;"></div>
</aside>
<main>
  <h1>Lesson Workspace</h1>
  <div id="workspace" class="card">Select a Class and Topic on the left, then click <strong>Generate Lesson Preview</strong>.</div>
</main>
<section>
  <h1>Output Library</h1>
  <div id="library">Loading...</div>
</section>
<script>
const gSel = document.getElementById("grade");
for (let g = 1; g <= 12; g++) {
  const o = document.createElement("option");
  o.value = String(g);
  o.textContent = "Class " + g;
  if (g === 5) o.selected = true;
  gSel.appendChild(o);
}
async function refreshLibrary() {
  const res = await fetch("/api/library");
  const data = await res.json();
  const el = document.getElementById("library");
  el.innerHTML = data.items.length ? "" : "<div class='card'>No rendered MP4s yet.</div>";
  data.items.forEach(it => {
    const d = document.createElement("div");
    d.className = "lib-item";
    d.innerHTML = `<strong>${it.topic}</strong><div style="font-size:12px;color:#475569;margin-top:4px;"><span class="badge">${it.production_state}</span> ${it.format} · ${it.size_mb} MB</div>`;
    el.appendChild(d);
  });
}
async function generateLesson(andRender) {
  const st = document.getElementById("status-box");
  st.style.display = "block";
  st.textContent = andRender ? "Generating lesson & rendering MP4..." : "Generating lesson plan & deterministic math...";
  const payload = {
    grade: Number(document.getElementById("grade").value),
    board: document.getElementById("board").value,
    subject: document.getElementById("subject").value,
    topic: document.getElementById("topic").value,
    teaching_style: document.getElementById("teaching_style").value,
    duration_minutes: Number(document.getElementById("duration").value),
    theme: document.getElementById("theme").value,
    format_name: document.getElementById("format_name").value,
    custom_notes: document.getElementById("custom_notes").value,
    render: andRender
  };
  const res = await fetch("/api/generate", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
  const data = await res.json();
  if (data.error) {
    st.textContent = "Error: " + data.error;
    return;
  }
  st.textContent = andRender ? ("Render complete: " + (data.video_path || "queued")) : "Lesson ready for review!";
  const ws = document.getElementById("workspace");
  ws.innerHTML = `
    <div class="card">
      <span class="badge">Class ${data.spec.request.grade} (Band ${data.spec.grade_profile.band})</span>
      <span class="badge">${data.spec.curriculum_ref.board}</span>
      <span class="badge">${data.spec.production_mode}</span>
      <h2>Learning Objectives</h2>
      <pre>${(data.spec.curriculum_ref.learning_objectives || []).join("\\n")}</pre>
      <h2>Verified Worked Example</h2>
      <pre>${data.worked.problem}\\nSteps:\\n- ${(data.worked.steps || []).join("\\n- ")}\\nAnswer: ${data.worked.final_answer}</pre>
      <h2>Practice Question</h2>
      <pre>${data.question.question}\\nChoices: ${(data.question.choices || []).join(" | ")}\\nAnswer: ${data.question.correct_answer}</pre>
      <h2>Storyboard & Narration (${data.plan.scenes.length} scenes)</h2>
      <pre>${data.plan.scenes.map(s => `[${s.scene_id}] ${s.purpose}\\n${s.narration}`).join("\\n\\n")}</pre>
    </div>`;
  refreshLibrary();
}
refreshLibrary();
</script>
</body>
</html>
"""


def create_studio_server(*, port: int = 8770, use_llm: bool = False) -> tuple[ThreadingHTTPServer, str]:
    """Start the local K-12 Educational Studio HTTP server strictly on 127.0.0.1."""

    class _StudioHandler(BaseHTTPRequestHandler):
        def _send_json(self, obj: Any, status: int = 200) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/library":
                self._send_json({"items": list_output_library()})
                return
            if path == "/api/curriculum":
                self._send_json({"entries": [v.model_dump() for v in CBSE_MATH_INDEX.values()]})
                return
            if path in ("/", "/index.html"):
                raw = STUDIO_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/api/generate":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                req = LessonRequest(
                    grade=int(body.get("grade", 5)),
                    board=str(body.get("board", "CBSE/NCERT")),
                    subject=str(body.get("subject", "Mathematics")),
                    topic=str(body.get("topic", "Adding fractions with different denominators")),
                    teaching_style=str(body.get("teaching_style", "concrete_to_abstract")),  # type: ignore[arg-type]
                    duration_minutes=float(body.get("duration_minutes", 1.5)),
                    theme=str(body.get("theme", "light")),
                    format_name=str(body.get("format_name", "widescreen_hd")),
                    custom_notes=str(body.get("custom_notes", "")),
                )
                prov = get_provider("gemini") if use_llm else None
                spec, plan, worked, question = generate_k12_lesson_plan(req, provider=prov)
                out_payload: dict[str, Any] = {
                    "spec": spec.model_dump(),
                    "worked": worked.model_dump(),
                    "question": question.model_dump(),
                    "plan": plan.model_dump(),
                }
                if body.get("render"):
                    from studio.pipeline import run_plan
                    import re as _re
                    slug = _re.sub(r"[^a-z0-9]+", "_", req.topic.lower()).strip("_")[:36]
                    stem = f"k12_c{req.grade}_{slug}"
                    p_dir = Path("build") / stem
                    p_dir.mkdir(parents=True, exist_ok=True)
                    p_json = p_dir / "plan.json"
                    p_json.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                    mp4 = run_plan(
                        p_json,
                        format_name=req.format_name,
                        theme_name=req.theme,
                        tts_engine="kokoro",
                        output_stem=stem,
                    )
                    out_payload["video_path"] = str(mp4)
                self._send_json(out_payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def log_message(self, format: str, *args: Any) -> None:
            return None

    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), _StudioHandler)
    actual_port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, name=f"k12-studio-{actual_port}", daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{actual_port}/"
