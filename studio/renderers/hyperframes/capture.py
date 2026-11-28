"""Deterministic static-skipping Chromium frame capture loop for HyperFrames."""

from __future__ import annotations

import base64
import subprocess
import time
from pathlib import Path

from studio.audio.tts import AudioTimeline
from studio.config import FormatProfile
from studio.renderers.hyperframes.browser import _CDPClient, _launch_chrome_cdp


def capture_frames_to_ffmpeg(
    *,
    port: int,
    prof: FormatProfile,
    audio_timeline: AudioTimeline,
    ff_cmd: list[str],
    tmp_profile: Path,
) -> tuple[int, int, list[str]]:
    """Run deterministic __HF_STEP capture over CDP, piping JPEG frames directly into FFmpeg."""
    fps = prof.fps
    total_frames = max(1, int((audio_timeline.total_duration * fps) + 0.9999) + 2)
    layout_issues: list[str] = []
    proc: subprocess.Popen[str] | None = None
    cdp: _CDPClient | None = None
    ff_proc: subprocess.Popen[bytes] | None = None
    captured_frames = 0

    try:
        url = f"http://127.0.0.1:{port}/index.html"
        proc, page_ws = _launch_chrome_cdp(url, prof.pixel_width, prof.pixel_height, tmp_profile)
        cdp = _CDPClient(page_ws)
        cdp.call("Fetch.enable", {"patterns": [{"urlPattern": "*"}]})
        cdp.call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": prof.pixel_width,
                "height": prof.pixel_height,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        cdp.call("Page.enable")
        cdp.call("Page.navigate", {"url": url})
        ready = False
        deadline = time.perf_counter() + 2.5
        while time.perf_counter() < deadline:
            chk = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": "document.readyState === 'complete' && typeof window.__HF_STEP === 'function'",
                    "returnByValue": True,
                },
            )
            if bool(chk.get("result", {}).get("value")):
                ready = True
                break
            time.sleep(0.02)
        if not ready:
            cdp.call("Runtime.evaluate", {"expression": "typeof window.__HF_STEP === 'function'"})

        ff_proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audit_sorted = sorted(int(round(0.5 * (sb.start_time + sb.end_time) * fps)) for sb in audio_timeline.beats)
        audit_ptr = 0
        seen_issues: set[str] = set()
        last_jpeg: bytes | None = None

        f_idx = 0
        while f_idx < total_frames:
            while audit_ptr < len(audit_sorted) and audit_sorted[audit_ptr] < f_idx:
                audit_ptr += 1
            next_audit = audit_sorted[audit_ptr] if audit_ptr < len(audit_sorted) else -1
            max_run = min(240, total_frames - f_idx)
            step = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": f"window.__HF_STEP_BATCH({f_idx}, {max_run}, {fps}, {next_audit})",
                    "returnByValue": True,
                },
            )
            val = step.get("result", {}).get("value") or {}
            for v in val.get("issues") or []:
                if v not in seen_issues:
                    seen_issues.add(v)
                    layout_issues.append(str(v))
            if bool(val.get("changed", True)) or last_jpeg is None:
                shot = cdp.call("Page.captureScreenshot", {"format": "jpeg", "quality": 92})
                last_jpeg = base64.b64decode(shot["data"])
                captured_frames += 1
            run_len = max(1, min(max_run, int(val.get("runLen", 1) or 1)))
            assert ff_proc.stdin is not None
            ff_proc.stdin.write(last_jpeg * run_len)
            f_idx += run_len

        if ff_proc.stdin:
            ff_proc.stdin.close()
        _, ff_err = ff_proc.communicate(timeout=180)
        if ff_proc.returncode != 0:
            raise RuntimeError(f"FFmpeg single-pass master failed: {ff_err.decode('utf-8', errors='ignore')}")
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except Exception:
                pass
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    return captured_frames, total_frames, layout_issues
