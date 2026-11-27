"""Headless Chromium CDP client, secure loopback static server, and process utilities."""

from __future__ import annotations

import functools
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any


def find_chromium_binary() -> str | None:
    """Locate local headless Chromium / Chrome / Edge binary (no remote downloads)."""
    env_p = os.environ.get("STUDIO_CHROME_BIN") or os.environ.get("CHROME_PATH")
    if env_p and Path(env_p).exists():
        return env_p
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("chrome"),
        shutil.which("msedge"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def _current_process_rss_mb() -> float:
    """Return current process peak/working set RSS in MB (Windows ctypes or POSIX fallback)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            hproc = k32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(hproc, ctypes.byref(counters), counters.cb):
                return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0


def validate_safe_relpath(root_dir: Path, raw_url_path: str) -> Path | None:
    """Path-traversal-safe resolver for local loopback static server."""
    clean = raw_url_path.split("?", 1)[0].split("#", 1)[0]
    try:
        decoded = urllib.parse.unquote(clean)
    except Exception:
        return None
    if "\x00" in decoded or "\\" in decoded:
        return None
    parts = [p for p in decoded.split("/") if p]
    if not parts:
        parts = ["index.html"]
    for part in parts:
        if part in (".", "..") or part.startswith("."):
            return None
    root_resolved = root_dir.resolve()
    candidate = (root_resolved.joinpath(*parts)).resolve()
    if not candidate.is_relative_to(root_resolved):
        return None
    return candidate


class _LoopbackHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, root_dir: Path, **kwargs: Any) -> None:
        self._root_dir = root_dir
        super().__init__(*args, directory=str(root_dir), **kwargs)

    def do_GET(self) -> None:
        safe = validate_safe_relpath(self._root_dir, self.path)
        if safe is None:
            self.send_error(403, "Forbidden path traversal")
            return
        if not safe.exists() or not safe.is_file():
            self.send_error(404, "Not found")
            return
        ctype = "text/html; charset=utf-8" if safe.suffix == ".html" else "application/json; charset=utf-8"
        if safe.suffix == ".svg":
            ctype = "image/svg+xml"
        elif safe.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        elif safe.suffix == ".ttf":
            ctype = "font/ttf"
        elif safe.suffix == ".otf":
            ctype = "font/otf"
        data = safe.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _start_loopback_server(root_dir: Path) -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(_LoopbackHandler, root_dir=root_dir)
    socketserver.TCPServer.allow_reuse_address = True
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = int(server.server_address[1])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _launch_chrome_cdp(url: str, width: int, height: int, user_data_dir: Path) -> tuple[subprocess.Popen[str], str]:
    chrome = find_chromium_binary()
    if not chrome:
        raise RuntimeError("No local Chrome/Chromium/Edge binary found on system for HyperFramesBackend.")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-extensions",
        "--mute-audio",
        f"--user-data-dir={user_data_dir}",
        "--remote-debugging-port=0",
        f"--window-size={width},{height}",
        "about:blank",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ws_url = ""
    deadline = time.time() + 15.0
    port_file = user_data_dir / "DevToolsActivePort"
    while time.time() < deadline:
        if port_file.exists():
            lines = [ln.strip() for ln in port_file.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            if len(lines) >= 2:
                ws_url = f"ws://127.0.0.1:{lines[0]}{lines[1]}"
                break
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Chromium exited early ({proc.returncode}): {err}")
        time.sleep(0.05)

    if not ws_url:
        proc.terminate()
        raise RuntimeError("Timed out waiting for Chromium DevToolsActivePort.")

    import urllib.request

    port = ws_url.split("://127.0.0.1:", 1)[1].split("/", 1)[0]
    pages_json = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5).read().decode("utf-8")
    pages = json.loads(pages_json)
    page_ws = next((p["webSocketDebuggerUrl"] for p in pages if p.get("type") == "page" and "webSocketDebuggerUrl" in p), "")
    if not page_ws:
        proc.terminate()
        raise RuntimeError("No page WebSocket debugger URL exposed by Chromium.")
    return proc, page_ws


class _CDPClient:
    """Minimal synchronous CDP WebSocket client using `websockets`."""

    def __init__(self, ws_url: str) -> None:
        from websockets.sync.client import connect

        self._ws = connect(ws_url, max_size=32 * 1024 * 1024)
        self._msg_id = 0

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._msg_id += 1
        cid = self._msg_id
        self._ws.send(json.dumps({"id": cid, "method": method, "params": params or {}}))
        while True:
            raw = self._ws.recv()
            msg = json.loads(raw)
            if msg.get("method") == "Fetch.requestPaused":
                rid = msg["params"]["requestId"]
                req_url = str(msg["params"].get("request", {}).get("url", ""))
                allow = req_url.startswith(("http://127.0.0.1", "http://localhost", "data:", "blob:", "about:"))
                self._msg_id += 1
                if allow:
                    self._ws.send(json.dumps({"id": self._msg_id, "method": "Fetch.continueRequest", "params": {"requestId": rid}}))
                else:
                    self._ws.send(json.dumps({"id": self._msg_id, "method": "Fetch.failRequest", "params": {"requestId": rid, "errorReason": "BlockedByClient"}}))
                continue
            if msg.get("id") == cid:
                if "error" in msg:
                    raise RuntimeError(f"CDP {method} failed: {msg['error']}")
                return msg.get("result", {})

    def close(self) -> None:
        self._ws.close()
