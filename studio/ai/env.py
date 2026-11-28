"""Minimal, dependency-free .env loader for API credentials (values are never logged)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api", "GEMINI_API")
NVIDIA_KEY_NAMES = ("NVIDIA_API_KEY", "NVAPI_KEY", "nvidia_api_key")


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, val = line.partition("=")
        val = val.strip().strip("'\"")
        values[key.strip()] = val
    return values


def _resolve_env_key(names: tuple[str, ...], error_msg: str) -> str:
    """Resolve an API key from process env first, then the project's .env file."""
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    dot = _read_dotenv(PROJECT_ROOT / ".env")
    for name in names:
        if dot.get(name):
            return dot[name]
    raise RuntimeError(error_msg)


def gemini_api_key() -> str:
    """Resolve the Gemini key from the process env, then the project's .env file."""
    return _resolve_env_key(
        KEY_NAMES,
        "No Gemini API key found. Set GEMINI_API_KEY (or gemini_api) in the environment or in .env.",
    )


def nvidia_api_key() -> str:
    """Resolve the NVIDIA NIM API key from the process env, then the project's .env file."""
    return _resolve_env_key(
        NVIDIA_KEY_NAMES,
        "No NVIDIA API key found. Set NVIDIA_API_KEY in the environment or in .env.",
    )
