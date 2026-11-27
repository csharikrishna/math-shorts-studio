"""SQLite job persistence and content-addressed artifact caching."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class JobStore:
    """Lightweight SQLite store for job checkpoints, resume state, and content-addressed cache."""

    def __init__(self, db_path: Path = Path("build/studio_jobs.db")) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                artifact_path TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self._conn.commit()

    @staticmethod
    def content_hash(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def upsert_job(self, job_id: str, topic: str, stage: str, status: str, meta: dict | None = None) -> None:
        self._conn.execute(
            """INSERT INTO jobs (job_id, topic, stage, status, meta_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(job_id) DO UPDATE SET
                 stage=excluded.stage, status=excluded.status,
                 meta_json=excluded.meta_json, updated_at=excluded.updated_at""",
            (job_id, topic, stage, status, json.dumps(meta or {}), time.time()),
        )
        self._conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT job_id, topic, stage, status, meta_json, updated_at FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if not row:
            return None
        return {"job_id": row[0], "topic": row[1], "stage": row[2], "status": row[3], "meta": json.loads(row[4]), "updated_at": row[5]}

    def get_cached(self, cache_key: str) -> Path | None:
        row = self._conn.execute("SELECT artifact_path FROM cache WHERE cache_key = ?", (cache_key,)).fetchone()
        if not row:
            return None
        p = Path(row[0])
        return p if p.exists() else None

    def put_cached(self, cache_key: str, artifact_path: Path) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, artifact_path, created_at) VALUES (?, ?, ?)",
            (cache_key, str(artifact_path), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


CACHE_STAGES: tuple[str, ...] = (
    "research",
    "knowledge",
    "script",
    "storyboard",
    "design",
    "math_model",
    "tts",
    "captions",
    "scene_render",
    "chapter_assembly",
)


class ArtifactCache:
    """Multi-stage dependency-aware artifact cache with SHA-256 integrity checks."""

    def __init__(self, store: JobStore | None = None, db_path: Path = Path("build/studio_jobs.db")) -> None:
        self._owns_store = store is None
        self.store = store or JobStore(db_path)
        self.store._conn.execute(
            """CREATE TABLE IF NOT EXISTS stage_cache (
                stage TEXT NOT NULL,
                logical_id TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                file_sha256 TEXT NOT NULL,
                upstream_keys_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (stage, logical_id)
            )"""
        )
        self.store._conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_stage_key(
        stage: str,
        payload: dict[str, Any],
        upstream_keys: dict[str, str] | None = None,
    ) -> str:
        """Compute deterministic content+dependency key sensitive to upstream stages and source code."""
        if stage not in CACHE_STAGES:
            raise ValueError(f"Unknown cache stage {stage!r}.")
        return JobStore.content_hash(
            {
                "stage": stage,
                "upstream": upstream_keys or {},
                "payload": payload,
            }
        )

    def lookup(
        self,
        stage: str,
        logical_id: str,
        payload: dict[str, Any],
        upstream_keys: dict[str, str] | None = None,
    ) -> tuple[bool, str, Path | None, str]:
        """Check whether an exact dependency-verified artifact is valid on disk."""
        expected_key = self.compute_stage_key(stage, payload, upstream_keys=upstream_keys)
        row = self.store._conn.execute(
            "SELECT cache_key, artifact_path, file_sha256 FROM stage_cache WHERE stage = ? AND logical_id = ?",
            (stage, logical_id),
        ).fetchone()
        if not row:
            self.misses += 1
            return False, expected_key, None, "miss:not_recorded"
        rec_key, rec_path_str, rec_sha = row
        if rec_key != expected_key:
            self.misses += 1
            return False, expected_key, None, "miss:dependency_or_payload_changed"
        p = Path(rec_path_str)
        if not p.exists() or p.stat().st_size == 0:
            self.misses += 1
            return False, expected_key, None, "miss:file_missing_or_empty"
        if self.file_sha256(p) != rec_sha:
            self.misses += 1
            return False, expected_key, None, "miss:checksum_mismatch"
        self.hits += 1
        return True, expected_key, p, "hit"

    def store_artifact(
        self,
        stage: str,
        logical_id: str,
        cache_key: str,
        artifact_path: Path,
        upstream_keys: dict[str, str] | None = None,
    ) -> str:
        """Record produced artifact alongside its SHA-256 checksum and upstream dependencies."""
        p = Path(artifact_path)
        if not p.exists():
            raise FileNotFoundError(f"Cannot cache missing artifact: {p}")
        sha = self.file_sha256(p)
        self.store._conn.execute(
            """INSERT OR REPLACE INTO stage_cache
               (stage, logical_id, cache_key, artifact_path, file_sha256, upstream_keys_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (stage, logical_id, cache_key, str(p), sha, json.dumps(upstream_keys or {}, sort_keys=True), time.time()),
        )
        self.store._conn.commit()
        return sha

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        if self._owns_store:
            self.store.close()


DEPENDENCY_GROUPS: dict[str, tuple[str, ...]] = {
    "script_gen": (
        "studio/creative/director.py",
        "studio/creative/script_ir.py",
        "studio/creative/dialogue.py",
        "studio/creative/models.py",
        "studio/ai/director.py",
    ),
    "math_compute": (
        "studio/ai/world_model.py",
        "studio/ai/math_truth.py",
        "studio/ai/expr.py",
    ),
    "math_validate": (
        "studio/ai/validate_v3.py",
        "studio/ai/v3_schema.py",
    ),
    "captions": (
        "studio/captions/model.py",
        "studio/captions/layout.py",
        "studio/captions/ass.py",
        "studio/timeline.py",
    ),
    "tts_voice": (
        "studio/audio/tts.py",
        "studio/audio/pronounce.py",
    ),
    "design_tokens": (
        "studio/design/spec.py",
        "studio/config.py",
    ),
    "fonts": (
        "studio/design/fonts.py",
    ),
    "components": (
        "studio/registry/__init__.py",
        "studio/registry/base.py",
        "studio/registry/math_ops.py",
        "studio/registry/visuals.py",
        "studio/registry/animations.py",
        "studio/registry/compositions.py",
    ),
    "manimgx_backend": (
        "studio/renderers/manimgx_backend.py",
        "scenes/v3_scene.py",
        "studio/layout/components.py",
        "studio/scene.py",
    ),
    "hyperframes_backend": (
        "studio/renderers/hyperframes_backend.py",
        "studio/renderers/hyperframes/compiler.py",
        "studio/renderers/hyperframes/components.py",
        "studio/renderers/hyperframes/runtime.py",
        "studio/renderers/hyperframes/captions.py",
    ),
    "layout_solver": (
        "studio/layout/solver.py",
    ),
    "audio_master": (
        "studio/audio/bgm.py",
        "studio/audio/sfx.py",
        "studio/audio/master.py",
    ),
}


def compute_dependency_fingerprints(project_root: Path | None = None) -> dict[str, str]:
    """Compute per-subsystem SHA-256 digests so theme/caption edits reuse math & TTS cleanly."""
    root = project_root or Path(__file__).resolve().parent.parent.parent
    out: dict[str, str] = {}
    for group, rel_paths in DEPENDENCY_GROUPS.items():
        h = hashlib.sha256()
        for rel in rel_paths:
            p = root / rel
            if p.exists():
                h.update(rel.encode("utf-8"))
                h.update(p.read_bytes())
        if group == "fonts":
            font_dir = root / "assets" / "fonts"
            if font_dir.exists():
                for fp in sorted(font_dir.glob("*")):
                    if fp.is_file():
                        h.update(fp.name.encode("utf-8"))
                        h.update(fp.read_bytes())
        out[group] = h.hexdigest()[:12]
    return out


def validate_sidecar_bundle(sidecars: list[Path]) -> tuple[bool, str]:
    """Verify sidecar artifacts (JSON/SRT/VTT/ASS/TXT) exist, are non-empty, and parse cleanly."""
    for sp in sidecars:
        if not sp.exists() or sp.stat().st_size == 0:
            return False, f"missing_or_empty_sidecar:{sp.name}"
        if sp.suffix.lower() == ".json":
            try:
                json.loads(sp.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                return False, f"corrupt_json_sidecar:{sp.name}:{exc}"
    return True, "sidecars_valid"
