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
