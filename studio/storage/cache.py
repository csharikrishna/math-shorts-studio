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
