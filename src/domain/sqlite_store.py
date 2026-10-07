"""SQLite system of record.

SQLite is the backbone the whole pipeline reads from. Everything between the
Scout and the report touches only this file, which is what keeps the cost and
failure surface small: a Tavily outage degrades to stale cache, never to a
broken report.

Tables
------
jobs           every posting ever seen, plus the user's status on it
companies      company metadata gathered alongside postings
applications   the application log (what was applied to, and when)
search_cache   cached Tavily responses keyed by query hash
ingest_log     which source files have been loaded into memory
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    STATUS_APPLIED,
    JobPosting,
    normalise_url,
    utcnow_iso,
)

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    title        TEXT NOT NULL,
    company      TEXT NOT NULL,
    location     TEXT DEFAULT '',
    description  TEXT DEFAULT '',
    requirements TEXT DEFAULT '[]',
    seniority    TEXT DEFAULT '',
    remote       INTEGER DEFAULT 0,
    salary       TEXT DEFAULT '',
    posted_at    TEXT DEFAULT '',
    source       TEXT DEFAULT 'tavily',
    discovered_at TEXT,
    updated_at   TEXT,
    status       TEXT DEFAULT 'discovered',
    match_score  REAL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_url ON jobs(url);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);

CREATE TABLE IF NOT EXISTS companies (
    name         TEXT PRIMARY KEY,
    description  TEXT DEFAULT '',
    url          TEXT DEFAULT '',
    industry     TEXT DEFAULT '',
    size         TEXT DEFAULT '',
    location     TEXT DEFAULT '',
    notes        TEXT DEFAULT '',
    first_seen   TEXT,
    last_seen    TEXT
);

CREATE TABLE IF NOT EXISTS applications (
    job_id       TEXT,
    company      TEXT,
    title        TEXT,
    url          TEXT,
    status       TEXT,
    applied_at   TEXT,
    notes        TEXT DEFAULT '',
    PRIMARY KEY (job_id, status)
);

CREATE TABLE IF NOT EXISTS search_cache (
    query_hash   TEXT PRIMARY KEY,
    query        TEXT,
    payload      TEXT,
    created_at   TEXT
);

CREATE TABLE IF NOT EXISTS ingest_log (
    source_path  TEXT PRIMARY KEY,
    content_hash TEXT,
    chunks       INTEGER DEFAULT 0,
    ingested_at  TEXT
);
"""


class SQLiteStore:
    """Thread-safe-enough SQLite wrapper (one connection guarded by a lock)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _cursor(self):
        with self._lock:
            cursor = self._conn.cursor()
            try:
                yield cursor
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cursor.close()

    # ------------------------------------------------------------------- jobs

    def upsert_job(self, job: JobPosting) -> str:
        """Insert or refresh a posting. Returns 'new', 'updated' or 'duplicate'.

        Existing user status is preserved on refresh: re-discovering a posting
        the user already applied to must not silently reset it to 'discovered'.
        """
        now = utcnow_iso()
        with self._cursor() as cur:
            cur.execute("SELECT job_id, status FROM jobs WHERE job_id = ?", (job.job_id,))
            row = cur.fetchone()

            if row is None:
                # Same URL reposted under a new id -> keep the old row.
                cur.execute(
                    "SELECT job_id FROM jobs WHERE url = ? AND job_id != ?",
                    (normalise_url(job.url), job.job_id),
                )
                url_row = cur.fetchone()
                if url_row:
                    return "duplicate"

                cur.execute(
                    """
                    INSERT INTO jobs (job_id, url, title, company, location, description,
                                      requirements, seniority, remote, salary, posted_at,
                                      source, discovered_at, updated_at, status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        job.job_id, normalise_url(job.url), job.title, job.company, job.location,
                        job.description, json.dumps(job.requirements), job.seniority,
                        int(job.remote), job.salary, job.posted_at, job.source,
                        job.discovered_at or now, now, job.status,
                    ),
                )
                return "new"

            # Refresh content, keep status and original discovered_at.
            cur.execute(
                """
                UPDATE jobs
                   SET url = ?, title = ?, company = ?, location = ?, description = ?,
                       requirements = ?, seniority = ?, remote = ?, salary = ?,
                       posted_at = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (
                    normalise_url(job.url), job.title, job.company, job.location,
                    job.description, json.dumps(job.requirements), job.seniority,
                    int(job.remote), job.salary, job.posted_at, now, job.job_id,
                ),
            )
            return "updated"

    def upsert_jobs(self, jobs: Iterable[JobPosting]) -> Dict[str, int]:
        counts = {"new": 0, "updated": 0, "duplicate": 0}
        for job in jobs:
            counts[self.upsert_job(job)] += 1
        return counts

    def get_job(self, job_id: str) -> Optional[JobPosting]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
        return _row_to_job(row) if row else None

    def all_jobs(self, include_closed: bool = True) -> List[JobPosting]:
        query = "SELECT * FROM jobs"
        if not include_closed:
            query += " WHERE status NOT IN ('applied','rejected','dismissed')"
        with self._cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        return [_row_to_job(r) for r in rows]

    def recent_jobs(self, days: int = 7) -> List[JobPosting]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM jobs WHERE discovered_at >= ? ORDER BY discovered_at DESC",
                (cutoff,),
            )
            rows = cur.fetchall()
        return [_row_to_job(r) for r in rows]

    def set_status(self, job_id: str, status: str) -> bool:
        with self._cursor() as cur:
            cur.execute("UPDATE jobs SET status = ? WHERE job_id = ?", (status, job_id))
            changed = cur.rowcount > 0
            if changed:
                cur.execute(
                    "SELECT company, title, url FROM jobs WHERE job_id = ?", (job_id,)
                )
                row = cur.fetchone()
                if row:
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO applications
                            (job_id, company, title, url, status, applied_at, notes)
                        VALUES (?,?,?,?,?,?,'')
                        """,
                        (job_id, row["company"], row["title"], row["url"], status, utcnow_iso()),
                    )
        return changed

    def set_match_score(self, job_id: str, score: float) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE jobs SET match_score = ? WHERE job_id = ?", (score, job_id))

    # -------------------------------------------------------------- companies

    def upsert_company(self, name: str, **fields: Any) -> None:
        now = utcnow_iso()
        allowed = {"description", "url", "industry", "size", "location", "notes"}
        values = {k: v for k, v in fields.items() if k in allowed and v}
        with self._cursor() as cur:
            cur.execute("SELECT name FROM companies WHERE name = ?", (name,))
            exists = cur.fetchone() is not None
            if exists:
                if values:
                    assignments = ", ".join(f"{k} = ?" for k in values)
                    cur.execute(
                        f"UPDATE companies SET {assignments}, last_seen = ? WHERE name = ?",
                        (*values.values(), now, name),
                    )
                else:
                    cur.execute("UPDATE companies SET last_seen = ? WHERE name = ?", (now, name))
            else:
                cur.execute(
                    """
                    INSERT INTO companies (name, description, url, industry, size, location,
                                           notes, first_seen, last_seen)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        name,
                        values.get("description", ""), values.get("url", ""),
                        values.get("industry", ""), values.get("size", ""),
                        values.get("location", ""), values.get("notes", ""),
                        now, now,
                    ),
                )

    def get_company(self, name: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM companies WHERE name = ?", (name,))
            row = cur.fetchone()
        return dict(row) if row else None

    def all_companies(self) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM companies ORDER BY name")
            return [dict(r) for r in cur.fetchall()]

    # ---------------------------------------------------------- search cache

    @staticmethod
    def hash_query(query: str, depth: str = "basic") -> str:
        return hashlib.sha256(f"{query.strip().lower()}|{depth}".encode()).hexdigest()

    def get_cached_search(self, query: str, depth: str = "basic", ttl_hours: int = 24) -> Optional[Dict[str, Any]]:
        key = self.hash_query(query, depth)
        with self._cursor() as cur:
            cur.execute("SELECT payload, created_at FROM search_cache WHERE query_hash = ?", (key,))
            row = cur.fetchone()
        if not row:
            return None
        created = datetime.fromisoformat(row["created_at"])
        if datetime.now(timezone.utc) - created > timedelta(hours=ttl_hours):
            logger.info("Search cache expired for %r", query)
            return None
        return json.loads(row["payload"])

    def set_cached_search(self, query: str, depth: str, payload: Dict[str, Any]) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO search_cache (query_hash, query, payload, created_at)
                VALUES (?,?,?,?)
                """,
                (self.hash_query(query, depth), query, json.dumps(payload), utcnow_iso()),
            )

    # ------------------------------------------------------------ ingest log

    def needs_ingest(self, source_path: str, content: str) -> bool:
        digest = hashlib.sha256(content.encode()).hexdigest()
        with self._cursor() as cur:
            cur.execute(
                "SELECT content_hash FROM ingest_log WHERE source_path = ?", (source_path,)
            )
            row = cur.fetchone()
        return not row or row["content_hash"] != digest

    def record_ingest(self, source_path: str, content: str, chunks: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO ingest_log (source_path, content_hash, chunks, ingested_at)
                VALUES (?,?,?,?)
                """,
                (source_path, hashlib.sha256(content.encode()).hexdigest(), chunks, utcnow_iso()),
            )

    # ----------------------------------------------------------------- stats

    def stats(self) -> Dict[str, Any]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT status, COUNT(*) AS n FROM jobs GROUP BY status
                """
            )
            by_status = {r["status"]: r["n"] for r in cur.fetchall()}
            cur.execute("SELECT COUNT(*) AS n FROM companies")
            companies = cur.fetchone()["n"]
        return {
            "jobs_total": sum(by_status.values()),
            "jobs_by_status": by_status,
            "companies": companies,
            "open_jobs": sum(
                n for s, n in by_status.items() if s not in ("applied", "rejected", "dismissed")
            ),
        }


def _row_to_job(row: sqlite3.Row) -> JobPosting:
    try:
        requirements = json.loads(row["requirements"] or "[]")
    except (json.JSONDecodeError, TypeError):
        requirements = []
    return JobPosting(
        job_id=row["job_id"],
        title=row["title"],
        company=row["company"],
        url=row["url"],
        location=row["location"] or "",
        description=row["description"] or "",
        requirements=requirements,
        seniority=row["seniority"] or "",
        remote=bool(row["remote"]),
        salary=row["salary"] or "",
        posted_at=row["posted_at"] or "",
        source=row["source"] or "tavily",
        discovered_at=row["discovered_at"] or "",
        status=row["status"] or "discovered",
    )
