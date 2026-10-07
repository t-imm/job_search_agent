"""Episodic memory - the "what happened" tier.

Every scouting run and every set of recommendations is recorded as an episode.
This is what lets a later run answer "show me what I was shown last Tuesday"
and spot patterns ("I keep being shown pure frontend roles I never want").
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .base import EPISODIC, BaseMemory, MemoryConfig, MemoryItem

logger = logging.getLogger(__name__)


class EpisodicMemory(BaseMemory):
    """Append-only run history kept in its own SQLite file.

    Episodes are a chronological log, not a searchable knowledge base, so they
    use plain SQL rather than the vector store. The semantic tier handles
    similarity; this tier handles time.
    """

    def __init__(self, config: MemoryConfig, db_path: str):
        super().__init__(config)
        self.memory_type = EPISODIC
        import sqlite3

        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id           TEXT PRIMARY KEY,
                user_id      TEXT,
                kind         TEXT,
                session_id   TEXT,
                content      TEXT,
                metadata     TEXT,
                importance   REAL DEFAULT 0.5,
                created_at   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);
            CREATE INDEX IF NOT EXISTS idx_episodes_kind ON episodes(kind);
            """
        )
        self._conn.commit()

    def add(self, memory_item: MemoryItem) -> str:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO episodes
                (id, user_id, kind, session_id, content, metadata, importance, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                memory_item.id,
                memory_item.user_id,
                memory_item.metadata.get("kind", "event"),
                memory_item.metadata.get("session_id", ""),
                memory_item.content,
                json.dumps(memory_item.metadata, default=str),
                memory_item.importance,
                memory_item.timestamp.isoformat(),
            ),
        )
        self._conn.commit()
        return memory_item.id

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        kind: Optional[str] = None,
        **kwargs,
    ) -> List[MemoryItem]:
        """Recent-first lookup. Episodes are matched on `kind`, not by meaning."""
        sql = "SELECT * FROM episodes WHERE 1=1"
        params: List[Any] = []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_item(r) for r in rows]

    def recent(self, limit: int = 10, kind: Optional[str] = None) -> List[MemoryItem]:
        return self.retrieve("", limit=limit, kind=kind)

    def since(self, days: int) -> List[MemoryItem]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE created_at >= ? ORDER BY created_at DESC", (cutoff,)
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    @staticmethod
    def _row_to_item(row) -> MemoryItem:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        try:
            timestamp = datetime.fromisoformat(row["created_at"])
        except (TypeError, ValueError):
            timestamp = datetime.now(timezone.utc)
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            memory_type=EPISODIC,
            user_id=row["user_id"] or "default",
            timestamp=timestamp,
            importance=row["importance"] or 0.5,
            metadata=metadata,
        )

    def has_history(self, kind: Optional[str] = None) -> bool:
        if kind:
            row = self._conn.execute(
                "SELECT 1 FROM episodes WHERE kind = ? LIMIT 1", (kind,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT 1 FROM episodes LIMIT 1").fetchone()
        return row is not None

    def update(self, memory_id: str, content=None, importance=None, metadata=None) -> bool:
        current = next((i for i in self.get_all() if i.id == memory_id), None)
        if current is None:
            return False
        if content is not None:
            current.content = content
        if importance is not None:
            current.importance = importance
        if metadata:
            current.metadata.update(metadata)
        return self.add(current) == memory_id

    def remove(self, memory_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM episodes WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def has_memory(self, memory_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM episodes WHERE id = ?", (memory_id,)
        ).fetchone() is not None

    def clear(self) -> None:
        self._conn.execute("DELETE FROM episodes")
        self._conn.commit()

    def get_all(self) -> List[MemoryItem]:
        rows = self._conn.execute("SELECT * FROM episodes").fetchall()
        return [self._row_to_item(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()["n"]
        by_kind = {
            r["kind"]: r["n"]
            for r in self._conn.execute(
                "SELECT kind, COUNT(*) AS n FROM episodes GROUP BY kind"
            ).fetchall()
        }
        return {"count": total, "by_kind": by_kind, "type": EPISODIC}
