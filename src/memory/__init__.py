"""Memory layer for the job-search agent.

Three tiers, following the HelloAgents reference architecture:

    SemanticMemory  vector + graph knowledge (Qdrant, Neo4j)
    EpisodicMemory  run history and recommendations log (SQLite)
    WorkingMemory   per-run scratchpad (in-process, TTL-bounded)

`MemoryManager` wires the three together and is the only object the rest of
the application talks to.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import EPISODIC, SEMANTIC, WORKING, BaseMemory, MemoryConfig, MemoryItem, utcnow
from .embedding import get_embedder
from .episodic import EpisodicMemory
from .semantic import JOBS_NS, PROFILE_NS, SemanticMemory
from .storage.neo4j_store import Neo4jStore
from .storage.qdrant_store import QdrantStore
from .working import WorkingMemory

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryManager",
    "MemoryItem",
    "MemoryConfig",
    "SemanticMemory",
    "EpisodicMemory",
    "WorkingMemory",
    "QdrantStore",
    "Neo4jStore",
    "PROFILE_NS",
    "JOBS_NS",
    "get_embedder",
]


class MemoryManager:
    """Facade over the three memory tiers.

    Backends degrade gracefully: if Neo4j is down the agent still runs with
    vector-only retrieval, because the graph is an enrichment rather than a
    source of truth.
    """

    def __init__(
        self,
        storage_path: str = "./memory_data",
        user_id: str = "job_seeker",
        config: Optional[MemoryConfig] = None,
        with_graph: bool = True,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id
        self.config = config or MemoryConfig(storage_path=str(self.storage_path))

        # The relational system of record: jobs, applications, search cache,
        # ingest log. Everything between the Scout and the report reads this.
        from ..domain.sqlite_store import SQLiteStore

        self.sqlite = SQLiteStore(self.storage_path / "jobs.db")

        self.vector_store = QdrantStore(vector_size=get_embedder().dimension)
        self.graph_store: Optional[Neo4jStore] = None
        if with_graph:
            try:
                self.graph_store = Neo4jStore()
            except Exception as exc:
                logger.warning("Neo4j unavailable, continuing without graph layer: %s", exc)

        self.semantic = SemanticMemory(self.config, self.vector_store, self.graph_store)
        self.episodic = EpisodicMemory(self.config, str(self.storage_path / "episodes.db"))
        self.working = WorkingMemory(self.config)

        logger.info("MemoryManager ready (user=%s, graph=%s)", user_id, self.graph_store is not None)

    # ------------------------------------------------------------- profile IO

    def add_profile_chunks(self, chunks: List[Dict[str, Any]], source: str) -> List[str]:
        """Store resume/preference sections in the `profile` namespace."""
        items = [
            MemoryItem(
                content=chunk["content"],
                memory_type=SEMANTIC,
                user_id=self.user_id,
                importance=float(chunk.get("importance", 0.7)),
                metadata={
                    "namespace": PROFILE_NS,
                    "source": source,
                    "section": chunk.get("section", ""),
                    "kind": chunk.get("kind", "profile"),
                    # Preserve document order so the profile can be rebuilt
                    # top-down rather than alphabetically.
                    "order": int(chunk.get("order", 0)),
                },
            )
            for chunk in chunks
            if chunk.get("content", "").strip()
        ]
        return self.semantic.add_many(items)

    def search_profile(self, query: str, limit: int = 8) -> List[MemoryItem]:
        return self.semantic.retrieve_profile(query, limit=limit)

    # ---------------------------------------------------------------- job IO

    def add_job(self, job, importance: float = 0.5) -> str:
        """Store a posting as a semantic memory and mirror it into the graph."""
        item = MemoryItem(
            content=job.text_for_matching,
            memory_type=SEMANTIC,
            user_id=self.user_id,
            importance=importance,
            metadata={
                "namespace": JOBS_NS,
                "kind": "job",
                "job_id": job.job_id,
                "company": job.company,
                "title": job.title,
                "url": job.url,
                "location": job.location,
                "status": job.status,
                "remote": job.remote,
                "source": job.source,
            },
        )
        memory_id = self.semantic.add(item)
        self.semantic.index_job_graph(job)
        return memory_id

    def search_jobs(self, query: str, limit: int = 20) -> List[MemoryItem]:
        return self.semantic.retrieve_jobs(query, limit=limit)

    def index_job_graph(self, job) -> None:
        self.semantic.index_job_graph(job)

    # ----------------------------------------------------------- event log

    def record_event(self, kind: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.episodic.add(
            MemoryItem(
                content=content,
                memory_type=EPISODIC,
                user_id=self.user_id,
                importance=0.6,
                metadata={"kind": kind, **(metadata or {})},
            )
        )

    def recent_events(self, kind: Optional[str] = None, limit: int = 20) -> List[MemoryItem]:
        return self.episodic.recent(limit=limit, kind=kind)

    def has_history(self, kind: Optional[str] = None) -> bool:
        return self.episodic.has_history(kind)

    # ------------------------------------------------------------- scratchpad

    def set_scratch(self, key: str, value: str) -> None:
        self.working.set(key, value)

    def get_scratch(self, key: str, default: Any = None) -> Any:
        return self.working.get(key, default)

    # ----------------------------------------------------------------- admin

    def get_stats(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "semantic": self.semantic.get_stats(),
            "episodic": self.episodic.get_stats(),
            "working": self.working.get_stats(),
        }

    def clear_all(self) -> None:
        self.semantic.clear()
        self.episodic.clear()
        self.working.clear()
