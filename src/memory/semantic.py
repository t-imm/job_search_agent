"""Semantic memory - the long-term knowledge tier backed by Qdrant + Neo4j.

Holds two kinds of content:
  * the user's profile (resume sections, preferences) under namespace "profile"
  * every discovered posting under namespace "jobs"

Both live in the same collection and are filtered by namespace at query time,
so a search for "what did I do at HKT" never returns a job posting.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import SEMANTIC, BaseMemory, MemoryConfig, MemoryItem
from .embedding import get_embedder
from .storage.neo4j_store import Neo4jStore
from .storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

PROFILE_NS = "profile"
JOBS_NS = "jobs"

#: Technologies recognised as graph skills. Kept local to avoid importing the
#: matcher (agent layer) from the memory layer.
_SKILL_VOCAB = (
    "python", "java", "javascript", "typescript", "golang", "rust", "c++", "c#",
    "kotlin", "swift", "scala", "php", "ruby", "r", "matlab", "sql", "bash",
    "react", "react.js", "vue", "vue.js", "angular", "next.js", "svelte",
    "node.js", "express", "django", "flask", "fastapi", "spring", "spring boot",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "jenkins", "ci/cd", "git", "linux", "unix", "windows",
    "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
    "kafka", "rabbitmq", "graphql", "rest", "grpc", "microservices",
    "machine learning", "deep learning", "computer vision", "nlp",
    "artificial intelligence", "pytorch", "tensorflow", "keras", "pandas",
    "numpy", "opencv", "yolo", "llm", "rag", "langchain",
    "networking", "wireless", "5g", "iot", "embedded systems", "tcp/ip",
    "distributed systems", "system design", "algorithms", "data structures",
    "power automate", "automation", "devops", "sre", "linux", "raspberry pi",
    "html", "css", "power bi", "tableau", "etl",
)


def _skill_names(job, limit: int = 20) -> List[str]:
    """Extract normalised technology names from a posting's text.

    Multi-word terms are matched first so "spring boot" wins over "spring".
    """
    haystack = f" {job.text_for_matching.lower()} "
    found: List[str] = []
    seen = set()
    for term in sorted(_SKILL_VOCAB, key=len, reverse=True):
        if term in seen:
            continue
        if term in haystack:
            seen.add(term)
            found.append(term)
            if len(found) >= limit:
                break
    return found


class SemanticMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, vector_store: QdrantStore, graph_store: Neo4jStore | None = None):
        super().__init__(config)
        self.memory_type = SEMANTIC
        self.embedder = get_embedder()
        # Trust the embedder's real dimension over the .env value.
        self.vector_store = vector_store
        if self.vector_store.vector_size != self.embedder.dimension:
            logger.warning(
                "Qdrant collection dim=%s but embedder produces %s",
                self.vector_store.vector_size, self.embedder.dimension,
            )
        self.graph_store = graph_store
        self._local: Dict[str, MemoryItem] = {}

    # ------------------------------------------------------------------ writes

    def add(self, memory_item: MemoryItem) -> str:
        self.add_many([memory_item])
        return memory_item.id

    def add_many(self, items: List[MemoryItem]) -> List[str]:
        """Embed and upsert in one batch - one embedding round-trip per call."""
        if not items:
            return []

        namespaces = {item.metadata.get("namespace", "default") for item in items}
        if len(namespaces) > 1:
            # Mixed namespaces need separate upserts to keep the filter clean.
            groups: Dict[str, List[MemoryItem]] = {}
            for item in items:
                groups.setdefault(item.metadata.get("namespace", "default"), []).append(item)
            written: List[str] = []
            for namespace, group in groups.items():
                written.extend(self._upsert_group(group, namespace))
            return written
        return self._upsert_group(items, namespaces.pop())

    def _upsert_group(self, items: List[MemoryItem], namespace: str) -> List[str]:
        vectors = self.embedder.encode([item.content for item in items])
        payloads = [self._to_payload(item) for item in items]
        ids = [item.id for item in items]
        self.vector_store.upsert(vectors, payloads, ids, namespace=namespace)
        for item in items:
            self._local[item.id] = item
        logger.info("Stored %s semantic memories in namespace=%s", len(items), namespace)
        return ids

    @staticmethod
    def _to_payload(item: MemoryItem) -> Dict[str, Any]:
        return {
            "memory_id": item.id,
            "user_id": item.user_id,
            "content": item.content,
            "memory_type": item.memory_type,
            "importance": item.importance,
            "timestamp": int(item.timestamp.timestamp()),
            "metadata": item.metadata,
            "company": item.metadata.get("company", ""),
            "status": item.metadata.get("status", ""),
        }

    def index_job_graph(self, job) -> None:
        """Mirror a posting into Neo4j so relationship queries work.

        Skill nodes hold normalised technology names ("python", "react"), not
        whole requirement sentences - otherwise `jobs_for_skill('python')`
        can never match and the graph is decorative.
        """
        if self.graph_store is None:
            return
        try:
            self.graph_store.upsert_company(
                job.company,
                {"location": job.location, "notes": f"source: {job.source}"},
            )
            self.graph_store.upsert_job(
                job.job_id,
                job.company,
                job.title,
                {
                    "url": job.url,
                    "location": job.location,
                    "remote": job.remote,
                    "status": job.status,
                    "posted_at": job.posted_at,
                    "source": job.source,
                },
            )
            self.graph_store.link_skills(job.job_id, _skill_names(job))
        except Exception as exc:
            # Graph indexing is an enrichment; never let it fail an ingest.
            logger.warning("Graph indexing failed for %s: %s", job.job_id, exc)

    # ------------------------------------------------------------------- reads

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        namespace: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> List[MemoryItem]:
        where: Dict[str, Any] = {"memory_type": SEMANTIC}
        if user_id:
            where["user_id"] = user_id
        if namespace:
            where["namespace"] = namespace

        try:
            vector = self.embedder.encode(query)
        except Exception as exc:
            logger.error("Embedding failed for query %r: %s", query[:60], exc)
            return []

        results = self.vector_store.search(vector, limit=limit, where=where, namespace=namespace)
        return [self._to_item(r) for r in results]

    def _to_item(self, hit: Dict[str, Any]) -> MemoryItem:
        from datetime import datetime, timezone

        payload = hit.get("payload") or {}
        ts = payload.get("timestamp")
        try:
            timestamp = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
        except (TypeError, ValueError):
            timestamp = datetime.now(timezone.utc)
        return MemoryItem(
            id=payload.get("memory_id", str(hit.get("id"))),
            content=payload.get("content", ""),
            memory_type=SEMANTIC,
            user_id=payload.get("user_id", "default"),
            timestamp=timestamp,
            importance=float(payload.get("importance", 0.5)),
            metadata={**(payload.get("metadata") or {}), "score": hit.get("score", 0.0)},
        )

    def retrieve_profile(self, query: str, limit: int = 8) -> List[MemoryItem]:
        return self.retrieve(query, limit=limit, namespace=PROFILE_NS)

    def retrieve_jobs(self, query: str, limit: int = 20) -> List[MemoryItem]:
        return self.retrieve(query, limit=limit, namespace=JOBS_NS)

    def list_namespace(self, namespace: str, limit: int = 1000) -> List[MemoryItem]:
        """Read every memory in a namespace back from persistent storage.

        The in-process `_local` cache is empty on a fresh start, so this is the
        only way to see what a previous run stored.
        """
        try:
            points = self.vector_store.scroll(namespace=namespace, limit=limit)
        except Exception as exc:
            logger.warning("Cannot scroll namespace %s: %s", namespace, exc)
            return []
        return [self._to_item(point) for point in points]

    def delete_job_vectors(self, memory_ids: List[str]) -> bool:
        return self.vector_store.delete_by_memory_ids(memory_ids)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> bool:
        item = self._local.get(memory_id)
        if item is None:
            return False
        if content is not None:
            item.content = content
        if importance is not None:
            item.importance = importance
        if metadata:
            item.metadata.update(metadata)
        self._upsert_group([item], item.metadata.get("namespace", "default"))
        return True

    def remove(self, memory_id: str) -> bool:
        self.vector_store.delete_by_memory_ids([memory_id])
        self._local.pop(memory_id, None)
        return True

    def has_memory(self, memory_id: str) -> bool:
        return memory_id in self._local

    def clear(self) -> None:
        if self.graph_store:
            self.graph_store.clear()
        self.vector_store.clear()
        self._local.clear()

    def get_all(self) -> List[MemoryItem]:
        return list(self._local.values())

    def get_stats(self) -> Dict[str, Any]:
        stats = {"count": self.vector_store.count(), "namespace": "all", "type": SEMANTIC}
        if self.graph_store:
            stats["graph"] = self.graph_store.get_stats()
        return stats
