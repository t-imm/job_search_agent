"""Qdrant vector store adapter.

Wraps the local Qdrant instance from docker_compose.yml. The collection is
created on demand with the dimension reported by the embedding provider, and
keyword payload indexes are added for the fields we filter on.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from ...config import QdrantConfig, get_settings

logger = logging.getLogger(__name__)

def _distance(name: str):
    """Resolve a distance metric, tolerating client enum renames."""
    key = (name or "cosine").lower()
    enum = models.Distance
    for candidate in ({"cosine": "COSINE", "dot": "DOT", "euclid": "EUCLID", "euclidean": "EUCLID"}[key],):
        value = getattr(enum, candidate, None)
        if value is not None:
            return value
    return enum.COSINE


class QdrantUnavailable(RuntimeError):
    """Qdrant could not be reached."""


class QdrantStore:
    def __init__(self, config: QdrantConfig | None = None, vector_size: int | None = None) -> None:
        self.config = config or get_settings().qdrant
        self.collection = self.config.collection
        # The embedder's real dimension wins over the .env value.
        self.vector_size = vector_size or self.config.vector_size
        self.client = self._connect()
        self._ensure_collection()

    def _connect(self) -> QdrantClient:
        kwargs: Dict[str, Any] = {"url": self.config.url, "timeout": self.config.timeout}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        try:
            client = QdrantClient(**kwargs)
            client.get_collections()
            logger.info("Connected to Qdrant at %s", self.config.url)
            return client
        except Exception as exc:
            raise QdrantUnavailable(
                f"Cannot reach Qdrant at {self.config.url}. "
                f"Start it with: docker compose -f docker_compose.yml up -d ({exc})"
            ) from exc

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=_distance(self.config.distance),
                ),
            )
            logger.info("Created Qdrant collection %s (dim=%s)", self.collection, self.vector_size)
        else:
            info = self.client.get_collection(self.collection)
            actual = info.config.params.vectors.size
            if actual != self.vector_size:
                raise QdrantUnavailable(
                    f"Collection '{self.collection}' has dimension {actual} but the embedder "
                    f"produces {self.vector_size}. Change QDRANT_COLLECTION in .env or recreate it."
                )

        for field, schema in [
            ("memory_type", models.PayloadSchemaType.KEYWORD),
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("memory_id", models.PayloadSchemaType.KEYWORD),
            ("namespace", models.PayloadSchemaType.KEYWORD),
            ("company", models.PayloadSchemaType.KEYWORD),
            ("status", models.PayloadSchemaType.KEYWORD),
            ("timestamp", models.PayloadSchemaType.INTEGER),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema=schema
                )
            except Exception:
                # Index already exists - harmless.
                pass

    @staticmethod
    def _as_point_id(value: Any) -> str:
        """Qdrant accepts UUID strings or unsigned ints; normalise to UUID."""
        text = str(value)
        try:
            return str(uuid.UUID(text))
        except (ValueError, AttributeError, TypeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, text))

    def upsert(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Any],
        namespace: str = "default",
    ) -> bool:
        if not vectors:
            return False
        if not (len(vectors) == len(payloads) == len(ids)):
            raise ValueError("vectors, payloads and ids must be the same length")

        points = []
        skipped = 0
        for vector, payload, raw_id in zip(vectors, payloads, ids):
            if len(vector) != self.vector_size:
                skipped += 1
                continue
            enriched = dict(payload)
            enriched["namespace"] = namespace
            points.append(
                models.PointStruct(
                    id=self._as_point_id(raw_id),
                    vector=vector,
                    payload=enriched,
                )
            )
        if skipped:
            logger.warning("Skipped %s point(s) with wrong vector dimension", skipped)
        if not points:
            return False

        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return True

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        conditions = []
        filters = dict(where or {})
        if namespace:
            filters["namespace"] = namespace
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        query_filter = models.Filter(must=conditions) if conditions else None

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            points = response.points
        except AttributeError:  # older qdrant-client used .search()
            points = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
        return [{"id": p.id, "score": p.score, "payload": p.payload or {}} for p in points]

    def scroll(
        self,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Page through stored points.

        Needed because vectors are the durable record: the in-process cache is
        empty on a fresh run, so anything that must survive a restart has to be
        read back from Qdrant.
        """
        conditions = []
        filters = dict(where or {})
        if namespace:
            filters["namespace"] = namespace
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        query_filter = models.Filter(must=conditions) if conditions else None

        points: List[Dict[str, Any]] = []
        offset = None
        while len(points) < limit:
            batch, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=min(256, limit - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not batch:
                break
            points.extend({"id": p.id, "payload": p.payload or {}} for p in batch)
            if offset is None:
                break
        return points[:limit]

    def delete_by_memory_ids(self, memory_ids: List[str]) -> bool:
        if not memory_ids:
            return True
        conditions = [
            models.FieldCondition(key="memory_id", match=models.MatchValue(value=str(mid)))
            for mid in memory_ids
        ]
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=conditions)),
            wait=True,
        )
        return True

    def count(self) -> int:
        try:
            return int(self.client.count(collection_name=self.collection, exact=True).count)
        except Exception:
            info = self.client.get_collection(self.collection)
            return int(info.points_count or 0)

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.error("Qdrant health check failed: %s", exc)
            return False

    def clear(self) -> None:
        self.client.delete_collection(collection_name=self.collection)
        self._ensure_collection()
