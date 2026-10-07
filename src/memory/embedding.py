"""Embedding provider backed by Aliyun DashScope (text-embedding-v3).

Wraps the OpenAI-compatible REST endpoint. Embeddings are generated in batch
because the agent embeds whole job batches at once during ingestion.

The dimension is probed once at construction rather than trusted from config,
so a mismatch with the Qdrant collection is caught immediately instead of
failing silently on the first upsert.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import List, Sequence, Union

import requests

from ..config import EmbeddingConfig, get_settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend cannot produce vectors."""


class DashScopeEmbedder:
    """text-embedding-v3 via the OpenAI-compatible `/embeddings` route."""

    #: Hard server-side limit for text-embedding-v3 inputs per request.
    max_batch_size = 10

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or get_settings().embedding
        if not self.config.api_key:
            raise EmbeddingError("EMBED_API_KEY is not configured")
        if not self.config.base_url:
            raise EmbeddingError("EMBED_BASE_URL is not configured")
        self._dimension: int | None = None
        self._lock = threading.Lock()
        self.batch_size = self.max_batch_size

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.encode("dimension probe"))
        return self._dimension

    def encode(self, texts: Union[str, Sequence[str]]) -> List[List[float]]:
        """Embed one string or a batch. Returns a list of float vectors."""
        single = isinstance(texts, str)
        inputs = [texts] if single else [t for t in texts]

        if not inputs:
            return []
        # DashScope rejects empty strings; substitute a placeholder so that a
        # blank job field never takes down a whole batch.
        payload_inputs = [(t if t and t.strip() else " ") for t in inputs]

        vectors: List[List[float]] = []
        # DashScope rejects requests with more than 10 inputs per call
        # ("batch size is invalid, it should not be larger than 10").
        for start in range(0, len(payload_inputs), self.batch_size):
            chunk = payload_inputs[start:start + self.batch_size]
            vectors.extend(self._encode_chunk(chunk))

        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])

        if single:
            return vectors[0] if vectors else []
        return vectors

    def _encode_chunk(self, chunk: List[str]) -> List[List[float]]:
        url = self.config.base_url.rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "input": chunk,
            "dimensions": self.config.dimension,
        }

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
                if resp.status_code >= 400:
                    # 4xx other than 429 will not improve on retry.
                    if resp.status_code < 500 and resp.status_code != 429:
                        raise EmbeddingError(
                            f"Embedding API rejected request ({resp.status_code}): {resp.text[:300]}"
                        )
                    raise EmbeddingError(f"Embedding API error {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                items = data.get("data") or []
                if len(items) != len(chunk):
                    raise EmbeddingError(
                        f"Embedding count mismatch: requested {len(chunk)}, got {len(items)}"
                    )
                return [item["embedding"] for item in items]
            except EmbeddingError as exc:
                last_error = exc
                if "rejected request" in str(exc):
                    raise
            except Exception as exc:  # network hiccup, malformed JSON, ...
                last_error = exc
            time.sleep(1.5 * (attempt + 1))

        raise EmbeddingError(f"Embedding request failed after 3 attempts: {last_error}")


_embedder: DashScopeEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> DashScopeEmbedder:
    """Process-wide singleton, so the dimension probe happens only once."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = DashScopeEmbedder()
    return _embedder
