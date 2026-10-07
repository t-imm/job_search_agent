"""Core data structures for the memory system.

Mirrors the HelloAgents reference architecture (MemoryItem / MemoryConfig /
BaseMemory) but is written in English with job-search semantics, and without
the reference's hard-coded dependency on a `src.core.database_config` module.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field

# The four memory tiers the agent uses.
WORKING = "working"
EPISODIC = "episodic"
SEMANTIC = "semantic"

MEMORY_TYPES = (WORKING, EPISODIC, SEMANTIC)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryItem(BaseModel):
    """A single stored memory.

    `content` is what gets embedded and searched. `metadata` carries the
    structured fields (source file, job id, company, ...) used for filtering
    and for building the final report.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: str = SEMANTIC
    user_id: str = "default"
    timestamp: datetime = Field(default_factory=utcnow)
    importance: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class MemoryConfig(BaseModel):
    """Tuning knobs shared by every memory tier."""
    storage_path: str = "./memory_data"
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95
    working_memory_capacity: int = 10
    working_memory_ttl_minutes: int = 120


class BaseMemory(ABC):
    """Common interface implemented by every memory tier."""

    def __init__(self, config: MemoryConfig, storage_backend=None) -> None:
        self.config = config
        self.storage = storage_backend
        self.memory_type = type(self).__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """Persist a memory and return its id."""

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """Return the memories most relevant to `query`."""

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> bool: ...

    @abstractmethod
    def remove(self, memory_id: str) -> bool: ...

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool: ...

    @abstractmethod
    def clear(self): ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]: ...

    def get_all(self) -> List[MemoryItem]:
        return []

    def __str__(self) -> str:
        return f"{type(self).__name__}(count={self.get_stats().get('count', 0)})"
