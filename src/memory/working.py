"""Working memory - the small, fast, short-lived scratchpad.

Holds the in-flight state of a single run (the profile digest, the current
plan, candidate jobs) so the planner and executor do not have to re-derive it.
TTL- and capacity-bounded: anything that matters long-term is promoted to
semantic or episodic memory instead.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .base import WORKING, BaseMemory, MemoryConfig, MemoryItem

logger = logging.getLogger(__name__)


class WorkingMemory(BaseMemory):
    def __init__(self, config: MemoryConfig) -> None:
        super().__init__(config)
        self.memory_type = WORKING
        self._items: Dict[str, MemoryItem] = {}
        self._lock = threading.RLock()

    def add(self, memory_item: MemoryItem) -> str:
        with self._lock:
            self._expire()
            self._items[memory_item.id] = memory_item
            self._enforce_capacity()
        return memory_item.id

    def get(self, key: str, default: Any = None) -> Any:
        """Fetch a value by its metadata key (used for scratchpad slots)."""
        with self._lock:
            self._expire()
            for item in self._items.values():
                if item.metadata.get("key") == key:
                    return item.content
        return default

    def set(self, key: str, content: str, importance: float = 0.5) -> str:
        return self.add(
            MemoryItem(
                content=content,
                memory_type=WORKING,
                user_id="default",
                importance=importance,
                metadata={"key": key},
            )
        )

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        with self._lock:
            self._expire()
            items = list(self._items.values())

        if not query:
            return sorted(items, key=lambda i: i.timestamp, reverse=True)[:limit]

        # Cheap token overlap - working memory is small, so exact scoring is fine.
        needle = {t for t in query.lower().split() if len(t) > 2}
        scored = []
        for item in items:
            haystack = set(item.content.lower().split())
            overlap = len(needle & haystack)
            if overlap:
                scored.append((overlap, item.timestamp.timestamp(), item))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item for _, _, item in scored[:limit]]

    def _expire(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=self.config.working_memory_ttl_minutes
        )
        stale = [mid for mid, item in self._items.items() if item.timestamp < cutoff]
        for mid in stale:
            self._items.pop(mid, None)

    def _enforce_capacity(self) -> None:
        overflow = len(self._items) - self.config.working_memory_capacity
        if overflow <= 0:
            return
        # Evict the least important, oldest first.
        victims = sorted(
            self._items.values(), key=lambda i: (i.importance, i.timestamp)
        )[:overflow]
        for victim in victims:
            self._items.pop(victim.id, None)

    def update(self, memory_id: str, content=None, importance=None, metadata=None) -> bool:
        with self._lock:
            item = self._items.get(memory_id)
            if not item:
                return False
            if content is not None:
                item.content = content
            if importance is not None:
                item.importance = importance
            if metadata:
                item.metadata.update(metadata)
        return True

    def remove(self, memory_id: str) -> bool:
        with self._lock:
            return self._items.pop(memory_id, None) is not None

    def has_memory(self, memory_id: str) -> bool:
        with self._lock:
            return memory_id in self._items

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def get_all(self) -> List[MemoryItem]:
        with self._lock:
            return list(self._items.values())

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "count": len(self._items),
                "capacity": self.config.working_memory_capacity,
                "type": WORKING,
            }
