"""Memory management for cognitive agents.

Provides three memory stores mirroring human cognitive architecture:

* **Short-term memory** – recent observations/actions with automatic expiry.
* **Long-term memory** – consolidated, importance-scored records with pruning.
* **Working memory** – a small scratchpad for the current OODA cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    """A single record stored in any memory layer."""

    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """Time-limited memory that automatically expires old entries.

    Parameters
    ----------
    ttl:
        Time-to-live in seconds for each record.  Default is 300 s (5 min).
    max_size:
        Maximum number of records.  When exceeded the oldest entries are
        evicted first.
    """

    def __init__(self, ttl: float = 300.0, max_size: int = 100) -> None:
        self._records: list[MemoryRecord] = []
        self.ttl = ttl
        self.max_size = max_size

    def store(self, key: str, value: Any, importance: float = 0.5) -> MemoryRecord:
        """Store a value, evicting expired and over-limit records first."""
        self._evict_expired()
        record = MemoryRecord(key=key, value=value, importance=importance)
        self._records.append(record)
        # Keep within size limit – drop oldest
        while len(self._records) > self.max_size:
            self._records.pop(0)
        return record

    def recall(self, key: str) -> list[MemoryRecord]:
        """Return all non-expired records matching *key*."""
        self._evict_expired()
        return [r for r in self._records if r.key == key]

    def recent(self, n: int = 10) -> list[MemoryRecord]:
        """Return the *n* most recent records."""
        self._evict_expired()
        return list(reversed(self._records[-n:]))

    def extract(self, min_importance: float = 0.0) -> list[MemoryRecord]:
        """Remove and return non-expired records with importance above the threshold."""
        self._evict_expired()
        matching = [r for r in self._records if r.importance >= min_importance]
        self._records = [r for r in self._records if r.importance < min_importance]
        return matching

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        self._evict_expired()
        return len(self._records)

    def _evict_expired(self) -> None:
        now = time.time()
        self._records = [r for r in self._records if (now - r.timestamp) < self.ttl]


class LongTermMemory:
    """Persistent memory with importance scoring and automatic pruning.

    Parameters
    ----------
    max_size:
        Maximum number of records.  When exceeded the least important
        records are pruned.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._records: list[MemoryRecord] = []
        self.max_size = max_size

    def store(
        self,
        key: str,
        value: Any,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Store a value, pruning low-importance records if over limit."""
        record = MemoryRecord(
            key=key, value=value, importance=importance, metadata=metadata or {}
        )
        self._records.append(record)
        self._prune()
        return record

    def recall(self, key: str) -> list[MemoryRecord]:
        """Return all records matching *key*, most important first."""
        return sorted(
            [r for r in self._records if r.key == key],
            key=lambda r: r.importance,
            reverse=True,
        )

    def search(self, min_importance: float = 0.0) -> list[MemoryRecord]:
        """Return all records with importance >= *min_importance*."""
        return [r for r in self._records if r.importance >= min_importance]

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def _prune(self) -> None:
        """Remove least-important records when over *max_size*."""
        if len(self._records) > self.max_size:
            self._records.sort(key=lambda r: r.importance, reverse=True)
            self._records = self._records[: self.max_size]


class WorkingMemory:
    """Small scratchpad used during a single OODA cycle.

    Typically holds the current observation, orientation, and decision
    so that each phase of the loop can access prior results.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def clear(self) -> None:
        self._data.clear()

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


class MemorySystem:
    """Unified memory facade combining all three memory layers.

    Parameters
    ----------
    short_term_ttl:
        TTL for short-term memory records (seconds).
    short_term_max:
        Max records in short-term memory.
    long_term_max:
        Max records in long-term memory.
    """

    def __init__(
        self,
        short_term_ttl: float = 300.0,
        short_term_max: int = 100,
        long_term_max: int = 1000,
    ) -> None:
        self.short = ShortTermMemory(ttl=short_term_ttl, max_size=short_term_max)
        self.long = LongTermMemory(max_size=long_term_max)
        self.working = WorkingMemory()

    def consolidate(self, min_importance: float = 0.3) -> int:
        """Move short-term records above *min_importance* into long-term memory.

        Returns the number of records consolidated.
        """
        records = self.short.extract(min_importance=min_importance)
        for r in records:
            self.long.store(
                key=r.key, value=r.value, importance=r.importance, metadata=r.metadata
            )
        return len(records)

    def clear_all(self) -> None:
        """Clear every memory layer."""
        self.short.clear()
        self.long.clear()
        self.working.clear()
