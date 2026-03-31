"""Tests for the memory system."""

import time

from cognitive_mesh.memory import (
    LongTermMemory,
    MemorySystem,
    ShortTermMemory,
    WorkingMemory,
)


class TestShortTermMemory:
    def test_store_and_recall(self) -> None:
        stm = ShortTermMemory(ttl=60)
        stm.store("key", "value")
        records = stm.recall("key")
        assert len(records) == 1
        assert records[0].value == "value"

    def test_recent(self) -> None:
        stm = ShortTermMemory(ttl=60)
        for i in range(5):
            stm.store("k", i)
        recent = stm.recent(3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].value == 4

    def test_max_size(self) -> None:
        stm = ShortTermMemory(ttl=60, max_size=3)
        for i in range(5):
            stm.store("k", i)
        assert len(stm) == 3

    def test_ttl_expiry(self) -> None:
        stm = ShortTermMemory(ttl=0.01)
        stm.store("k", "v")
        time.sleep(0.02)
        assert len(stm) == 0

    def test_clear(self) -> None:
        stm = ShortTermMemory()
        stm.store("a", 1)
        stm.clear()
        assert len(stm) == 0


class TestLongTermMemory:
    def test_store_and_recall(self) -> None:
        ltm = LongTermMemory()
        ltm.store("key", "data", importance=0.8)
        records = ltm.recall("key")
        assert len(records) == 1
        assert records[0].importance == 0.8

    def test_recall_sorted_by_importance(self) -> None:
        ltm = LongTermMemory()
        ltm.store("k", "low", importance=0.3)
        ltm.store("k", "high", importance=0.9)
        records = ltm.recall("k")
        assert records[0].value == "high"

    def test_search_by_importance(self) -> None:
        ltm = LongTermMemory()
        ltm.store("a", 1, importance=0.2)
        ltm.store("b", 2, importance=0.7)
        results = ltm.search(min_importance=0.5)
        assert len(results) == 1
        assert results[0].key == "b"

    def test_prune(self) -> None:
        ltm = LongTermMemory(max_size=2)
        ltm.store("a", 1, importance=0.1)
        ltm.store("b", 2, importance=0.5)
        ltm.store("c", 3, importance=0.9)
        assert len(ltm) == 2
        # The lowest importance record should be pruned
        assert all(r.importance >= 0.5 for r in ltm.search())

    def test_clear(self) -> None:
        ltm = LongTermMemory()
        ltm.store("k", "v")
        ltm.clear()
        assert len(ltm) == 0


class TestWorkingMemory:
    def test_set_and_get(self) -> None:
        wm = WorkingMemory()
        wm.set("key", "value")
        assert wm.get("key") == "value"

    def test_get_default(self) -> None:
        wm = WorkingMemory()
        assert wm.get("missing", "default") == "default"

    def test_contains(self) -> None:
        wm = WorkingMemory()
        wm.set("x", 1)
        assert "x" in wm
        assert "y" not in wm

    def test_clear(self) -> None:
        wm = WorkingMemory()
        wm.set("a", 1)
        wm.clear()
        assert len(wm) == 0

    def test_keys(self) -> None:
        wm = WorkingMemory()
        wm.set("a", 1)
        wm.set("b", 2)
        assert set(wm.keys()) == {"a", "b"}


class TestMemorySystem:
    def test_consolidate(self) -> None:
        ms = MemorySystem()
        ms.short.store("important", "data", importance=0.8)
        ms.short.store("trivial", "data", importance=0.1)

        count = ms.consolidate(min_importance=0.5)
        assert count == 1
        assert len(ms.long) == 1
        assert len(ms.short) == 1

    def test_consolidate_is_idempotent_after_move(self) -> None:
        ms = MemorySystem()
        ms.short.store("important", "data", importance=0.8)

        first = ms.consolidate(min_importance=0.5)
        second = ms.consolidate(min_importance=0.5)

        assert first == 1
        assert second == 0
        assert len(ms.long) == 1
        assert len(ms.short) == 0

    def test_consolidate_skips_expired_short_term_records(self) -> None:
        ms = MemorySystem(short_term_ttl=0.01)
        ms.short.store("important", "data", importance=0.8)

        time.sleep(0.02)

        count = ms.consolidate(min_importance=0.5)
        assert count == 0
        assert len(ms.long) == 0

    def test_clear_all(self) -> None:
        ms = MemorySystem()
        ms.short.store("a", 1)
        ms.long.store("b", 2)
        ms.working.set("c", 3)
        ms.clear_all()
        assert len(ms.short) == 0
        assert len(ms.long) == 0
        assert len(ms.working) == 0
