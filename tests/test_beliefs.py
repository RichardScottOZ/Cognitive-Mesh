"""Tests for the BeliefSystem."""

from cognitive_mesh.beliefs import BeliefSystem


class TestBeliefSystem:
    def test_add_new_belief(self) -> None:
        bs = BeliefSystem()
        belief = bs.update("sky_is_blue", True, confidence=0.9)

        assert belief.value is True
        assert belief.confidence == 0.9
        assert len(belief.history) == 1
        assert "sky_is_blue" in bs

    def test_update_consistent_belief(self) -> None:
        bs = BeliefSystem()
        bs.update("fact", True, confidence=0.5)
        belief = bs.update("fact", True, confidence=0.3)

        # Consistent evidence should increase confidence
        assert belief.confidence > 0.5
        assert len(belief.history) == 2

    def test_update_inconsistent_belief(self) -> None:
        bs = BeliefSystem()
        bs.update("fact", True, confidence=0.8)
        belief = bs.update("fact", False, confidence=0.5)

        # Inconsistent evidence should decrease confidence
        assert belief.confidence < 0.8
        assert belief.value is False

    def test_query_existing(self) -> None:
        bs = BeliefSystem()
        bs.update("key", "value")
        result = bs.query("key")
        assert result is not None
        assert result.value == "value"

    def test_query_missing(self) -> None:
        bs = BeliefSystem()
        assert bs.query("nonexistent") is None

    def test_get_by_confidence(self) -> None:
        bs = BeliefSystem()
        bs.update("high", True, confidence=0.9)
        bs.update("low", True, confidence=0.3)

        high_confidence = bs.get_by_confidence(0.7)
        assert len(high_confidence) == 1
        assert high_confidence[0][0] == "high"

    def test_remove(self) -> None:
        bs = BeliefSystem()
        bs.update("key", "value")
        removed = bs.remove("key")
        assert removed is not None
        assert bs.query("key") is None

    def test_remove_missing(self) -> None:
        bs = BeliefSystem()
        assert bs.remove("nonexistent") is None

    def test_len(self) -> None:
        bs = BeliefSystem()
        assert len(bs) == 0
        bs.update("a", 1)
        bs.update("b", 2)
        assert len(bs) == 2

    def test_keys(self) -> None:
        bs = BeliefSystem()
        bs.update("x", 1)
        bs.update("y", 2)
        assert set(bs.keys()) == {"x", "y"}

    def test_bayesian_update_consistent(self) -> None:
        result = BeliefSystem.bayesian_update(0.5, 0.3, is_consistent=True)
        assert result > 0.5

    def test_bayesian_update_inconsistent(self) -> None:
        result = BeliefSystem.bayesian_update(0.5, 0.3, is_consistent=False)
        assert result < 0.5

    def test_bayesian_update_clamped(self) -> None:
        # Extremely high consistent evidence shouldn't exceed 1.0
        result = BeliefSystem.bayesian_update(0.99, 0.99, is_consistent=True)
        assert result <= 1.0
        # Extremely high inconsistent evidence shouldn't go below 0.0
        result = BeliefSystem.bayesian_update(0.01, 0.99, is_consistent=False)
        assert result >= 0.0
