"""Tests for the IntentionSystem."""

import pytest

from cognitive_mesh.intentions import (
    Desire,
    Intention,
    IntentionState,
    IntentionSystem,
    Plan,
    PlanStep,
)


class TestDesire:
    def test_create_desire(self) -> None:
        d = Desire(goal="test_goal", priority=0.8)
        assert d.goal == "test_goal"
        assert d.priority == 0.8
        assert d.id.startswith("des_")

    def test_default_priority(self) -> None:
        d = Desire(goal="g")
        assert d.priority == 0.5


class TestIntentionSystem:
    def test_add_desire(self) -> None:
        sys = IntentionSystem()
        d = sys.add_desire(Desire(goal="g1", priority=0.5))
        assert sys.desire_count == 1
        assert sys.find_desire(d.id) is d

    def test_desires_sorted_by_priority(self) -> None:
        sys = IntentionSystem()
        sys.add_desire(Desire(goal="low", priority=0.2))
        sys.add_desire(Desire(goal="high", priority=0.9))
        sys.add_desire(Desire(goal="mid", priority=0.5))

        desires = sys.get_desires()
        assert desires[0].goal == "high"
        assert desires[-1].goal == "low"

    def test_remove_desire(self) -> None:
        sys = IntentionSystem()
        d = sys.add_desire(Desire(goal="g"))
        removed = sys.remove_desire(d.id)
        assert removed is d
        assert sys.desire_count == 0

    def test_remove_missing_desire(self) -> None:
        sys = IntentionSystem()
        assert sys.remove_desire("no_such_id") is None

    def test_commit_intention(self) -> None:
        sys = IntentionSystem()
        d = sys.add_desire(Desire(goal="g"))
        plan = Plan(goal="g", steps=[PlanStep(action="step1")])
        intention = sys.commit_intention(d.id, plan)

        assert intention.desire_id == d.id
        assert intention.state == IntentionState.PENDING
        assert len(intention.checkpoints) == 1
        assert sys.intention_count == 1

    def test_commit_intention_missing_desire(self) -> None:
        sys = IntentionSystem()
        with pytest.raises(ValueError, match="Desire not found"):
            sys.commit_intention("fake_id", Plan(goal="g"))

    def test_update_intention_state(self) -> None:
        sys = IntentionSystem()
        d = sys.add_desire(Desire(goal="g"))
        plan = Plan(goal="g", steps=[PlanStep(action="s")])
        intention = sys.commit_intention(d.id, plan)

        sys.update_intention(intention.id, IntentionState.ACTIVE, progress=0.5)
        assert intention.state == IntentionState.ACTIVE
        assert intention.progress == 0.5

    def test_completed_intention_removed(self) -> None:
        sys = IntentionSystem()
        d = sys.add_desire(Desire(goal="g"))
        intention = sys.commit_intention(d.id, Plan(goal="g"))

        sys.update_intention(intention.id, IntentionState.COMPLETED)
        assert sys.intention_count == 0

    def test_failed_intention_removed(self) -> None:
        sys = IntentionSystem()
        d = sys.add_desire(Desire(goal="g"))
        intention = sys.commit_intention(d.id, Plan(goal="g"))

        sys.update_intention(intention.id, IntentionState.FAILED)
        assert sys.intention_count == 0

    def test_update_missing_intention(self) -> None:
        sys = IntentionSystem()
        with pytest.raises(ValueError, match="Intention not found"):
            sys.update_intention("nope", IntentionState.ACTIVE)

    def test_get_intentions_by_state(self) -> None:
        sys = IntentionSystem()
        d = sys.add_desire(Desire(goal="g"))
        plan = Plan(goal="g", steps=[])
        i1 = sys.commit_intention(d.id, plan)
        i2 = sys.commit_intention(d.id, plan)
        sys.update_intention(i1.id, IntentionState.ACTIVE)

        active = sys.get_intentions(IntentionState.ACTIVE)
        assert len(active) == 1
        pending = sys.get_intentions(IntentionState.PENDING)
        assert len(pending) == 1
