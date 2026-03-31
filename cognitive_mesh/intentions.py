"""Desire and Intention management system.

Desires represent goals the agent wants to achieve, ordered by priority.
Intentions are commitments to execute a specific plan to satisfy a desire.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentionState(str, Enum):
    """Lifecycle states for an intention."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Desire:
    """A goal the agent wants to achieve."""

    goal: str
    priority: float = 0.5
    criteria: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: "")
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _generate_id("des")


@dataclass
class PlanStep:
    """A single step in an execution plan."""

    action: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """An execution plan attached to an intention."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)


@dataclass
class Checkpoint:
    """Tracks completion of a single plan step."""

    step: int
    completed: bool = False


@dataclass
class Intention:
    """A committed plan to satisfy a desire."""

    desire_id: str
    plan: Plan
    id: str = field(default_factory=lambda: "")
    state: IntentionState = IntentionState.PENDING
    created_at: float = field(default_factory=time.time)
    last_update: float | None = None
    progress: float = 0.0
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _generate_id("int")
        if not self.checkpoints and self.plan:
            self.checkpoints = [
                Checkpoint(step=i) for i in range(len(self.plan.steps))
            ]


class IntentionSystem:
    """Manages agent desires and intentions.

    Example::

        system = IntentionSystem()
        system.add_desire(Desire(goal="resolve_issue", priority=0.9))
        desires = system.get_desires()
        intention = system.commit_intention(
            desires[0].id,
            Plan(goal="resolve_issue", steps=[PlanStep(action="investigate")])
        )
    """

    def __init__(self) -> None:
        self._desires: list[Desire] = []
        self._intentions: list[Intention] = []

    # ------------------------------------------------------------------
    # Desires
    # ------------------------------------------------------------------

    def add_desire(self, desire: Desire) -> Desire:
        """Add a desire and keep the list sorted by descending priority."""
        self._desires.append(desire)
        self._desires.sort(key=lambda d: d.priority, reverse=True)
        return desire

    def remove_desire(self, desire_id: str) -> Desire | None:
        """Remove and return a desire by id."""
        for i, d in enumerate(self._desires):
            if d.id == desire_id:
                return self._desires.pop(i)
        return None

    def get_desires(self) -> list[Desire]:
        """Return a copy of the current desires list."""
        return list(self._desires)

    def find_desire(self, desire_id: str) -> Desire | None:
        """Lookup a desire by id."""
        for d in self._desires:
            if d.id == desire_id:
                return d
        return None

    # ------------------------------------------------------------------
    # Intentions
    # ------------------------------------------------------------------

    def commit_intention(self, desire_id: str, plan: Plan) -> Intention:
        """Create an intention from a desire and an execution plan.

        Raises
        ------
        ValueError
            If the referenced desire does not exist.
        """
        desire = self.find_desire(desire_id)
        if desire is None:
            raise ValueError(f"Desire not found: {desire_id}")

        intention = Intention(desire_id=desire_id, plan=plan)
        self._intentions.append(intention)
        return intention

    def update_intention(
        self,
        intention_id: str,
        state: IntentionState,
        progress: float = 0.0,
    ) -> Intention:
        """Update an intention's state and progress.

        Terminal states (``COMPLETED``, ``FAILED``, ``CANCELLED``) cause
        the intention to be removed from the active list.

        Raises
        ------
        ValueError
            If the intention is not found.
        """
        intention = self.find_intention(intention_id)
        if intention is None:
            raise ValueError(f"Intention not found: {intention_id}")

        intention.state = state
        intention.progress = progress
        intention.last_update = time.time()

        if state in (
            IntentionState.COMPLETED,
            IntentionState.FAILED,
            IntentionState.CANCELLED,
        ):
            self._intentions = [i for i in self._intentions if i.id != intention_id]

        return intention

    def find_intention(self, intention_id: str) -> Intention | None:
        """Lookup an intention by id."""
        for i in self._intentions:
            if i.id == intention_id:
                return i
        return None

    def get_intentions(
        self, state: IntentionState | None = None
    ) -> list[Intention]:
        """Return intentions, optionally filtered by *state*."""
        if state is None:
            return list(self._intentions)
        return [i for i in self._intentions if i.state == state]

    @property
    def desire_count(self) -> int:
        return len(self._desires)

    @property
    def intention_count(self) -> int:
        return len(self._intentions)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _generate_id(prefix: str = "id") -> str:
    """Generate a unique identifier with a prefix."""
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:9]}"
