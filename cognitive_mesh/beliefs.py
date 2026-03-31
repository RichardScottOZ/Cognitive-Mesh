"""Belief System with Bayesian confidence updates.

Implements a belief management system where each belief has a value,
confidence score, creation time, and history of updates. Confidence
is updated using a simplified Bayesian inference model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BeliefRecord:
    """A single historical record of a belief value and confidence."""

    value: Any
    confidence: float
    timestamp: float


@dataclass
class Belief:
    """A belief held by the agent."""

    value: Any
    confidence: float
    created_at: float
    last_update: float
    history: list[BeliefRecord] = field(default_factory=list)


class BeliefSystem:
    """Manages agent beliefs with Bayesian confidence updates.

    Each belief is keyed by a string identifier and has an associated value,
    confidence score (0.0–1.0), and full history of updates.

    Example::

        bs = BeliefSystem()
        bs.update("sky_is_blue", True, confidence=0.9)
        belief = bs.query("sky_is_blue")
        assert belief.confidence == 0.9
    """

    def __init__(self) -> None:
        self._beliefs: dict[str, Belief] = {}

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update(self, key: str, value: Any, confidence: float = 0.5) -> Belief:
        """Add or update a belief.

        If the belief already exists its confidence is updated using
        :meth:`bayesian_update`.  A full history of changes is kept.

        Parameters
        ----------
        key:
            Unique identifier for the belief.
        value:
            The value of the belief (can be any type).
        confidence:
            Evidence confidence in the range ``[0, 1]``.

        Returns
        -------
        Belief
            The created or updated belief.
        """
        now = time.time()
        existing = self._beliefs.get(key)

        if existing is not None:
            is_consistent = value == existing.value
            new_confidence = self.bayesian_update(
                existing.confidence, confidence, is_consistent
            )
            existing.value = value
            existing.confidence = new_confidence
            existing.last_update = now
            existing.history.append(
                BeliefRecord(value=value, confidence=new_confidence, timestamp=now)
            )
            return existing

        belief = Belief(
            value=value,
            confidence=confidence,
            created_at=now,
            last_update=now,
            history=[BeliefRecord(value=value, confidence=confidence, timestamp=now)],
        )
        self._beliefs[key] = belief
        return belief

    def remove(self, key: str) -> Belief | None:
        """Remove and return a belief, or ``None`` if not found."""
        return self._beliefs.pop(key, None)

    def apply_evidence(
        self,
        key: str,
        evidence: float,
        is_consistent: bool,
    ) -> Belief | None:
        """Adjust the confidence of an existing belief without changing its value."""
        existing = self._beliefs.get(key)
        if existing is None:
            return None

        now = time.time()
        new_confidence = self.bayesian_update(
            existing.confidence, evidence, is_consistent
        )
        existing.confidence = new_confidence
        existing.last_update = now
        existing.history.append(
            BeliefRecord(
                value=existing.value,
                confidence=new_confidence,
                timestamp=now,
            )
        )
        return existing

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query(self, key: str) -> Belief | None:
        """Return the belief for *key*, or ``None``."""
        return self._beliefs.get(key)

    def get_by_confidence(self, threshold: float = 0.7) -> list[tuple[str, Belief]]:
        """Return all beliefs whose confidence meets *threshold*."""
        return [
            (k, b) for k, b in self._beliefs.items() if b.confidence >= threshold
        ]

    def keys(self) -> list[str]:
        """Return all belief keys."""
        return list(self._beliefs.keys())

    def __len__(self) -> int:
        return len(self._beliefs)

    def __contains__(self, key: str) -> bool:
        return key in self._beliefs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def bayesian_update(
        prior: float, evidence: float, is_consistent: bool
    ) -> float:
        """Simplified Bayesian confidence update.

        When new evidence is *consistent* with the current belief the
        confidence increases; otherwise it decreases.

        Parameters
        ----------
        prior:
            Current confidence in ``[0, 1]``.
        evidence:
            Strength of the new evidence in ``[0, 1]``.
        is_consistent:
            Whether the new evidence agrees with the existing belief.

        Returns
        -------
        float
            Updated confidence clamped to ``[0, 1]``.
        """
        if is_consistent:
            result = prior + (1 - prior) * evidence
        else:
            result = prior * (1 - evidence)
        return max(0.0, min(1.0, result))
