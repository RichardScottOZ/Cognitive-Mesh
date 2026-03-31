"""Cognitive Agent – BDI architecture combined with the OODA loop.

The :class:`CognitiveAgent` is the central piece of the Cognitive Mesh
protocol.  It maintains beliefs, desires, intentions, and three-layer
memory while running continuous OODA cycles (Observe → Orient → Decide → Act).

Sub-classes override the ``observe``, ``orient``, ``decide``, and ``act``
hooks (as well as helper methods) to specialise the agent for a particular
domain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .beliefs import BeliefSystem
from .intentions import (
    Desire,
    Intention,
    IntentionState,
    IntentionSystem,
    Plan,
    PlanStep,
)
from .memory import MemorySystem


@dataclass
class AgentConfig:
    """Configuration for a :class:`CognitiveAgent`."""

    name: str = "CognitiveAgent"
    role: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Data gathered during the *observe* phase."""

    timestamp: float
    context: dict[str, Any]
    sensory: dict[str, Any]
    memory: list[Any]


@dataclass
class Orientation:
    """Analysis produced during the *orient* phase."""

    matched_beliefs: list[tuple[str, Any]]
    patterns: list[str]
    relevance: float
    confidence: float


@dataclass
class ActionResult:
    """Result of executing a single plan step."""

    step: PlanStep
    success: bool
    data: Any = None
    belief: str | None = None


@dataclass
class CycleResult:
    """Full result of one OODA cycle."""

    plan: Plan
    results: list[ActionResult]


class CognitiveAgent:
    """Base cognitive agent implementing the Cognitive Mesh protocol.

    The default implementations of ``observe``, ``orient``, ``decide``, and
    ``act`` provide a simple but functional pipeline.  Override them in
    sub-classes to add domain-specific logic.

    Parameters
    ----------
    config:
        Agent configuration (name, role, etc.).
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.beliefs = BeliefSystem()
        self.intention_system = IntentionSystem()
        self.memory = MemorySystem()
        self._ooda_state: str = "observe"
        self._context: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def set_context(self, context: dict[str, Any]) -> None:
        """Set the external context for the next cycle."""
        self._context = context

    # ------------------------------------------------------------------
    # OODA loop
    # ------------------------------------------------------------------

    async def cycle(self) -> CycleResult:
        """Execute one full OODA cycle.

        Returns
        -------
        CycleResult
            Contains the executed plan and the results of each step.
        """
        self._ooda_state = "observe"
        observation = await self.observe()

        self._ooda_state = "orient"
        orientation = await self.orient(observation)

        self._ooda_state = "decide"
        decision = await self.decide(orientation)

        self._ooda_state = "act"
        cycle_result = await self.act(decision)

        # Update beliefs based on action outcomes
        await self.update_beliefs(cycle_result)

        return cycle_result

    # ------------------------------------------------------------------
    # OODA phases (override in sub-classes)
    # ------------------------------------------------------------------

    async def observe(self) -> Observation:
        """Gather data from the environment.

        Override to add custom data-collection logic.
        """
        data = Observation(
            timestamp=time.time(),
            context=await self.get_context(),
            sensory=await self.get_sensory_input(),
            memory=self.get_relevant_memories(),
        )
        self.memory.working.set("observation", data)
        return data

    async def orient(self, observation: Observation) -> Orientation:
        """Analyse and contextualise the observation."""
        analysis = Orientation(
            matched_beliefs=self.match_beliefs(observation),
            patterns=self.detect_patterns(observation),
            relevance=self.assess_relevance(observation),
            confidence=self.calculate_confidence(observation),
        )
        self.memory.working.set("orientation", analysis)
        return analysis

    async def decide(self, orientation: Orientation) -> Plan:
        """Choose an action plan based on the orientation."""
        active = [
            i
            for i in self.intention_system.get_intentions()
            if self.is_intention_relevant(i, orientation)
        ]
        selected = self.select_intention(active, orientation)
        plan = await self.create_plan(selected, orientation)
        self.memory.working.set("decision", plan)
        return plan

    async def act(self, plan: Plan) -> CycleResult:
        """Execute the plan steps and record results."""
        results: list[ActionResult] = []

        for step in plan.steps:
            result = await self.execute_step(step)
            results.append(result)

            # Store in short-term memory
            self.memory.short.store(
                key=step.action,
                value={"step": step, "result": result},
                importance=0.5,
            )

        # Consolidate short-term → long-term
        self.memory.consolidate()

        return CycleResult(plan=plan, results=results)

    # ------------------------------------------------------------------
    # Belief updates
    # ------------------------------------------------------------------

    async def update_beliefs(self, cycle_result: CycleResult) -> None:
        """Strengthen or weaken beliefs based on action outcomes."""
        for result in cycle_result.results:
            if result.belief is None:
                continue
            if result.success:
                self.strengthen_belief(result.belief)
            else:
                self.weaken_belief(result.belief)

    def strengthen_belief(self, key: str, amount: float = 0.1) -> None:
        """Increase confidence in an existing belief."""
        belief = self.beliefs.query(key)
        if belief is not None:
            self.beliefs.update(key, belief.value, confidence=amount)

    def weaken_belief(self, key: str, amount: float = 0.3) -> None:
        """Decrease confidence in an existing belief by providing
        contradictory evidence."""
        belief = self.beliefs.query(key)
        if belief is not None:
            # Supply the *opposite* value so bayesian_update treats it as
            # inconsistent evidence.
            opposite = not belief.value if isinstance(belief.value, bool) else None
            self.beliefs.update(key, opposite, confidence=amount)

    # ------------------------------------------------------------------
    # Default helper implementations (override for domain-specific logic)
    # ------------------------------------------------------------------

    async def get_context(self) -> dict[str, Any]:
        """Return the current external context."""
        return dict(self._context)

    async def get_sensory_input(self) -> dict[str, Any]:
        """Return raw sensory input.  Override in sub-classes."""
        return {}

    def get_relevant_memories(self) -> list[Any]:
        """Retrieve memories relevant to the current cycle."""
        return [r.value for r in self.memory.short.recent(5)]

    def match_beliefs(self, observation: Observation) -> list[tuple[str, Any]]:
        """Return beliefs relevant to the *observation*."""
        return [
            (k, b.value)
            for k, b in self.beliefs.get_by_confidence(threshold=0.5)
        ]

    def detect_patterns(self, observation: Observation) -> list[str]:
        """Detect patterns in the observation.  Override for real logic."""
        return []

    def assess_relevance(self, observation: Observation) -> float:
        """Score how relevant the observation is (0–1)."""
        return 1.0 if observation.context else 0.5

    def calculate_confidence(self, observation: Observation) -> float:
        """Calculate overall confidence in the observation."""
        if not observation.sensory:
            return 0.5
        return 0.8

    def is_intention_relevant(
        self, intention: Intention, orientation: Orientation
    ) -> bool:
        """Return ``True`` if *intention* should be considered given *orientation*."""
        return intention.state in (IntentionState.PENDING, IntentionState.ACTIVE)

    def select_intention(
        self,
        intentions: list[Intention],
        orientation: Orientation,
    ) -> Intention | None:
        """Pick the best intention from the candidate list."""
        return intentions[0] if intentions else None

    async def create_plan(
        self,
        intention: Intention | None,
        orientation: Orientation,
    ) -> Plan:
        """Build a plan for the selected intention.

        If no intention is selected a default no-op plan is returned.
        """
        if intention is not None:
            return intention.plan
        return Plan(goal="noop", steps=[])

    async def execute_step(self, step: PlanStep) -> ActionResult:
        """Execute a single plan step.  Override for real execution."""
        return ActionResult(step=step, success=True, data=None, belief=None)
