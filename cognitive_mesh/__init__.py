"""Cognitive Mesh – BDI + OODA production-ready agent framework.

Combines the BDI (Belief-Desire-Intention) agent architecture with the
OODA (Observe-Orient-Decide-Act) loop to create a unified protocol for
building reliable AI agents.

Quick start::

    from cognitive_mesh import CognitiveAgent, AgentConfig

    agent = CognitiveAgent(AgentConfig(name="my-agent", role="demo"))
    result = await agent.cycle()
"""

from .agent import (
    ActionResult,
    AgentConfig,
    CognitiveAgent,
    CycleResult,
    Observation,
    Orientation,
)
from .beliefs import Belief, BeliefRecord, BeliefSystem
from .intentions import (
    Desire,
    Intention,
    IntentionState,
    IntentionSystem,
    Plan,
    PlanStep,
)
from .memory import (
    LongTermMemory,
    MemoryRecord,
    MemorySystem,
    ShortTermMemory,
    WorkingMemory,
)

__all__ = [
    # Agent
    "ActionResult",
    "AgentConfig",
    "CognitiveAgent",
    "CycleResult",
    "Observation",
    "Orientation",
    # Beliefs
    "Belief",
    "BeliefRecord",
    "BeliefSystem",
    # Intentions
    "Desire",
    "Intention",
    "IntentionState",
    "IntentionSystem",
    "Plan",
    "PlanStep",
    # Memory
    "LongTermMemory",
    "MemoryRecord",
    "MemorySystem",
    "ShortTermMemory",
    "WorkingMemory",
]
