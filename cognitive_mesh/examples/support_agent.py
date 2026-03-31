"""Customer Support Agent – a domain-specific CognitiveAgent.

Demonstrates how to extend :class:`~cognitive_mesh.agent.CognitiveAgent`
for a concrete use-case: handling customer support interactions.

The agent:

1. Analyses customer sentiment and urgency during *observe*.
2. Classifies the issue and estimates difficulty during *orient*.
3. Creates an urgent-response plan when urgency is high during *decide*.
4. Monitors sentiment after providing a solution during *act*.
"""

from __future__ import annotations

import re
from typing import Any

from ..agent import (
    ActionResult,
    AgentConfig,
    CognitiveAgent,
    CycleResult,
    Observation,
    Orientation,
)
from ..intentions import Desire, Plan, PlanStep


class SupportAgent(CognitiveAgent):
    """A customer-support agent built on the Cognitive Mesh protocol.

    Example::

        agent = SupportAgent()
        agent.set_context({
            "message": "My account is locked and I need help urgently!",
            "customer_id": "cust_42",
        })
        result = await agent.cycle()
    """

    URGENCY_THRESHOLD: float = 0.8

    def __init__(self) -> None:
        super().__init__(
            AgentConfig(name="SupportAgent", role="customer_support")
        )

        # Seed initial beliefs
        self.beliefs.update("customers_want_fast_responses", True, confidence=0.9)
        self.beliefs.update(
            "customers_prefer_solutions_over_apologies", True, confidence=0.8
        )
        self.beliefs.update(
            "technical_issues_have_root_causes", True, confidence=0.95
        )

        # Seed desires
        self.intention_system.add_desire(
            Desire(
                goal="resolve_customer_issue",
                priority=0.9,
                criteria=[
                    "customer_confirmed_resolution",
                    "issue_did_not_recur",
                ],
            )
        )
        self.intention_system.add_desire(
            Desire(
                goal="maintain_customer_satisfaction",
                priority=0.7,
                criteria=["response_time < 60s", "customer_sentiment > 0.7"],
            )
        )

    # ------------------------------------------------------------------
    # OODA overrides
    # ------------------------------------------------------------------

    async def observe(self) -> Observation:
        observation = await super().observe()

        # Enrich with customer-specific data
        message = observation.context.get("message", "")
        customer_id = observation.context.get("customer_id")

        observation.sensory["customer"] = {
            "sentiment": self.analyse_sentiment(message),
            "urgency": self.assess_urgency(message),
            "history": self.get_customer_history(customer_id),
        }
        return observation

    async def orient(self, observation: Observation) -> Orientation:
        orientation = await super().orient(observation)

        message = observation.context.get("message", "")
        customer_data = observation.sensory.get("customer", {})

        # Attach customer-specific orientation info as extra attributes
        orientation.issue_category = self.classify_issue(message)  # type: ignore[attr-defined]
        orientation.estimated_difficulty = self.assess_difficulty(customer_data)  # type: ignore[attr-defined]
        orientation.recommended_actions = self.get_recommendations(  # type: ignore[attr-defined]
            customer_data
        )
        return orientation

    async def decide(self, orientation: Orientation) -> Plan:
        # Check if this is an urgent case
        obs: Observation | None = self.memory.working.get("observation")
        urgency = 0.0
        if obs is not None:
            customer_data = obs.sensory.get("customer", {})
            urgency = customer_data.get("urgency", 0.0)

        if urgency > self.URGENCY_THRESHOLD:
            return Plan(
                goal="rapid_response",
                steps=[
                    PlanStep(action="acknowledge_immediately"),
                    PlanStep(action="gather_context"),
                    PlanStep(action="provide_initial_guidance"),
                ],
            )

        return Plan(
            goal="resolve_customer_issue",
            steps=[
                PlanStep(action="acknowledge_issue"),
                PlanStep(action="investigate_problem"),
                PlanStep(action="provide_solution"),
            ],
        )

    async def act(self, plan: Plan) -> CycleResult:
        results: list[ActionResult] = []

        for step in plan.steps:
            result = await self.execute_support_action(step)
            results.append(result)

            # Store in short-term memory
            self.memory.short.store(
                key=step.action,
                value={"step": step, "result": result},
                importance=0.6,
            )

            # Monitor sentiment after providing a solution
            if step.action == "provide_solution":
                sentiment = self._latest_customer_sentiment()
                if sentiment < 0.3:
                    escalation = await self.escalate_to_human()
                    results.append(escalation)

        self.memory.consolidate()
        return CycleResult(plan=plan, results=results)

    # ------------------------------------------------------------------
    # Support-specific helpers
    # ------------------------------------------------------------------

    async def execute_support_action(self, step: PlanStep) -> ActionResult:
        """Execute a customer-support action.

        Override this in production to call real tools/APIs.
        """
        return ActionResult(step=step, success=True, data=None, belief=None)

    async def escalate_to_human(self) -> ActionResult:
        """Escalate the case to a human agent."""
        step = PlanStep(action="escalate_to_human")
        return ActionResult(
            step=step,
            success=True,
            data={"reason": "low_customer_sentiment"},
        )

    # ------------------------------------------------------------------
    # Analysis helpers (stubs – replace with real ML/NLP in production)
    # ------------------------------------------------------------------

    @staticmethod
    def analyse_sentiment(text: str) -> float:
        """Return a sentiment score in ``[0, 1]``.

        This is a placeholder implementation that uses simple keyword
        matching.  Replace with a real NLP model in production.
        """
        negative = {"angry", "frustrated", "broken", "terrible", "worst", "locked"}
        positive = {"thanks", "great", "happy", "resolved", "excellent", "good"}

        words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
        neg = len(words & negative)
        pos = len(words & positive)
        total = neg + pos
        if total == 0:
            return 0.5
        return pos / total

    @staticmethod
    def assess_urgency(text: str) -> float:
        """Return an urgency score in ``[0, 1]``."""
        urgent_keywords = {"urgent", "urgently", "asap", "immediately", "emergency", "critical"}
        words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
        matches = len(words & urgent_keywords)
        return min(1.0, matches * 0.4 + 0.2)

    @staticmethod
    def classify_issue(text: str) -> str:
        """Classify the issue from the customer message."""
        text_lower = text.lower()
        if any(w in text_lower for w in ("password", "login", "locked", "access")):
            return "account_access"
        if any(w in text_lower for w in ("bill", "charge", "payment", "invoice")):
            return "billing"
        if any(w in text_lower for w in ("bug", "error", "crash", "broken")):
            return "technical"
        return "general"

    @staticmethod
    def assess_difficulty(customer_data: dict[str, Any]) -> float:
        """Estimate how difficult the issue is (0–1)."""
        urgency = customer_data.get("urgency", 0.5)
        sentiment = customer_data.get("sentiment", 0.5)
        # High urgency + low sentiment → harder
        return min(1.0, urgency + (1 - sentiment)) / 2

    @staticmethod
    def get_recommendations(customer_data: dict[str, Any]) -> list[str]:
        """Suggest recommended actions based on customer data."""
        recs: list[str] = []
        if customer_data.get("urgency", 0) > 0.7:
            recs.append("prioritize_response")
        if customer_data.get("sentiment", 1) < 0.4:
            recs.append("empathize_first")
        if not recs:
            recs.append("standard_response")
        return recs

    @staticmethod
    def get_customer_history(customer_id: str | None) -> list[dict[str, Any]]:
        """Retrieve customer interaction history.

        Stub – returns an empty list.  Replace with a real DB lookup.
        """
        return []

    def _latest_customer_sentiment(self) -> float:
        """Get the sentiment from the most recent observation."""
        obs: Observation | None = self.memory.working.get("observation")
        if obs is not None:
            return obs.sensory.get("customer", {}).get("sentiment", 0.5)
        return 0.5
