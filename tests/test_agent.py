"""Tests for the CognitiveAgent and SupportAgent."""

import pytest

from cognitive_mesh.agent import (
    ActionResult,
    AgentConfig,
    CognitiveAgent,
    CycleResult,
    Observation,
    Orientation,
)
from cognitive_mesh.intentions import Desire, IntentionState, Plan, PlanStep
from cognitive_mesh.examples.support_agent import SupportAgent


class TestCognitiveAgent:
    @pytest.mark.asyncio
    async def test_default_cycle(self) -> None:
        agent = CognitiveAgent(AgentConfig(name="test", role="test"))
        result = await agent.cycle()
        assert isinstance(result, CycleResult)
        assert result.plan.goal == "noop"

    @pytest.mark.asyncio
    async def test_cycle_with_intention(self) -> None:
        agent = CognitiveAgent()
        d = agent.intention_system.add_desire(Desire(goal="do_something"))
        plan = Plan(
            goal="do_something",
            steps=[PlanStep(action="step1"), PlanStep(action="step2")],
        )
        agent.intention_system.commit_intention(d.id, plan)

        result = await agent.cycle()
        assert result.plan.goal == "do_something"
        assert len(result.results) == 2
        assert all(r.success for r in result.results)

    @pytest.mark.asyncio
    async def test_set_context(self) -> None:
        agent = CognitiveAgent()
        agent.set_context({"key": "value"})
        ctx = await agent.get_context()
        assert ctx == {"key": "value"}

    @pytest.mark.asyncio
    async def test_observe_stores_in_working_memory(self) -> None:
        agent = CognitiveAgent()
        obs = await agent.observe()
        stored = agent.memory.working.get("observation")
        assert stored is obs

    @pytest.mark.asyncio
    async def test_orient_stores_in_working_memory(self) -> None:
        agent = CognitiveAgent()
        obs = await agent.observe()
        ori = await agent.orient(obs)
        stored = agent.memory.working.get("orientation")
        assert stored is ori

    def test_strengthen_belief(self) -> None:
        agent = CognitiveAgent()
        agent.beliefs.update("test", True, confidence=0.5)
        agent.strengthen_belief("test")
        belief = agent.beliefs.query("test")
        assert belief is not None
        assert belief.confidence > 0.5

    def test_weaken_belief(self) -> None:
        agent = CognitiveAgent()
        agent.beliefs.update("test", True, confidence=0.8)
        agent.weaken_belief("test")
        belief = agent.beliefs.query("test")
        assert belief is not None
        assert belief.confidence < 0.8

    @pytest.mark.asyncio
    async def test_update_beliefs_from_results(self) -> None:
        agent = CognitiveAgent()
        agent.beliefs.update("b1", True, confidence=0.5)
        agent.beliefs.update("b2", True, confidence=0.5)

        cycle_result = CycleResult(
            plan=Plan(goal="test"),
            results=[
                ActionResult(
                    step=PlanStep(action="a1"), success=True, belief="b1"
                ),
                ActionResult(
                    step=PlanStep(action="a2"), success=False, belief="b2"
                ),
            ],
        )
        await agent.update_beliefs(cycle_result)

        b1 = agent.beliefs.query("b1")
        b2 = agent.beliefs.query("b2")
        assert b1 is not None and b1.confidence > 0.5
        assert b2 is not None and b2.confidence < 0.5


class TestSupportAgent:
    @pytest.mark.asyncio
    async def test_initial_beliefs(self) -> None:
        agent = SupportAgent()
        belief = agent.beliefs.query("customers_want_fast_responses")
        assert belief is not None
        assert belief.confidence == 0.9

    @pytest.mark.asyncio
    async def test_initial_desires(self) -> None:
        agent = SupportAgent()
        desires = agent.intention_system.get_desires()
        assert len(desires) == 2
        assert desires[0].goal == "resolve_customer_issue"

    @pytest.mark.asyncio
    async def test_urgent_cycle(self) -> None:
        agent = SupportAgent()
        agent.set_context({
            "message": "My account is locked urgently need help immediately!",
            "customer_id": "cust_1",
        })
        result = await agent.cycle()
        assert result.plan.goal == "rapid_response"
        assert len(result.results) == 3

    @pytest.mark.asyncio
    async def test_normal_cycle(self) -> None:
        agent = SupportAgent()
        agent.set_context({
            "message": "I have a question about my bill.",
            "customer_id": "cust_2",
        })
        result = await agent.cycle()
        assert isinstance(result, CycleResult)

    def test_analyse_sentiment(self) -> None:
        assert SupportAgent.analyse_sentiment("This is great, thanks!") > 0.5
        assert SupportAgent.analyse_sentiment("Terrible broken service") < 0.5
        assert SupportAgent.analyse_sentiment("Hello") == 0.5

    def test_assess_urgency(self) -> None:
        assert SupportAgent.assess_urgency("urgent help needed immediately") > 0.5
        assert SupportAgent.assess_urgency("just a quick question") < 0.5

    def test_classify_issue(self) -> None:
        assert SupportAgent.classify_issue("My account is locked") == "account_access"
        assert SupportAgent.classify_issue("billing issue") == "billing"
        assert SupportAgent.classify_issue("app crash bug") == "technical"
        assert SupportAgent.classify_issue("hello there") == "general"
