# Cognitive-Mesh

A Python implementation of the **Cognitive Mesh Protocol** – combining
[BDI (Belief-Desire-Intention)](https://en.wikipedia.org/wiki/Belief%E2%80%93desire%E2%80%93intention_software_model)
agent architecture with the
[OODA (Observe-Orient-Decide-Act)](https://en.wikipedia.org/wiki/OODA_loop)
loop for building production-ready AI agents.

Based on the article: [Real-World Implementation of Cognitive Architectures for AI Agents](https://www.loonix.pt/articles/bdi-ooda-production.html)

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              COGNITIVE MESH PROTOCOL                 │
├──────────────────────────────────────────────────────┤
│  BDI LAYER (Deliberate Reasoning)                    │
│    Beliefs → Desires → Intentions                    │
│                                                      │
│  OODA LAYER (Rapid Response)                         │
│    Observe → Orient → Decide → Act                   │
│                                                      │
│  MEMORY LAYER (Context Management)                   │
│    Short-Term  │  Long-Term  │  Working Memory       │
└──────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/RichardScottOZ/Cognitive-Mesh.git
cd Cognitive-Mesh

# Install test dependencies
pip install pytest pytest-asyncio
```

## Quick Start

```python
import asyncio
from cognitive_mesh import CognitiveAgent, AgentConfig, Desire, Plan, PlanStep

async def main():
    # Create an agent
    agent = CognitiveAgent(AgentConfig(name="my-agent", role="demo"))

    # Add a desire (goal)
    desire = agent.intention_system.add_desire(
        Desire(goal="greet_user", priority=0.9)
    )

    # Commit to a plan
    agent.intention_system.commit_intention(
        desire.id,
        Plan(goal="greet_user", steps=[PlanStep(action="say_hello")])
    )

    # Run one OODA cycle
    result = await agent.cycle()
    print(f"Plan: {result.plan.goal}, Steps: {len(result.results)}")

asyncio.run(main())
```

### Customer Support Agent Example

```python
import asyncio
from cognitive_mesh.examples.support_agent import SupportAgent

async def main():
    agent = SupportAgent()
    agent.set_context({
        "message": "My account is locked and I need help urgently!",
        "customer_id": "cust_42",
    })
    result = await agent.cycle()
    print(f"Plan: {result.plan.goal}")  # "rapid_response"

asyncio.run(main())
```

## Package Structure

| Module | Description |
|--------|-------------|
| `cognitive_mesh.beliefs` | `BeliefSystem` with Bayesian confidence updates |
| `cognitive_mesh.intentions` | `IntentionSystem` managing desires and intentions |
| `cognitive_mesh.memory` | Three-layer memory: short-term, long-term, working |
| `cognitive_mesh.agent` | `CognitiveAgent` base class with OODA loop |
| `cognitive_mesh.examples.support_agent` | Domain-specific `SupportAgent` example |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Key Concepts

- **Beliefs** – What the agent knows, with Bayesian confidence tracking
- **Desires** – Goals the agent wants to achieve, ordered by priority
- **Intentions** – Committed plans to satisfy desires
- **OODA Cycle** – Observe → Orient → Decide → Act loop for rapid response
- **Memory** – Short-term (with TTL), long-term (with importance pruning), and working memory
