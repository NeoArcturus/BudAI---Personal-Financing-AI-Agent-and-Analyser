from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field

class AgentCard(BaseModel):
    """
    Agent Capability Card defining an agent's domain, handled intents, and SLA.
    """
    name: str
    description: str
    intents_handled: List[str]
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    avg_latency_ms: float = Field(default=100.0)

class AgentRegistry:
    """
    In-memory dynamic registry for peer agent discovery and routing.
    """
    def __init__(self):
        self._agents: Dict[str, AgentCard] = {}
        self._handlers: Dict[str, Callable] = {}

    def register(self, card: AgentCard, handler: Callable):
        self._agents[card.name] = card
        self._handlers[card.name] = handler

    def get_agent(self, name: str) -> Optional[AgentCard]:
        return self._agents.get(name)

    def find_agent_for_intent(self, intent: str) -> Optional[str]:
        for name, card in self._agents.items():
            if intent in card.intents_handled:
                return name
        return None

    def get_handler(self, name: str) -> Optional[Callable]:
        return self._handlers.get(name)

    def list_agents(self) -> List[AgentCard]:
        return list(self._agents.values())

_registry_instance: Optional[AgentRegistry] = None

def get_agent_registry() -> AgentRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = AgentRegistry()
    return _registry_instance
