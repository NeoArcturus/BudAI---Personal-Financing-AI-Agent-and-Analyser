from langchain_core.tools import tool
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from models.graph_state import BudAIState
from langchain_core.runnables import RunnableConfig
from services.logger_setup import get_core_logger
import re

logger = get_core_logger(__name__)

@tool("call_monte_carlo_simulation_agent", description="Use this tool to delegate advanced stochastic modeling, simulating thousands of market/spending paths, and determining the probabilistic success rate of long-term financial plans.")
async def call_monte_carlo_simulation_agent(
    query: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[BudAIState, InjectedState]
):
    """Delegate to the MonteCarloSimulation Agent."""
    try:
        user_uuid = state.get("user_uuid", "ea0e5c07-ab5b-4c14-9ad9-95a036b24637")
        
        from agents.intelligence.MonteCarloSimulation_Agent.MonteCarloSimulationAgent import MonteCarloSimulationAgent
        agent = MonteCarloSimulationAgent()
        
        result = await agent.app.ainvoke({"messages": [{"role": "user", "content": query}], "user_uuid": user_uuid})
        raw_output = result["messages"][-1].content
        output = raw_output if isinstance(raw_output, str) else "".join([b if isinstance(
            b, str) else b.get("text", "") for b in raw_output if isinstance(b, (str, dict))])
        
        cache_id, chart_type = None, None
        match = re.search(r'\[TRIGGER_([A-Z_]+):([^\]]+)\]', output)
        if match:
            chart_type = match.group(1)
            cache_id = match.group(2).split(':')[0]
            output = re.sub(r'\[TRIGGER_[A-Z_]+:[^\]]+\]', '', output).strip()

        return Command(update={
            "cache_id": cache_id,
            "chart_type": chart_type,
            "messages": [
                ToolMessage(content=output, tool_call_id=tool_call_id)
            ]
        })
    except Exception as e:
        logger.error(f"Error in MonteCarloSimulation worker: {e}")
        return Command(update={
            "messages": [
                ToolMessage(content=f"Error executing MonteCarloSimulation tool: {str(e)}", tool_call_id=tool_call_id)
            ]
        })
