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

@tool("call_health_strategist", description="Use this tool ONLY for personal financial wellness checks, user emergency fund metrics, user debt analysis, or user wealth acceleration.")
async def call_health_agent(
    query: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[BudAIState, InjectedState]
):
    """Refined subagent tool for financial health strategy."""
    try:
        user_uuid = state.get("user_uuid", "ea0e5c07-ab5b-4c14-9ad9-95a036b24637")
        
        from agents.intelligence.Health_Agent.HealthAgent import HealthAgent
        agent = HealthAgent()
        
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
        import traceback
        error_msg = f"CRITICAL HEALTH TOOL CRASH: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return Command(update={
            "messages": [
                ToolMessage(content=f"Error executing health tool: {str(e)}", tool_call_id=tool_call_id)
            ]
        })
