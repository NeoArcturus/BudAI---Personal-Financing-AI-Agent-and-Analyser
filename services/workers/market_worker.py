from typing import Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage
from langgraph.types import Command
import re

from models.graph_state import BudAIState
from agents.intelligence.Market_Agent.MarketAgent import MarketAgent
from services.profile_builder import ProfileBuilder
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

@tool("call_market_intelligence", description="Use this tool ONLY for external real-time AND historical market data (stocks, commodities like gold), forex/currency conversions, economic news, or correlating external markets with user spending.")
async def call_market_agent(
    query: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[BudAIState, InjectedState]
):
    """Refined subagent tool for market intelligence."""
    user_uuid = state.get("user_uuid", "ea0e5c07-ab5b-4c14-9ad9-95a036b24637")
    
    profile_builder = ProfileBuilder(user_uuid)
    mrfp = await profile_builder.build_profile()
    
    query_with_context = f"<USER_PROFILE>\n{mrfp}\n</USER_PROFILE>\n\nUSER QUERY:\n{query}"
    
    agent = MarketAgent()
    result = await agent.app.ainvoke({
        "messages": [{"role": "user", "content": query_with_context}], 
        "user_uuid": user_uuid
    })
    
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
