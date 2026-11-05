from services.workers.worker_utils import resolve_user_uuid
import re
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from models.graph_state import BudAIState
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

@tool("call_analyser_agent", description="Use this tool to delegate deeply analytical queries, historical transaction filtering, spending trend aggregations (month-over-month), and identifying significant spikes in financial behavior.")
async def call_analyser_agent(
    query: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[BudAIState, InjectedState]
):
    """Delegate to the Analyser Agent."""
    try:
        user_uuid = resolve_user_uuid(state)
        
        from agents.core_financial.Analyser_Agent.AnalyserAgent import AnalyserAgent
        agent = AnalyserAgent()
        
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
        err_details = getattr(e, "body", getattr(e, "message", str(e)))
        logger.error(f"Error in analyser worker: {err_details}")

        # Deterministic Fallback: If agent invocation fails, query transactions directly to satisfy query
        try:
            from services.mcp_tools.core_query_tools import query_transactions
            fallback_res = query_transactions(user_uuid=user_uuid)
            fallback_content = (
                f"Direct transaction data retrieval (analyser fallback due to cluster error):\n{fallback_res}"
            )
            return Command(update={
                "messages": [
                    ToolMessage(content=fallback_content, tool_call_id=tool_call_id)
                ]
            })
        except Exception as fb_err:
            logger.error(f"Fallback direct query also failed: {fb_err}")
            return Command(update={
                "messages": [
                    ToolMessage(content=f"Error executing Analyser tool: {str(e)}", tool_call_id=tool_call_id)
                ]
            })
