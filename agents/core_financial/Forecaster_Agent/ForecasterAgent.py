import os
import re
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from services.logger_setup import get_core_logger
from services.mcp_bridge import MCPBridge
from services.mcp_tools.external_tools import export_advisory_state, export_custom_statement
from models.graph_state import BudAIState

logger = get_core_logger(__name__)
bridge = MCPBridge()

current_date_str = datetime.now().strftime("%Y-%m-%d")
current_year_str = str(datetime.now().year)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_uuid: str

@tool
async def generate_expense_forecast_wrapper(state: Annotated[AgentState, InjectedState], account_id: Optional[str] = None, days: int = 30) -> str:
    """Forecast future cumulative expenses, evaluate monthly run-rate, and compare spend pacing against active bucket goals."""
    user_uuid = state.get("user_uuid")
    payload = {"days": days, "user_uuid": user_uuid}
    if account_id:
        payload["account_id"] = account_id
    return await bridge.call_tool("forecaster", "generate_expense_forecast", payload)

@tool
async def get_connected_accounts_wrapper(state: Annotated[AgentState, InjectedState]) -> str:
    """Use this tool to fetch the user's connected account IDs if you need to reference specific accounts."""
    user_uuid = state.get("user_uuid")
    from services.mcp_tools.account_tools import get_connected_accounts
    return get_connected_accounts.invoke({"user_uuid": user_uuid})

tools = [
    generate_expense_forecast_wrapper,
    get_connected_accounts_wrapper
]

class ForecasterAgent:
    """
    Standalone LangGraph Agent for expenditure trajectory forecasting and goal-aligned pace evaluation.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("ForecasterAgent evaluating state...")
            messages = state.get("messages", [])
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        f"### ROLE: Specialist Financial Forecaster\n"
                        f"You project future expenditure trends, compare spend velocity against prior monthly run-rates, and evaluate pacing against active bucket goals.\n"
                        f"- Current Date: {current_date_str}\n\n"
                        f"### STRICT ROLE BOUNDARIES & EXPERTISE ###\n"
                        f"You are a specialized autonomous agent within the BudAI network. Your role is strictly focused on expense projections and goal pacing.\n"
                        f"Communicate professionally, mathematically, and directly. Avoid conversational filler and never use emojis.\n\n"
                        f"### CRITICAL ANTI-HALLUCINATION PROTOCOL ###\n"
                        f"1. NO FABRICATION: Use strictly verified figures returned by the tool.\n"
                        f"2. TOOL EXECUTION: Always call generate_expense_forecast_wrapper for projections.\n"
                        f"3. MULTI-CURRENCY: Respect the native currency returned by the tools.\n\n"
                        f"ROUTING TOOLS:\n"
                        f"- generate_expense_forecast_wrapper: Forecast future expenses and evaluate goal pacing.\n"
                        f"- get_connected_accounts_wrapper: Fetch user connected account IDs.\n"
                    )
                )
                messages = [sys_msg] + messages
                
            response = llm.invoke(messages)
            return {"messages": [response]}
            
        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END
            
        workflow = StateGraph(AgentState)
        workflow.add_node("evaluator", evaluate_node)
        workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("evaluator")
        workflow.add_conditional_edges("evaluator", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "evaluator")
        
        return workflow.compile()
