import os
import re
from datetime import datetime
from typing import Annotated, Sequence, TypedDict
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
    cache_id: str
    chart_type: str

@tool
async def generate_expense_forecast_wrapper(account_id: str, days: int, state: Annotated[AgentState, InjectedState]) -> str:
    """Forecast future expenses for selected accounts."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("forecaster", "generate_expense_forecast", {"account_id": account_id, "days": days, "user_uuid": user_uuid})

@tool
async def generate_financial_forecast_wrapper(account_id: str, days: int, discipline_multiplier: float, state: Annotated[AgentState, InjectedState]) -> str:
    """Forecast overall financial balance for selected accounts."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("forecaster", "generate_financial_forecast", {"account_id": account_id, "days": days, "discipline_multiplier": discipline_multiplier, "user_uuid": user_uuid})

@tool
async def generate_hypothetical_scenario_wrapper(account_id: str, days: int, injections: list[dict], state: Annotated[AgentState, InjectedState]) -> str:
    """Generates a financial forecast based on a hypothetical scenario."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("forecaster", "generate_hypothetical_scenario", {"user_uuid": user_uuid, "account_id": account_id, "days": days, "injections": injections})

@tool
async def forecast_budget_impact_wrapper(account_id: str, days: int, state: Annotated[AgentState, InjectedState]) -> str:
    """Run a detailed financial forecast incorporating the user's live budget pacing (Spend Velocity & Variance)."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("forecaster", "forecast_budget_impact", {"account_id": account_id, "days": days, "user_uuid": user_uuid})

@tool
async def get_connected_accounts_wrapper(state: Annotated[AgentState, InjectedState]) -> str:
    """Use this tool to fetch the user's connected account IDs if you need to reference specific accounts."""
    user_uuid = state.get("user_uuid")
    from services.mcp_tools.account_tools import get_connected_accounts
    return get_connected_accounts.invoke({"user_uuid": user_uuid})

tools = [
    generate_expense_forecast_wrapper,
    generate_financial_forecast_wrapper,
    generate_hypothetical_scenario_wrapper,
    forecast_budget_impact_wrapper,
    get_connected_accounts_wrapper
]

class ForecasterAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for ForecasterAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        llm = ChatOpenAI(
            model="Qwen3.5-9B-GGUF",
            base_url=ollama_base_url,
            api_key="ollama",
            temperature=0
        ).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("ForecasterAgent evaluating state...")
            messages = state.get("messages", [])
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        f"### ROLE: Specialist Financial Forecaster\n"
                        f"You project future liquidity and scenarios based on mathematical trends.\n"
                        f"- Date: {current_date_str}\n\n"
                        f"\n\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.\n\n### CRITICAL STRICT ANTI-HALLUCINATION PROTOCOL ###\n"
                        f"1. NO FABRICATION: You are strictly forbidden from fabricating data. Use ONLY data from tool DATA SUMMARY blocks.\n"
                        f"2. TOOL EXECUTION: You have access to specialized tools. You must use them to fetch data when required.\n"
                        f"3. ADMIT IGNORANCE: If a tool returns no data, state 'I do not have the data.' Do not guess.\n"
                        f"4. MULTI-CURRENCY: Respect the native currency returned by the tools (e.g., £, $, €). Do not force GBP. No emojis.\n"
                        f"5. FRESH DATA: Always execute tools for fresh data. Never copy-paste numbers from chat history.\n\n"
                        f"- YOUR SCOPE: Only projections where Year > {current_year_str}.\n\n"
                        f"ROUTING (Use these tools):\n"
                        f"- generate_expense_forecast_wrapper: Forecast future expenses.\n"
                        f"- generate_financial_forecast_wrapper: Forecast overall financial balance.\n"
                        f"- generate_hypothetical_scenario_wrapper: Forecast based on hypothetical events.\n"
                        f"- get_connected_accounts_wrapper: Get account IDs.\n"
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
