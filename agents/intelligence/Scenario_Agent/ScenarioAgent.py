import os
import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, InjectedState
from services.logger_setup import get_core_logger
from datetime import datetime
from services.mcp_bridge import MCPBridge
from services.mcp_tools.external_tools import export_advisory_state, export_custom_statement

logger = get_core_logger(__name__)
bridge = MCPBridge()

current_date_str = datetime.now().strftime("%Y-%m-%d")

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    user_uuid: str

@tool
async def generate_hypothetical_scenario_wrapper(account_id: str, days: int, injections: list[dict], state: Annotated[AgentState, InjectedState]) -> str:
    """Generates a financial forecast based on a hypothetical scenario."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("forecaster", "generate_hypothetical_scenario", {"user_uuid": user_uuid, "account_id": account_id, "days": days, "injections": injections})

@tool
async def perform_currency_conversion_wrapper(amount: float, from_currency: str, to_currency: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Performs currency conversion using latest forex rates."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("macro", "perform_currency_conversion", {"amount": amount, "from_currency": from_currency, "to_currency": to_currency})

@tool
async def get_live_market_data_wrapper(assets: list[str], state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches real-time price and change for specific tickers."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("macro", "get_live_market_data", {"assets": assets})

@tool
async def get_financial_news_wrapper(query: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches latest financial news/headlines for a topic."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("macro", "get_financial_news", {"query": query})

@tool
async def export_advisory_state_wrapper(chart_type: str, raw_data: dict, ai_analysis: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Saves the current analytical state and AI insights to a persistent JSON file."""
    user_uuid = state.get("user_uuid")
    return export_advisory_state.invoke({"user_uuid": user_uuid, "chart_type": chart_type, "raw_data": raw_data, "ai_analysis": ai_analysis})

@tool
async def export_custom_statement_wrapper(ai_summary: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Generates a downloadable CSV transaction statement with embedded AI analysis."""
    user_uuid = state.get("user_uuid")
    return export_custom_statement.invoke({"user_uuid": user_uuid, "ai_summary": ai_summary})

@tool
async def get_connected_accounts_wrapper(state: Annotated[AgentState, InjectedState]) -> str:
    """Use this tool to fetch the user's connected account IDs if you need to reference specific accounts."""
    user_uuid = state.get("user_uuid")
    from services.mcp_tools.account_tools import get_connected_accounts
    return get_connected_accounts.invoke({"user_uuid": user_uuid})

@tool
async def ask_user(question: str) -> str:
    """Ask the user a question for clarification regarding the scenario planning."""
    return "Thinking..."

class ScenarioAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for ScenarioAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tools = [
            generate_hypothetical_scenario_wrapper, 
            perform_currency_conversion_wrapper, 
            get_live_market_data_wrapper, 
            get_financial_news_wrapper, 
            export_advisory_state_wrapper,
            export_custom_statement_wrapper,
            get_connected_accounts_wrapper,
            ask_user
        ]
        tool_node = ToolNode(tools)
        
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not ollama_base_url.endswith("/v1"):
            ollama_base_url = f"{ollama_base_url}/v1"
            
        llm = ChatOpenAI(
            model="Qwen3.5-9B-GGUF",
            base_url=ollama_base_url,
            api_key="budai-local",
            temperature=0,
            streaming=False,
            max_tokens=4000,
            timeout=600,
        ).bind_tools(tools)
        
        async def evaluate_node(state: AgentState):
            logger.info("ScenarioAgent evaluating state...")
            messages = state.get("messages", [])
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        f"### ROLE: Specialist Scenario Planner & Strategist\n"
                        f"You model complex 'what-if' life events.\n"
                        f"- Date: {current_date_str}\n\n"
                        f"\n\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.\n\n### CRITICAL STRICT ANTI-HALLUCINATION PROTOCOL ###\n"
                        f"1. NO FABRICATION: You are strictly forbidden from fabricating data. Use ONLY data from tool DATA SUMMARY blocks.\n"
                        f"2. TOOL EXECUTION: You have access to specialized tools. You must use them to fetch data when required.\n"
                        f"3. ADMIT IGNORANCE: If a tool returns no data, state 'I do not have the data.' Do not guess.\n"
                        f"4. MULTI-CURRENCY: Respect the native currency returned by the tools (e.g., £, $, €). Do not force GBP. No emojis.\n\n"
                        f"ROUTING (Use these tools):\n"
                        f"- generate_hypothetical_scenario_wrapper: Forecast scenario.\n"
                        f"- perform_currency_conversion_wrapper: FX conversions.\n"
                        f"- get_live_market_data_wrapper: Real-time price of assets.\n"
                        f"- get_financial_news_wrapper: Latest financial news.\n"
                        f"- export_advisory_state_wrapper: Save state.\n"
                        f"- export_custom_statement_wrapper: Generate CSV.\n"
                        f"- get_connected_accounts_wrapper: Get account IDs.\n"
                        f"- ask_user: Ask clarifying questions.\n\n"
                        f"THE USER'S FINANCIAL PROFILE WILL BE PROVIDED IN THE FIRST MESSAGE. USE IT FOR ALL INJECTIONS AND SCENARIOS.\n"
                    )
                )
                messages = [sys_msg] + messages
                
            response = await llm.ainvoke(messages)
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
