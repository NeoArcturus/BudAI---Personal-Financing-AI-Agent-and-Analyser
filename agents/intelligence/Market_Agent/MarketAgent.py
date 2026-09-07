import os
import re
from datetime import datetime
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from services.logger_setup import get_core_logger
from services.mcp_bridge import MCPBridge
from services.mcp_tools.market_tools import get_historical_market_data, compare_spending_to_market
from services.mcp_tools.external_tools import export_advisory_state, export_custom_statement

logger = get_core_logger(__name__)

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    user_uuid: str

bridge = MCPBridge()

@tool
async def get_live_market_data_wrapper(assets: list[str], state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches real-time price and change for specific tickers."""
    return await bridge.call_tool("macro", "get_live_market_data", {"assets": assets})

@tool
async def get_financial_news_wrapper(query: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches latest financial news/headlines for a topic."""
    return await bridge.call_tool("macro", "get_financial_news", {"query": query})

@tool
async def get_historical_market_data_wrapper(ticker: str, state: Annotated[AgentState, InjectedState], period: str = "6mo") -> str:
    """Fetches historical price trends and volatility for a specific ticker."""
    return get_historical_market_data.invoke({"ticker": ticker, "period": period})

@tool
async def compare_spending_to_market_wrapper(category: str, ticker: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Analyzes how a specific market asset correlates with user spending."""
    user_uuid = state.get("user_uuid")
    return compare_spending_to_market.invoke({"user_uuid": user_uuid, "category": category, "ticker": ticker})

@tool
async def perform_currency_conversion_wrapper(amount: float, from_currency: str, to_currency: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Performs currency conversion between two currencies."""
    return await bridge.call_tool("macro", "perform_currency_conversion", {"amount": amount, "from_currency": from_currency, "to_currency": to_currency})

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
    """Ask the user a question for clarification regarding market analysis."""
    return "Thinking..."

tools = [
    get_live_market_data_wrapper,
    get_financial_news_wrapper,
    get_historical_market_data_wrapper,
    compare_spending_to_market_wrapper,
    perform_currency_conversion_wrapper,
    export_advisory_state_wrapper,
    export_custom_statement_wrapper,
    get_connected_accounts_wrapper,
    ask_user
]

class MarketAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for MarketAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
            
        llm = ChatOpenAI(
            model="Qwen3.5-9B-GGUF",
            base_url=base_url,
            api_key="budai-local",
            temperature=0,
            streaming=False,
            max_tokens=4000,
            timeout=600,
        ).bind_tools(tools)
        
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        
        system_prompt = f"""### ROLE: Specialist Market Intelligence Agent
You correlate global market trends with personal finances. Your job is to monitor macroeconomic trends, stock prices, and interest rates, alerting the user to external risks.
- Date: {current_date_str}

### CRITICAL STRICT ANTI-HALLUCINATION PROTOCOL ###
1. NO FABRICATION: You are strictly forbidden from fabricating data. Use ONLY data from tool DATA SUMMARY blocks.
2. TOOL EXECUTION: You have access to specialized tools. You must use them to fetch data when required.
3. ADMIT IGNORANCE: If a tool returns no data, state "I do not have the data." Do not guess.
4. MULTI-CURRENCY: Respect the native currency returned by the tools (e.g., £, $, €). Do not force GBP. No emojis.
5. FRESH DATA: Always execute tools for fresh data. Never copy-paste numbers from chat history.

ROUTING (Use these tools):
- get_live_market_data_wrapper: Real-time price of assets.
- get_financial_news_wrapper: Latest financial news.
- get_historical_market_data_wrapper: Historical price trends.
- compare_spending_to_market_wrapper: Market correlation with spending.
- perform_currency_conversion_wrapper: FX conversions.
- export_advisory_state_wrapper: Save state.
- export_custom_statement_wrapper: Generate CSV.
- get_connected_accounts_wrapper: Get account IDs.
- ask_user: Ask clarifying questions.

     - Gold: 'GC=F', Silver: 'SI=F', Copper: 'HG=F'
     - Oil (Brent): 'BZ=F', Oil (WTI): 'CL=F', Natural Gas: 'NG=F', Heating Oil: 'HO=F'
     - Wheat: 'ZW=F', Corn: 'ZC=F', Sugar: 'SB=F', Cocoa: 'CC=F', Coffee: 'KC=F'
     - S&P 500: '^GSPC', FTSE 100: '^FTSE', FTSE 250: '^FTMC'
"""
        
        async def evaluate_node(state: AgentState):
            logger.info("MarketAgent evaluating state...")
            messages = state.get("messages", [])
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(content='\\n\\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.' + system_prompt)
                messages = [sys_msg] + list(messages)
                
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
