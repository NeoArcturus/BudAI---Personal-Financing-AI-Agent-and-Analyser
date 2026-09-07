import os
import json
from datetime import datetime
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, InjectedState
from services.logger_setup import get_core_logger
from services.mcp_bridge import MCPBridge
from services.mcp_tools.external_tools import export_advisory_state, export_custom_statement, search_user_memory, save_to_user_memory

logger = get_core_logger(__name__)

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    user_uuid: str

bridge = MCPBridge()

@tool
async def search_financial_history_semantic_wrapper(query: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Perform a semantic search on the user's transaction history."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("memory", "search_financial_history_semantic", {"query": query, "user_uuid": user_uuid})

@tool
async def get_seasonal_behavior_context_wrapper(state: Annotated[AgentState, InjectedState]) -> str:
    """Retrieves a summary of how the user historically behaves in the current month."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("memory", "get_seasonal_behavior_context", {"user_uuid": user_uuid})

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
    """Ask the user a question for clarification regarding their financial history or preferences."""
    return "Thinking..."

class MemoryAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for MemoryAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tools = [
            search_financial_history_semantic_wrapper,
            get_seasonal_behavior_context_wrapper,
            search_user_memory,
            save_to_user_memory,
            export_advisory_state_wrapper,
            export_custom_statement_wrapper,
            get_connected_accounts_wrapper,
            ask_user
        ]
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
        
        def evaluate_node(state: AgentState):
            logger.info("MemoryAgent evaluating state...")
            messages = state.get("messages", [])
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                current_date_str = datetime.now().strftime("%Y-%m-%d")
                sys_msg = SystemMessage(
                    content=(
                        f"### ROLE: Specialist Knowledge & History Agent\n"
                        f"You retrieve qualitative facts and search multi-year history.\n"
                        f"- Date: {current_date_str}\n\n"
                        f"\n\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.\n\n### CRITICAL STRICT ANTI-HALLUCINATION PROTOCOL ###\n"
                        f"1. NO FABRICATION: You are strictly forbidden from fabricating data. Use ONLY data from tool DATA SUMMARY blocks.\n"
                        f"2. TOOL EXECUTION: You have access to specialized tools. You must use them to fetch data when required.\n"
                        f"3. ADMIT IGNORANCE: If a tool returns no data, state 'I do not have the data.' Do not guess.\n"
                        f"4. MULTI-CURRENCY: Respect the native currency returned by the tools (e.g., £, $, €). Do not force GBP. No emojis.\n\n"
                        f"ROUTING (Use these tools):\n"
                        f"- search_financial_history_semantic_wrapper: Semantic search on transactions.\n"
                        f"- get_seasonal_behavior_context_wrapper: Summary of historical seasonal behavior.\n"
                        f"- search_user_memory: Search user memory.\n"
                        f"- save_to_user_memory: Save to user memory.\n"
                        f"- export_advisory_state_wrapper: Save state.\n"
                        f"- export_custom_statement_wrapper: Generate CSV.\n"
                        f"- get_connected_accounts_wrapper: Get account IDs.\n"
                        f"- ask_user: Ask clarifying questions.\n"
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
