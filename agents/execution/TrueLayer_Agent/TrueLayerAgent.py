import os
import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_uuid: str

@tool
def trigger_account_sync(account_id: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Synchronizes physical bank data for a specific account via TrueLayer."""
    user_uuid = state.get("user_uuid")
    if not user_uuid:
        return "Error: Missing user_uuid in state."
    try:
        from services.api_integrator.truelayer_sync import TrueLayerSync
        syncer = TrueLayerSync(user_id=user_uuid)
        syncer.trigger_sync(account_id=account_id, user_uuid=user_uuid)
        return f"Successfully triggered sync for account {account_id}."
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return f"Failed to sync account: {str(e)}"

@tool
def refresh_bank_tokens(provider_id: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Refreshes the OAuth access tokens for a specific bank provider."""
    user_uuid = state.get("user_uuid")
    if not user_uuid:
        return "Error: Missing user_uuid in state."
    try:
        from services.api_integrator.access_token_generator import AccessTokenGenerator
        token_gen = AccessTokenGenerator()
        success = token_gen.refresh_token(provider_id, user_uuid)
        if success:
            return f"Successfully refreshed tokens for provider {provider_id}."
        return f"Failed to refresh tokens for provider {provider_id}."
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        return f"Failed to refresh tokens: {str(e)}"

tools = [trigger_account_sync, refresh_bank_tokens]

class TrueLayerAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for TrueLayer integration.
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
            logger.info("TrueLayerAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: TrueLayer Integration Agent\n"
                        "You are the TrueLayer Agent for BudAI. Your job is to manage physical bank integrations, safely process incoming webhook events, and ensure OAuth token synchronization.\n\n"
                        "\n\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.\n\n### STRICT ANTI-HALLUCINATION PROTOCOL ###\n"
                        "1. You MUST use the `trigger_account_sync` tool to fetch physical bank data for an account. Do not make up transactions.\n"
                        "2. You MUST use `refresh_bank_tokens` if tokens are expired.\n"
                        "3. Never expose raw API keys or access tokens to the user.\n"
                        "4. Ensure that all webhook events are properly verified before processing.\n"
                    )
                )
                messages.insert(0, sys_msg)
                
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
