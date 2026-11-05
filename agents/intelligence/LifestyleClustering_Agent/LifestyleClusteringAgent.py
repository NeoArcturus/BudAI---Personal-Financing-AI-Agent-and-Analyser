import os
import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_uuid: str

from services.mcp_tools.lifestyle_tools import (
    get_user_lifestyle_profile,
    get_semantic_anomalies,
    benchmark_persona_budget,
    get_lifestyle_trajectory
)

@tool
def get_user_lifestyle_profile_wrapper(state: Annotated[AgentState, InjectedState]) -> str:
    """Retrieves the user's Macro-Persona and top HDBSCAN micro-lifestyles."""
    user_uuid = state.get("user_uuid")
    return get_user_lifestyle_profile.invoke({"user_uuid": user_uuid})

@tool
def get_semantic_anomalies_wrapper(state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches recent transactions flagged as semantic outliers."""
    user_uuid = state.get("user_uuid")
    return get_semantic_anomalies.invoke({"user_uuid": user_uuid})

@tool
def benchmark_persona_budget_wrapper(state: Annotated[AgentState, InjectedState]) -> str:
    """Benchmarks the user's category spending velocity against demographic peers."""
    user_uuid = state.get("user_uuid")
    return benchmark_persona_budget.invoke({"user_uuid": user_uuid})

@tool
def get_lifestyle_trajectory_wrapper(state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches the temporal shift and trajectory across user micro-lifestyle clusters over time."""
    user_uuid = state.get("user_uuid")
    return get_lifestyle_trajectory.invoke({"user_uuid": user_uuid})

tools = [
    get_user_lifestyle_profile_wrapper,
    get_semantic_anomalies_wrapper,
    benchmark_persona_budget_wrapper,
    get_lifestyle_trajectory_wrapper
]

class LifestyleClusteringAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for LifestyleClusteringAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("LifestyleClusteringAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Lifestyle Clustering Specialist\n"
                        "You are the Lifestyle Clustering Agent. Your job is to use HDBSCAN to group merchants and identify long-term lifestyle shifts.\n\n"
                        "\n\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.\n\n### STRICT ANTI-HALLUCINATION PROTOCOL ###\n"
                        "1. You MUST use your dedicated tools to perform actions or retrieve data.\n"
                        "2. Never hallucinate financial data or system states.\n"
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
