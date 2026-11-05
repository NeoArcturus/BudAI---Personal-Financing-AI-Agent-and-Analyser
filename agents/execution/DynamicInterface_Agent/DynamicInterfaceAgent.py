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


import uuid
from sqlalchemy import text
from config import SessionLocal
from models.database_models import SystemAlert
import json

@tool
def push_chart_config(chart_json: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Pushes a JSON chart configuration to the frontend WebSocket."""
    user_uuid = state.get("user_uuid")
    try:
        parsed = json.loads(chart_json) if isinstance(chart_json, str) else chart_json
        payload_str = json.dumps(parsed) if isinstance(parsed, dict) else str(chart_json)
        with SessionLocal() as session:
            alert = SystemAlert(
                id=str(uuid.uuid4()),
                user_id=user_uuid,
                title="Dynamic Chart Configuration",
                event_type="DYNAMIC_UI",
                message=payload_str,
                urgency_level=1,
                status_code="308-200",
                action_type="RENDER_CHART",
                action_payload=parsed if isinstance(parsed, dict) else None
            )
            session.add(alert)
            session.commit()
        return "Successfully pushed chart configuration to UI."
    except Exception as e:
        return f"Failed to push chart: {e}"

tools = [push_chart_config]

class DynamicInterfaceAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for DynamicInterfaceAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("DynamicInterfaceAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Dynamic Interface Specialist\n"
                        "You are the Dynamic Interface Agent. Your job is to send JSON configurations to the frontend WebSocket to render custom UI charts. Output valid JSON chart configs.\n\n"
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
