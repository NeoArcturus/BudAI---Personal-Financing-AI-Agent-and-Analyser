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
from sqlmodel import select

@tool
def fetch_pending_alerts(state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches all unread system alerts for the user."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        alerts = session.execute(
            select(SystemAlert).where(
                SystemAlert.user_id == user_uuid,
                SystemAlert.is_read == False
            ).order_by(SystemAlert.timestamp.desc())
        ).scalars().all()
        if not alerts:
            return "No pending alerts."
        return "\n".join([f"[Urgency: {a.urgency_level}] {a.title or a.event_type}: {a.message}" for a in alerts])

@tool
def send_daily_digest(digest_content: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Sends the batched daily digest notification to the user and marks prior pending alerts as read."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        session.execute(
            text("UPDATE system_alerts SET is_read = true WHERE user_id = :u AND is_read = false"),
            {"u": user_uuid}
        )
        digest = SystemAlert(
            id=str(uuid.uuid4()),
            user_id=user_uuid,
            title="Daily Financial Digest",
            event_type="DAILY_DIGEST",
            message=digest_content,
            urgency_level=2,
            status_code="308-200"
        )
        session.add(digest)
        session.commit()
    return "Sent daily digest and marked pending alerts as read."

tools = [fetch_pending_alerts, send_daily_digest]

class NotificationBatchingAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for NotificationBatchingAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("NotificationBatchingAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Notification Batching Specialist\n"
                        "You are the Notification Batching Agent. Your job is to intercept system alerts from all other agents and combine them into a single, daily digest to prevent spam. Use fetch_pending_alerts and send_daily_digest.\n\n"
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
