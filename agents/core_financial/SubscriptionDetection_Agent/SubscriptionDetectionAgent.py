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
from models.database_models import SystemAlert, Subscription
from sqlmodel import select

@tool
def scan_recurring_payments(state: Annotated[AgentState, InjectedState]) -> str:
    """Scans transaction history for recurring payment patterns and active subscriptions."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        subs = session.execute(
            select(Subscription).where(Subscription.user_uuid == user_uuid)
        ).scalars().all()
        if not subs:
            return "No active subscriptions found."
        lines = []
        for s in subs:
            hike_note = " [PRICE HIKE DETECTED]" if s.is_price_hike else ""
            lines.append(f"{s.merchant_name}: £{s.expected_amount:.2f} ({s.predicted_frequency}) Next: {s.next_expected_date.strftime('%Y-%m-%d')}{hike_note}")
        return "\n".join(lines)

@tool
def flag_subscription_hike(merchant: str, increase_amount: float, state: Annotated[AgentState, InjectedState]) -> str:
    """Flags a subscription price hike for the user and persists a system alert."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        alert = SystemAlert(
            id=str(uuid.uuid4()),
            user_id=user_uuid,
            event_type="SUBSCRIPTION_HIKE",
            title=f"Price Hike: {merchant}",
            message=f"Subscription price increase detected for {merchant}: +£{increase_amount:.2f}",
            urgency_level=3,
            status_code="308-200"
        )
        session.add(alert)
        session.commit()
    return f"Flagged {merchant} hike of £{increase_amount:.2f}."

tools = [scan_recurring_payments, flag_subscription_hike]

class SubscriptionDetectionAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for SubscriptionDetectionAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("SubscriptionDetectionAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Subscription Detection Specialist\n"
                        "You are the Subscription Detection Agent. Your job is to scan recurring payments and flag unexpected price hikes. 1. Use scan_recurring_payments to find subscriptions. 2. Use flag_subscription_hike to alert the user if a subscription increases by more than 10%.\n\n"
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
