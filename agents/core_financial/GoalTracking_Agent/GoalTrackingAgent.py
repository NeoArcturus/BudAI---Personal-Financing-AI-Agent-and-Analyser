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

from config import SessionLocal
from models.database_models import Bucket, BucketType
from sqlmodel import select
from datetime import datetime, timezone

@tool
def check_goal_progress(state: Annotated[AgentState, InjectedState]) -> str:
    """Calculates progress towards target-date and accumulating financial goals."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        goals = session.execute(
            select(Bucket).where(
                Bucket.user_id == user_uuid,
                Bucket.type.in_([BucketType.TARGET_DATE, BucketType.ACCUMULATING]),
                Bucket.is_active == True
            )
        ).scalars().all()
        if not goals:
            return "No target-date or accumulating goal buckets found."
        
        now = datetime.now(timezone.utc)
        results = []
        for g in goals:
            target = g.target_amount or 0.0
            pct = round((g.cached_balance / target) * 100, 1) if target > 0 else 100.0
            days_left = (g.target_date.replace(tzinfo=timezone.utc) - now).days if g.target_date else None
            results.append({
                "goal_name": g.name,
                "current_savings": round(g.cached_balance, 2),
                "target_amount": round(target, 2),
                "progress_pct": pct,
                "shortfall": max(0.0, round(target - g.cached_balance, 2)),
                "days_remaining": days_left
            })
        return json.dumps(results)

tools = [check_goal_progress]

class GoalTrackingAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for GoalTrackingAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("GoalTrackingAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Goal Tracking Specialist\n"
                        "You are the Goal Tracking Agent. Your job is to monitor target-date buckets and alert the user if they fall behind schedule.\n\n"
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
