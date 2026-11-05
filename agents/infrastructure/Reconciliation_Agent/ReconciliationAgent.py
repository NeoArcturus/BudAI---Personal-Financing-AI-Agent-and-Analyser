import os
import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, InjectedState
from sqlalchemy import text
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class ReconState(TypedDict):
    messages: Sequence[BaseMessage]
    user_uuid: str

@tool
def fetch_balances(state: Annotated[ReconState, InjectedState]) -> str:
    """Fetches the physical bank balance and the virtual bucket balance for a user."""
    user_uuid = state.get('user_uuid')
    with SessionLocal() as session:
        p_query = text("SELECT COALESCE(SUM(account_balance), 0) FROM accounts WHERE user_uuid = :uuid")
        physical = session.execute(p_query, {"uuid": user_uuid}).scalar()
        
        v_query = text("SELECT COALESCE(SUM(cached_balance), 0) FROM buckets WHERE user_id = :uuid")
        virtual = session.execute(v_query, {"uuid": user_uuid}).scalar()
        
        return json.dumps({"physical_balance": float(physical), "virtual_balance": float(virtual)})

import uuid
from models.database_models import SystemAlert

@tool
def fire_audit_alert(user_uuid: str, delta: float) -> str:
    """Fires a system alert if the physical and virtual balances do not match."""
    with SessionLocal() as session:
        alert = SystemAlert(
            id=str(uuid.uuid4()),
            user_id=user_uuid,
            title="System Audit Discrepancy",
            event_type="SYSTEM_AUDIT_FAILURE",
            message=f"Zero-sum reconciliation discrepancy detected: physical balance deviates from virtual buckets by £{delta:.2f}.",
            urgency_level=5,
            status_code="308-500"
        )
        session.add(alert)
        session.commit()
    return "Audit failure alert fired successfully."

class DatabaseReconciliationAgent:
    """
    Refactored to be a standalone LangGraph Agent with ChatOpenAI capabilities.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tools = [fetch_balances, fire_audit_alert]
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0)
        llm_with_tools = llm.bind_tools(tools)
        
        def evaluate_node(state: ReconState):
            logger.info("DatabaseReconciliationAgent evaluating state...")
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
            
        def should_continue(state: ReconState):
            messages = state["messages"]
            last_message = messages[-1]
            if not last_message.tool_calls:
                return "end"
            return "continue"
            
        workflow = StateGraph(ReconState)
        workflow.add_node("evaluator", evaluate_node)
        workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("evaluator")
        workflow.add_conditional_edges("evaluator", should_continue, {"continue": "tools", "end": END})
        workflow.add_edge("tools", "evaluator")
        
        return workflow.compile()
        
    @staticmethod
    def audit_zero_sum_math():
        logger.info("Starting Agentic Math Audit...")
        agent = DatabaseReconciliationAgent()
        
        
        with SessionLocal() as session:
            users = session.execute(text("SELECT user_uuid FROM users")).fetchall()
            for user in users:
                user_uuid = user.user_uuid
                messages = [
                    SystemMessage(content='\\n\\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.' + "You are the Database Reconciliation Agent. Use the `fetch_balances` tool. If the physical balance and virtual balance are NOT exactly identical (delta > 0.01), you MUST use the `fire_audit_alert` tool. If they are identical, say 'All good' and exit."),
                    HumanMessage(content=f"Please audit the zero-sum math for user {user_uuid}.")
                ]
                agent.app.invoke({"messages": messages, "user_uuid": user_uuid})
        return True
