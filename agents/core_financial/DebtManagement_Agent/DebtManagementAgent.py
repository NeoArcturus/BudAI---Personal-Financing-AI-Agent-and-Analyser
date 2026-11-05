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


from sqlalchemy import text
from config import SessionLocal
from models.database_models import Bucket, BucketType
from sqlmodel import select
from services.api_integrator.account_reader import AccountReader
import pandas as pd

@tool
def fetch_liabilities(state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches all liability buckets and their balances."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        liabilities = session.execute(
            select(Bucket).where(
                Bucket.user_id == user_uuid,
                Bucket.type == BucketType.LIABILITY,
                Bucket.is_active == True
            )
        ).scalars().all()
        if not liabilities:
            return "No liability buckets found for this user."
        return "\n".join([f"{b.name} (ID: {b.id}): Balance=£{b.cached_balance:.2f}, Priority Index={b.priority_index}" for b in liabilities])

@tool
def calculate_payoff_velocity(strategy: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Calculates time to debt freedom based on strategy (avalanche/snowball) using empirical monthly surplus."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        liabilities = session.execute(
            select(Bucket).where(
                Bucket.user_id == user_uuid,
                Bucket.type == BucketType.LIABILITY,
                Bucket.is_active == True
            )
        ).scalars().all()
        if not liabilities:
            return "No liabilities to calculate payoff velocity."
            
        total_debt = sum(abs(b.cached_balance) for b in liabilities)
        
        # Estimate monthly net cashflow from 90 days of transactions
        reader = AccountReader(user_id=user_uuid)
        df = reader.get_transactions(user_uuid=user_uuid)
        monthly_surplus = 200.0
        if not df.empty and 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
            inflows = df[df['amount'] > 0]['amount'].sum()
            outflows = df[df['amount'] < 0]['amount'].abs().sum()
            net_90 = inflows - outflows
            monthly_surplus = max(float(net_90 / 3.0), 50.0)
            
        months = total_debt / monthly_surplus if monthly_surplus > 0 else 999.0
        
        # Strategy breakdown
        # Snowball: lowest balance first
        # Avalanche: highest priority index / APR first
        ordered = sorted(liabilities, key=lambda x: abs(x.cached_balance)) if strategy.lower() == "snowball" else sorted(liabilities, key=lambda x: x.priority_index, reverse=True)
        priority_order = [f"{b.name}: £{abs(b.cached_balance):.2f}" for b in ordered]
        
        return json.dumps({
            "strategy": strategy.upper(),
            "total_debt": round(total_debt, 2),
            "estimated_monthly_surplus": round(monthly_surplus, 2),
            "months_to_debt_free": round(months, 1),
            "priority_order": priority_order
        })

tools = [fetch_liabilities, calculate_payoff_velocity]

class DebtManagementAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for DebtManagementAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("DebtManagementAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Debt Management Specialist\n"
                        "You are the Debt Management Agent. Your job is to monitor liability buckets and calculate optimal payoff velocity based on interest rates. Recommend Avalanche or Snowball strategies.\n\n"
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
