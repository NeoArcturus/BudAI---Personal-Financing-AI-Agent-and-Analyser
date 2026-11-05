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

from services.api_integrator.account_reader import AccountReader
import pandas as pd

@tool
def analyze_income_streams(state: Annotated[AgentState, InjectedState]) -> str:
    """Analyzes incoming transactions to identify true payroll deposits vs ad-hoc transfers."""
    user_uuid = state.get("user_uuid")
    reader = AccountReader(user_id=user_uuid)
    df = reader.get_transactions(user_uuid=user_uuid)
    if df.empty or 'amount' not in df.columns:
        return json.dumps({"status": "no_data", "message": "No transaction history available."})
    
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    inflows = df[df['amount'] > 0].copy()
    if inflows.empty:
        return json.dumps({"status": "no_inflows", "message": "No incoming deposits recorded."})
    
    # Group by description
    grouped = inflows.groupby('description').agg(
        total_amount=('amount', 'sum'),
        count=('amount', 'count'),
        avg_amount=('amount', 'mean')
    ).reset_index()
    
    streams = []
    for _, row in grouped.iterrows():
        count = int(row['count'])
        avg_val = round(float(row['avg_amount']), 2)
        desc = str(row['description'])
        is_payroll = count >= 2 and avg_val >= 500.0
        streams.append({
            "source": desc,
            "stream_type": "PAYROLL" if is_payroll else "TRANSFER_OR_OTHER",
            "frequency_occurrences": count,
            "average_amount": avg_val,
            "total_received": round(float(row['total_amount']), 2)
        })
        
    return json.dumps({
        "status": "success",
        "total_inflows_recorded": round(float(inflows['amount'].sum()), 2),
        "income_streams": sorted(streams, key=lambda x: x['total_received'], reverse=True)
    })

tools = [analyze_income_streams]

class IncomeDetectionAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for IncomeDetectionAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("IncomeDetectionAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Income Detection Specialist\n"
                        "You are the Income Detection Agent. Your job is to differentiate standard transfers from true payroll deposits. Flag any unexpected income drops.\n\n"
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
