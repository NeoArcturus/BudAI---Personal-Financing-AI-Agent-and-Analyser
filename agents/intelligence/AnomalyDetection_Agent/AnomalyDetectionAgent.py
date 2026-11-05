import os
import json
from typing import Annotated, Sequence, TypedDict, Optional
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

@tool
def scan_for_anomalies(state: Annotated[AgentState, InjectedState], lookback_days: int = 30) -> str:
    """Scans recent transactions for statistical anomalies, z-score spikes, and flagged semantic outliers."""
    user_uuid = state.get("user_uuid")
    from services.mcp_tools.lifestyle_tools import get_semantic_anomalies
    from services.api_integrator.account_reader import AccountReader
    import pandas as pd
    import numpy as np

    semantic_res = get_semantic_anomalies(user_uuid)
    reader = AccountReader(user_id=user_uuid)
    df = reader.get_transactions(user_uuid=user_uuid)
    
    anomalies = []
    if not df.empty and 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        expenses = df[df['amount'] < 0].copy()
        if len(expenses) >= 5:
            amounts = expenses['amount'].abs()
            mean = amounts.mean()
            std = amounts.std()
            if std > 0:
                for _, row in expenses.iterrows():
                    amt = abs(row['amount'])
                    z = (amt - mean) / std
                    if z > 2.5 or (amt > 3.0 * mean and amt > 80.0):
                        anomalies.append({
                            "date": str(row.get('date', '')),
                            "description": str(row.get('description', '')),
                            "amount": float(row['amount']),
                            "z_score": round(float(z), 2),
                            "severity": "CRITICAL" if z > 4.0 else "WARNING"
                        })
    
    return json.dumps({
        "status": "success",
        "semantic_anomalies": semantic_res,
        "statistical_outliers": anomalies[:10],
        "total_statistical_anomalies": len(anomalies)
    })

tools = [scan_for_anomalies]

class AnomalyDetectionAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for AnomalyDetectionAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("AnomalyDetectionAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Anomaly Detection Specialist\n"
                        "You are the Anomaly Detection Agent. Your job is to flag mathematically impossible or extremely out-of-character transactions using standard deviations and historical context.\n\n"
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
