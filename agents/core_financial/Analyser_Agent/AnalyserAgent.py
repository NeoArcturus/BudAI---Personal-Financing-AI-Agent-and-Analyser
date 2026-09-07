from services.mcp_tools.core_query_tools import query_transactions, aggregate_financial_data
import os
import uuid
from typing import TypedDict, Annotated, Sequence
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from sqlmodel import select, Session
from config import engine
from models.database_models import Transaction, SystemAlert

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_uuid: str

@tool
def send_system_alert(event_type: str, message: str, user_uuid: Annotated[str, InjectedState("user_uuid")] = "") -> str:
    """Inserts a row into the system_alerts table."""
    with Session(engine) as session:
        alert = SystemAlert(
            id=str(uuid.uuid4()),
            user_id=user_uuid,
            event_type=event_type,
            message=message,
            timestamp=datetime.utcnow()
        )
        session.add(alert)
        session.commit()
    return "Alert sent successfully."

tools = [query_transactions, aggregate_financial_data, send_system_alert]

class AnalyserAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for proactive insights.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        llm = ChatOpenAI(
            model="Qwen3.5-9B-GGUF",
            base_url=ollama_base_url,
            api_key="ollama"
        ).bind_tools(tools)
        
        tool_node = ToolNode(tools)

        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        def call_model(state: AgentState):
            messages = state["messages"]
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Proactive Financial Analyser\n"
                        "You are the Analyser Agent for BudAI. Your job is to monitor and query spending trends for the user to proactively identify financial risks, spikes, or irregularities. Respect the native currency of the transaction (e.g., £, $, €). Do not force GBP.\n\n"
                        "\n\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.\n\n### STRICT ANTI-HALLUCINATION PROTOCOL ###\n"
                        "1. You MUST use the `aggregate_financial_data` and `query_transactions` tools to fetch accurate mathematical data before making any conclusions.\n"
                        "2. If you see a mathematically significant spike (e.g., 30% or more in a category compared to last month), you MUST use the `send_system_alert` tool to alert the user.\n"
                        "3. Do not make up numbers. Use only the SQL output provided by your tools.\n"
                        "4. Be proactive but concise. DO NOT use emojis in your response under any circumstances.\n"
                    )
                )
                messages = [sys_msg] + messages
                
            response = llm.invoke(messages)
            return {"messages": [response]}
            
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
