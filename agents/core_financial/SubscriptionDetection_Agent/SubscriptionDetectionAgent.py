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

@tool
def scan_recurring_payments(state: Annotated[AgentState, InjectedState]) -> str:
    """Scans transaction history for recurring payment patterns."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        res = session.execute(text("SELECT merchant_name, amount, frequency FROM subscriptions WHERE user_uuid = :u AND is_active = true"), {"u": user_uuid}).fetchall()
        if not res: return "No active subscriptions found."
        return "\n".join([f"{r[0]}: {r[1]} ({r[2]})" for r in res])

@tool
def flag_subscription_hike(merchant: str, increase_amount: float, state: Annotated[AgentState, InjectedState]) -> str:
    """Flags a subscription price hike for the user."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        session.execute(text("INSERT INTO system_alerts (user_uuid, alert_type, message, severity) VALUES (:u, 'SUBSCRIPTION_HIKE', :m, 'WARNING')"), 
            {"u": user_uuid, "m": f"Subscription hike detected for {merchant}: +{increase_amount}"})
        session.commit()
    return f"Flagged {merchant} hike of {increase_amount}."


tools = []

class SubscriptionDetectionAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for SubscriptionDetectionAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        llm = ChatOpenAI(
            model="Qwen3.5-9B-GGUF",
            base_url=ollama_base_url,
            api_key="ollama",
            temperature=0
        ).bind_tools(tools)
        
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
