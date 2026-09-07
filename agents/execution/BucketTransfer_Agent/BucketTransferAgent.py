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
def fetch_bucket_balances(state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches the current balances of all virtual buckets."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        res = session.execute(text("SELECT bucket_name, bucket_type, balance FROM buckets WHERE user_uuid = :u"), {"u": user_uuid}).fetchall()
        if not res: return "No buckets found."
        return "\n".join([f"{r[0]} ({r[1]}): {r[2]}" for r in res])

@tool
def execute_virtual_transfer(source_bucket: str, dest_bucket: str, amount: float, state: Annotated[AgentState, InjectedState]) -> str:
    """Executes a strict double-entry virtual transfer between buckets."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        try:
            # Check source balance
            src = session.execute(text("SELECT bucket_uuid, balance FROM buckets WHERE user_uuid = :u AND bucket_name = :n"), {"u": user_uuid, "n": source_bucket}).fetchone()
            dst = session.execute(text("SELECT bucket_uuid, balance FROM buckets WHERE user_uuid = :u AND bucket_name = :n"), {"u": user_uuid, "n": dest_bucket}).fetchone()
            if not src or not dst: return "Invalid bucket names."
            if src[1] < amount: return f"Insufficient funds in {source_bucket}."
            
            # Double entry
            session.execute(text("UPDATE buckets SET balance = balance - :a WHERE bucket_uuid = :id"), {"a": amount, "id": src[0]})
            session.execute(text("UPDATE buckets SET balance = balance + :a WHERE bucket_uuid = :id"), {"a": amount, "id": dst[0]})
            
            session.execute(text("INSERT INTO virtual_transfers (from_bucket_uuid, to_bucket_uuid, amount, user_uuid) VALUES (:f, :t, :a, :u)"), 
                {"f": src[0], "t": dst[0], "a": amount, "u": user_uuid})
            session.commit()
            return f"Successfully transferred {amount} from {source_bucket} to {dest_bucket}."
        except Exception as e:
            session.rollback()
            return f"Transfer failed: {e}"


tools = []

class BucketTransferAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for BucketTransferAgent.
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
            logger.info("BucketTransferAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Bucket Transfer Specialist\n"
                        "You are the Bucket Transfer Agent. Your job is to autonomously sweep detected unallocated income into lower-priority virtual buckets following the user's waterfall strategy. 1. Use fetch_bucket_balances to check balances. 2. Use execute_virtual_transfer to move money. Do NOT hallucinate transfers.\n\n"
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
