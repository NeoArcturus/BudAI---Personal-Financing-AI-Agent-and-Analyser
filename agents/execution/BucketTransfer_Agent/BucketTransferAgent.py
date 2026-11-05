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
from models.database_models import Bucket
from sqlmodel import select

@tool
def fetch_bucket_balances(state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches the current balances of all virtual buckets."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        buckets = session.execute(
            select(Bucket).where(Bucket.user_id == user_uuid, Bucket.is_active == True)
        ).scalars().all()
        if not buckets:
            return "No buckets found for user."
        return "\n".join([f"{b.name} (Type: {b.type.value if hasattr(b.type, 'value') else b.type}, ID: {b.id}): £{b.cached_balance:.2f}" for b in buckets])

@tool
def execute_virtual_transfer(source_bucket: str, dest_bucket: str, amount: float, state: Annotated[AgentState, InjectedState]) -> str:
    """Executes a strict double-entry virtual transfer between buckets. Accepts bucket names or IDs."""
    user_uuid = state.get("user_uuid")
    from services.bucket_service import execute_bucket_transfer as core_transfer
    with SessionLocal() as session:
        try:
            buckets = session.execute(
                select(Bucket).where(Bucket.user_id == user_uuid, Bucket.is_active == True)
            ).scalars().all()
            
            src = next((b for b in buckets if b.id == source_bucket or b.name.lower() == source_bucket.lower()), None)
            dst = next((b for b in buckets if b.id == dest_bucket or b.name.lower() == dest_bucket.lower()), None)
            
            if not src or not dst:
                return f"Invalid bucket names or IDs: source='{source_bucket}', dest='{dest_bucket}'."
            if src.id == dst.id:
                return "Source and destination buckets cannot be the same."
            if amount <= 0:
                return "Transfer amount must be strictly greater than 0."
            if src.cached_balance < amount and getattr(src.type, "value", str(src.type)) != "DEFAULT":
                return f"Insufficient funds in {src.name} (Balance: £{src.cached_balance:.2f}, Requested: £{amount:.2f})."
                
            tx = core_transfer(
                session=session,
                user_uuid=user_uuid,
                source_bucket_id=src.id,
                target_bucket_id=dst.id,
                amount=amount
            )
            session.commit()
            return f"Successfully transferred £{amount:.2f} from {src.name} to {dst.name} (Transfer ID: {tx.id})."
        except Exception as e:
            session.rollback()
            return f"Transfer failed: {str(e)}"

tools = [fetch_bucket_balances, execute_virtual_transfer]

class BucketTransferAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for BucketTransferAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
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
