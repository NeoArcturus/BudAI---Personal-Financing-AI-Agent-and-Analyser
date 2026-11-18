import os
import json
import uuid
from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from sqlmodel import select
from config import SessionLocal
from models.database_models import Bucket, BucketType, VirtualTransfer, TransferSuggestion
from services.logger_setup import get_core_logger
import services.bucket_service as bucket_service

logger = get_core_logger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_uuid: str

@tool
def fetch_bucket_balances(state: Annotated[AgentState, InjectedState]) -> str:
    """Fetches all virtual buckets, their types, balances, and bill timeline statuses."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        buckets = session.execute(
            select(Bucket).where(Bucket.user_id == user_uuid, Bucket.is_active == True)
        ).scalars().all()
        if not buckets:
            return "No buckets found for user."

        lines = []
        for b in buckets:
            b_type = b.type.value if hasattr(b.type, "value") else str(b.type)
            if b_type == BucketType.UNALLOCATED.value:
                curr_breakdown = json.dumps(b.cached_balances or {})
                lines.append(f"Unallocated (Total: £{b.cached_balance:.2f}) - Currencies & Accounts: {curr_breakdown}")
            elif b_type == BucketType.BILLS.value:
                lines.append(f"Bill '{b.name}' ({b.currency}): Allocated £{b.allocated_amount:.2f} of £{b.target_amount or 0:.2f} - Status: {b.status_text or 'Pending'} - Account: {b.associated_account_name or 'N/A'}")
            else:
                lines.append(f"Goal '{b.name}' ({b.currency}, Priority {b.priority}): Saved £{b.cached_balance:.2f} of £{b.target_amount or 0:.2f}")

        return "\n".join(lines)

@tool
def sync_unallocated_from_bank_accounts(state: Annotated[AgentState, InjectedState]) -> str:
    """Syncs the Unallocated bucket with the user's connected bank account balances."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        unallocated = bucket_service.sync_unallocated_from_accounts(session, user_uuid)
        session.commit()
        return f"Synced Unallocated bucket. Total: £{unallocated.cached_balance:.2f} across currencies {list(unallocated.cached_balances.keys())}."

@tool
def process_stage3_bills(bills_json: str, state: Annotated[AgentState, InjectedState]) -> str:
    """
    Receives detected recurring bills from Stage 3, applies the timeline rules,
    runs automated allocation for bills under 15 days, and detects physical account shortfalls.
    """
    user_uuid = state.get("user_uuid")
    try:
        bills_data = json.loads(bills_json) if isinstance(bills_json, str) else bills_json
        if not isinstance(bills_data, list):
            bills_data = [bills_data]

        with SessionLocal() as session:
            result = bucket_service.process_stage3_bills(session, user_uuid, bills_data)
            session.commit()
            return f"Processed {result['processed_count']} bills. {len(result['shortfalls'])} shortfalls flagged."
    except Exception as e:
        return f"Failed to process Stage 3 bills: {str(e)}"

@tool
def deduct_transaction_outflow(transaction_json: str, state: Annotated[AgentState, InjectedState]) -> str:
    """
    Deducts an outgoing expense using the strict hierarchy:
    Bill match -> Unallocated -> Goals (lower priority first) -> Bills (with alert).
    """
    user_uuid = state.get("user_uuid")
    try:
        tx_data = json.loads(transaction_json) if isinstance(transaction_json, str) else transaction_json
        with SessionLocal() as session:
            result = bucket_service.deduct_transaction_outflow(session, user_uuid, tx_data)
            session.commit()
            return json.dumps(result)
    except Exception as e:
        return f"Deduction failed: {str(e)}"

@tool
def execute_virtual_transfer(source_bucket_id: str, target_bucket_id: str, amount: float, state: Annotated[AgentState, InjectedState]) -> str:
    """Moves virtual money between two buckets within the same currency."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        try:
            tx = bucket_service.execute_bucket_transfer(
                session=session,
                user_uuid=user_uuid,
                source_bucket_id=source_bucket_id,
                target_bucket_id=target_bucket_id,
                amount=amount
            )
            session.commit()
            return f"Successfully transferred £{amount:.2f} (Transfer ID: {tx.id})."
        except Exception as e:
            session.rollback()
            return f"Transfer failed: {str(e)}"

@tool
def check_reality_transfers(state: Annotated[AgentState, InjectedState]) -> str:
    """Checks whether physical bank balances updated for approved transfers after 24h or 48h."""
    user_uuid = state.get("user_uuid")
    with SessionLocal() as session:
        reminders = bucket_service.check_reality_transfers(session, user_uuid)
        session.commit()
        if not reminders:
            return "All approved transfers verified or within deadline window."
        return json.dumps(reminders)

tools = [
    fetch_bucket_balances,
    sync_unallocated_from_bank_accounts,
    process_stage3_bills,
    deduct_transaction_outflow,
    execute_virtual_transfer,
    check_reality_transfers
]

class BucketAgent:
    """
    Standalone LangGraph Agent managing the virtual bucket hierarchy,
    bill timeline allocations, and zero-sum operations for Stage 4.
    """
    def __init__(self):
        self.app = self._build_graph()

    def _build_graph(self):
        tool_node = ToolNode(tools)

        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)

        def agent_node(state: AgentState):
            messages = state["messages"]
            system_prompt = (
                "You are BucketAgent, the virtual capital and bill allocator for BudAI.\n"
                "You manage 3 bucket types: 'UNALLOCATED', 'BILLS', and 'GOALS'.\n"
                "All buckets must have balance >= 0.00 at all times.\n"
                "You follow this exact spending hierarchy: Bills > Goals (Unallocated -> Goals [lower-priority first] -> Bills with alert).\n"
                "Automated bill allocation happens when a bill is under 15 days from due date.\n"
                "Do NOT hallucinate transfers. Use your native tools to execute state updates."
            )
            prompt = [SystemMessage(content=system_prompt)] + list(messages)
            response = llm.invoke(prompt)
            return {"messages": [response]}

        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return END

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")

        return workflow.compile()
