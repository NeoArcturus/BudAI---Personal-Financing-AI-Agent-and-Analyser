import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from sqlalchemy import text
from config import SessionLocal
from models.database_models import Transaction, Account
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class IntakeState(TypedDict):
    messages: Annotated[list, add_messages]
    user_uuid: str

def parse_transaction_timestamp(date_str: Optional[str]) -> datetime:
    """Parses ISO timestamp strings into UTC datetime objects safely."""
    if not date_str:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.utcnow()

def compute_dedup_hash(user_uuid: str, account_id: str, date_val: datetime, amount: float, desc: str) -> str:
    """Deterministic hash for deduplicating transactions."""
    raw = f"{user_uuid}_{account_id}_{date_val.strftime('%Y-%m-%d')}_{amount:.2f}_{desc.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()

def process_transaction_lifecycle_item(session, tx: Dict[str, Any], user_uuid: str, account_id: str, bank_uuid: str) -> Dict[str, Any]:
    """
    Evaluates a single transaction entity:
    - Matches pending to settled transitions.
    - Segregates transfers, salary inflows, and consumer expenses.
    - Upserts clean records into PostgreSQL.
    """
    raw_status = str(tx.get("status", "SETTLED")).upper()
    is_pending = raw_status == "PENDING"
    amount = float(tx.get("amount", 0.0))
    currency = str(tx.get("currency", "GBP"))
    original_desc = str(tx.get("description", ""))
    norm_id = tx.get("normalised_provider_transaction_id") or tx.get("provider_transaction_id")
    date_val = parse_transaction_timestamp(tx.get("timestamp") or tx.get("date"))
    tx_type = str(tx.get("transaction_type", "DEBIT")).upper()
    category = str(tx.get("transaction_category", "")).upper()

    tx_hash = compute_dedup_hash(user_uuid, account_id, date_val, amount, original_desc)
    provider_id = norm_id or tx_hash

    # 1. State Transition: Match pending to settled
    if not is_pending and norm_id:
        existing_pending = session.execute(
            text("SELECT transaction_uuid FROM transactions WHERE provider_transaction_id = :pid AND is_pending = TRUE"),
            {"pid": norm_id}
        ).fetchone()

        if existing_pending:
            tx_uuid = existing_pending[0]
            session.execute(
                text("UPDATE transactions SET is_pending = FALSE, amount = :amt, date = :dt WHERE transaction_uuid = :uid"),
                {"amt": amount, "dt": date_val, "uid": tx_uuid}
            )
            session.commit()
            return {
                "action": "TRANSITION_SETTLED",
                "transaction_uuid": tx_uuid,
                "amount": amount,
                "is_pending": False,
                "nature": "OUTFLOW"
            }

    # 2. Segregate: Internal Transfers
    desc_upper = original_desc.upper()
    if "TRANSFER" in category or "CREDIT CARD PAYMENT" in desc_upper or "BILL PAYMENT" in category:
        return {
            "action": "INTERNAL_TRANSFER",
            "amount": amount,
            "currency": currency,
            "description": original_desc,
            "nature": "TRANSFER"
        }

    # 3. Segregate: Salary / Inflows
    if tx_type == "CREDIT" or (amount > 0 and "SALARY" in desc_upper):
        return {
            "action": "SALARY_INFLOW",
            "amount": amount,
            "currency": currency,
            "source": tx.get("merchant_name") or original_desc,
            "timestamp": date_val.isoformat(),
            "nature": "INFLOW"
        }

    # 4. Standard Consumer Outflow: Upsert into PostgreSQL
    tx_uuid = str(uuid.uuid4())
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    
    stmt = pg_insert(Transaction).values({
        "transaction_uuid": tx_uuid,
        "user_uuid": user_uuid,
        "bank_uuid": bank_uuid,
        "account_id": account_id,
        "provider_transaction_id": provider_id,
        "date": date_val,
        "amount": amount,
        "currency": currency,
        "description": original_desc,
        "semi_cleaned_description": original_desc.lower(),
        "fully_cleaned_description": original_desc.lower(),
        "is_pending": is_pending
    })
    stmt = stmt.on_conflict_do_update(
        constraint="uq_transaction_account_provider",
        set_={
            "amount": stmt.excluded.amount,
            "date": stmt.excluded.date,
            "is_pending": stmt.excluded.is_pending,
            "description": stmt.excluded.description
        }
    )
    session.execute(stmt)
    session.commit()

    return {
        "action": "NEW_EXPENSE",
        "transaction_uuid": tx_uuid,
        "provider_id": provider_id,
        "amount": amount,
        "currency": currency,
        "clean_merchant": tx.get("merchant_name") or original_desc,
        "raw_description": original_desc,
        "classifications": tx.get("transaction_classification", []),
        "is_pending": is_pending,
        "timestamp": date_val.isoformat(),
        "nature": "OUTFLOW"
    }

@tool
def process_incoming_transactions_tool(user_uuid: str, account_id: str, bank_uuid: str, transactions_json: str) -> str:
    """
    Processes a list of raw transaction dictionaries:
    Resolves pending-to-settled transitions, filters internal transfers,
    stores verified records in the database, and returns classified entities.
    """
    try:
        tx_list = json.loads(transactions_json)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Invalid JSON payload: {e}"})

    results = []
    with SessionLocal() as session:
        for tx in tx_list:
            outcome = process_transaction_lifecycle_item(session, tx, user_uuid, account_id, bank_uuid)
            results.append(outcome)

    return json.dumps({"status": "success", "processed_count": len(results), "items": results})

intake_tools = [process_incoming_transactions_tool]

class TransactionIntakeAgent:
    """
    Autonomous Agent managing the transaction entity lifecycle:
    - Pending authorization to settled transitions
    - Cash flow segregation (transfers vs inflows vs genuine spending)
    - Storage and A2A event generation for downstream categorization
    """
    def __init__(self):
        self.app = self._build_graph()

    def _build_graph(self):
        tool_node = ToolNode(intake_tools)
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(intake_tools)

        def evaluate_node(state: IntakeState):
            messages = list(state.get("messages", []))
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "You are the Transaction Intake Agent. Your job is to process raw bank transactions, "
                        "update pending authorizations when they settle, filter out internal transfers and income, "
                        "and ensure genuine consumer expenses are safely stored for categorization."
                    )
                )
                messages.insert(0, sys_msg)
            response = llm.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: IntakeState):
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        workflow = StateGraph(IntakeState)
        workflow.add_node("evaluator", evaluate_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("evaluator")
        workflow.add_conditional_edges("evaluator", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "evaluator")

        return workflow.compile()
