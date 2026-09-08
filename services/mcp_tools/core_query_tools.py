import json
from typing import List, Optional, Any, Annotated
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from sqlalchemy import text, select, func, and_
from config import SessionLocal
from models.database_models import Transaction, Account, Bank, Subscription
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

# ----------------- Schemas -----------------

class QueryTransactionsInput(BaseModel):
    account_id: Optional[str] = Field(default=None, description="The specific account ID or Bank Name to filter by.")
    start_date: Optional[str] = Field(default=None, description="Start date in YYYY-MM-DD format.")
    end_date: Optional[str] = Field(default=None, description="End date in YYYY-MM-DD format.")
    categories: Optional[List[str]] = Field(default=None, description="List of categories to filter by. EXACT MATCH ONLY. Valid strings: 'Entertainment & Lifestyle', 'Fees & Charges', 'Food & Dining', 'Housing', 'Income', 'Shopping & Retail', 'Subscriptions & Digital Services', 'Taxes & Government Payments', 'Transfers & Payments', 'Transportation', 'Utilities'.")
    min_amount: Optional[float] = Field(default=None, description="Minimum absolute transaction amount.")
    max_amount: Optional[float] = Field(default=None, description="Maximum absolute transaction amount.")
    transaction_type: Optional[str] = Field(default=None, description="'income' or 'expense'.")

class AggregateFinancialDataInput(BaseModel):
    account_id: Optional[str] = Field(default=None, description="The specific account ID or Bank Name to filter by.")
    start_date: Optional[str] = Field(default=None, description="Start date in YYYY-MM-DD format.")
    end_date: Optional[str] = Field(default=None, description="End date in YYYY-MM-DD format.")
    group_by: str = Field(..., description="'day', 'month', or 'category'.")
    metric: str = Field(default="sum", description="'sum', 'average', or 'count'.")
    transaction_type: Optional[str] = Field(default=None, description="'income' or 'expense'.")
    categories: Optional[List[str]] = Field(default=None, description="List of categories to filter by. EXACT MATCH ONLY. Valid strings: 'Entertainment & Lifestyle', 'Fees & Charges', 'Food & Dining', 'Housing', 'Income', 'Shopping & Retail', 'Subscriptions & Digital Services', 'Taxes & Government Payments', 'Transfers & Payments', 'Transportation', 'Utilities'.")

# ----------------- Helper -----------------

def _apply_transaction_filters(query: Any, user_uuid: str, account_id: Optional[str], 
                               start_date: Optional[str], end_date: Optional[str], 
                               categories: Optional[List[str]], min_amount: Optional[float], 
                               max_amount: Optional[float], transaction_type: Optional[str], session: Any):
    # Base filter
    query = query.where(Transaction.user_uuid == user_uuid)

    # Account resolution
    if account_id:
        # Check if it's a Bank Name or an actual Account ID
        account = session.execute(
            select(Account.account_id)
            .join(Bank, Account.bank_uuid == Bank.bank_uuid)
            .where(and_(Account.user_uuid == user_uuid, 
                        (Account.account_id == account_id) | (Bank.bank_name.ilike(f"%{account_id}%"))))
        ).scalars().first()
        
        if account:
            query = query.where(Transaction.account_id == account)
        else:
            # Fallback to direct string match if DB lookup fails
            query = query.where(Transaction.account_id == account_id)

    # Date filters
    if start_date:
        if "T" in start_date: start_date = start_date.split("T")[0]
        query = query.where(Transaction.date >= datetime.strptime(start_date, "%Y-%m-%d"))
    if end_date:
        if "T" in end_date: end_date = end_date.split("T")[0]
        query = query.where(Transaction.date <= datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1))

    # Categories
    if categories:
        from models.database_models import MerchantKnowledge
        query = query.join(MerchantKnowledge, Transaction.merchant_knowledge_uuid == MerchantKnowledge.knowledge_uuid, isouter=True)
        query = query.where(MerchantKnowledge.category.in_(categories))

    # Amount & Type filters
    if transaction_type == "income":
        query = query.where(Transaction.amount > 0)
        if min_amount is not None:
            query = query.where(Transaction.amount >= min_amount)
        if max_amount is not None:
            query = query.where(Transaction.amount <= max_amount)
    elif transaction_type == "expense":
        query = query.where(Transaction.amount < 0)
        if min_amount is not None:
            query = query.where(func.abs(Transaction.amount) >= min_amount)
        if max_amount is not None:
            query = query.where(func.abs(Transaction.amount) <= max_amount)
    else:
        # Absolute checks regardless of sign
        if min_amount is not None:
            query = query.where(func.abs(Transaction.amount) >= min_amount)
        if max_amount is not None:
            query = query.where(func.abs(Transaction.amount) <= max_amount)
            
    return query

# ----------------- Tools -----------------

@tool(args_schema=QueryTransactionsInput)
def query_transactions(account_id: Optional[str] = None, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, categories: Optional[List[str]] = None, 
                       min_amount: Optional[float] = None, max_amount: Optional[float] = None, 
                       transaction_type: Optional[str] = None, user_uuid: Annotated[str, InjectedState("user_uuid")] = "") -> str:
    """
    Dynamically filter and read transaction records from the database. 
    Returns a JSON string containing the transactions.
    """
    logger.info(json.dumps({"message": "Executing MCP Tool: query_transactions", "status_code": 200}))
    try:
        with SessionLocal() as session:
            query = select(Transaction)
            query = _apply_transaction_filters(query, user_uuid, account_id, start_date, end_date, categories, min_amount, max_amount, transaction_type, session)
            
            # Order by most recent
            query = query.order_by(Transaction.date.desc()).limit(500) # Safeguard limit
            
            transactions = session.execute(query).scalars().all()
            
            if not transactions:
                return json.dumps({"status": "success", "data": []})
                
            results = []
            for t in transactions:
                results.append({
                    "id": t.transaction_uuid,
                    "date": t.date.strftime("%Y-%m-%d") if t.date else None,
                    "amount": t.amount,
                    "currency": t.currency,
                    "category": t.category,
                    "description": t.description
                })
            
            return json.dumps({"status": "success", "count": len(results), "data": results})
            
    except Exception as e:
        logger.error(json.dumps({"message": f"Query Error: {e}", "status_code": 500}))
        return json.dumps({"status": "error", "message": str(e)})

@tool(args_schema=AggregateFinancialDataInput)
def aggregate_financial_data(group_by: str, metric: str = "sum", account_id: Optional[str] = None, 
                             start_date: Optional[str] = None, end_date: Optional[str] = None, 
                             transaction_type: Optional[str] = None, categories: Optional[List[str]] = None, user_uuid: Annotated[str, InjectedState("user_uuid")] = "") -> str:
    """
    Aggregate transaction data (e.g. sum by category, average by month) using the database directly.
    Returns JSON grouping.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: aggregate_financial_data by {group_by}", "status_code": 200}))
    try:
        with SessionLocal() as session:
            # Determine grouping column
            from models.database_models import MerchantKnowledge
            if group_by == "day":
                group_col = func.time_bucket(text("'1 day'"), Transaction.date)
            elif group_by == "month":
                group_col = func.time_bucket(text("'1 month'"), Transaction.date)
            elif group_by == "category":
                group_col = MerchantKnowledge.category
            else:
                return json.dumps({"status": "error", "message": "Invalid group_by parameter. Use 'day', 'month', or 'category'."})

            # Determine metric
            if metric == "sum":
                metric_col = func.sum(Transaction.amount).label('total')
            elif metric == "average":
                metric_col = func.avg(Transaction.amount).label('total')
            elif metric == "count":
                metric_col = func.count(Transaction.transaction_uuid).label('total')
            else:
                return json.dumps({"status": "error", "message": "Invalid metric parameter."})

            query = select(group_col.label('group_key'), metric_col).select_from(Transaction)
            if group_by == "category":
                query = query.join(MerchantKnowledge, Transaction.merchant_knowledge_uuid == MerchantKnowledge.knowledge_uuid, isouter=True)
            query = _apply_transaction_filters(query, user_uuid, account_id, start_date, end_date, categories, None, None, transaction_type, session)
            
            query = query.group_by(group_col)
            
            results = session.execute(query).all()
            
            data = []
            for row in results:
                key = row.group_key
                # Format dates to string
                if isinstance(key, datetime):
                    key = key.strftime("%Y-%m-%d")
                
                # Format amounts to round to 2 decimals
                val = round(float(row.total), 2) if row.total is not None else 0.0
                
                # For expenses, convert to positive absolute values for easier chart rendering
                if transaction_type == 'expense' and metric in ["sum", "average"]:
                    val = abs(val)
                    
                data.append({
                    "group": str(key) if key else "Uncategorized",
                    "value": val
                })
                
            return json.dumps({"status": "success", "data": data})

    except Exception as e:
        logger.error(json.dumps({"message": f"Aggregation Error: {e}", "status_code": 500}))
        return json.dumps({"status": "error", "message": str(e)})

class GetLiabilityHorizonInput(BaseModel):
    pass

@tool(args_schema=GetLiabilityHorizonInput)
def get_liability_horizon(user_uuid: Annotated[str, InjectedState("user_uuid")] = "") -> str:
    """
    Fetches the user's Liability Horizon (Upcoming Direct Debits, Standing Orders, and Pending Transactions).
    Provides the Orchestrator LLM with perfect foresight to execute the Priority Waterfall sweeps.
    """
    logger.info(json.dumps({"message": "Executing MCP Tool: get_liability_horizon", "status_code": 200}))
    try:
        with SessionLocal() as session:
            # 1. Fetch Subscriptions (Direct Debits & Standing Orders)
            subs = session.execute(
                select(Subscription).where(Subscription.user_uuid == user_uuid, Subscription.status != "CANCELLED")
            ).scalars().all()
            
            # 2. Fetch Pending Transactions
            pending_txs = session.execute(
                select(Transaction).where(Transaction.user_uuid == user_uuid, Transaction.is_pending == True)
            ).scalars().all()
            
            subs_data = []
            for s in subs:
                subs_data.append({
                    "merchant": s.merchant_name,
                    "expected_amount": s.expected_amount,
                    "next_date": s.next_expected_date.strftime("%Y-%m-%d") if s.next_expected_date else None,
                    "frequency": s.predicted_frequency
                })
                
            pending_data = []
            for tx in pending_txs:
                pending_data.append({
                    "id": tx.transaction_uuid,
                    "date": tx.date.strftime("%Y-%m-%d") if tx.date else None,
                    "amount": tx.amount,
                    "merchant": tx.description,
                    "category": tx.category
                })
                
            return json.dumps({
                "status": "success", 
                "subscriptions": subs_data,
                "pending_transactions": pending_data
            })
            
    except Exception as e:
        logger.error(json.dumps({"message": f"Liability Horizon Error: {e}", "status_code": 500}))
        return json.dumps({"status": "error", "message": str(e)})
