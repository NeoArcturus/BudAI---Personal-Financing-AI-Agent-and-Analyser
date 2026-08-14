from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta

from models.database_models import User
from middleware.auth_middleware import get_current_user
from services.Analyser_Agent.expense_analysis import ExpenseAnalysis
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

router = APIRouter(prefix="/api/widgets", tags=["Widgets"])

@router.get("/spending-trends")
async def get_spending_trends(
    account_id: str = Query(..., description="Bank Account ID"),
    time_type: str = Query("monthly", description="'daily', 'weekly', or 'monthly'"),
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Dedicated zero-latency UI endpoint for the Spending Trends line chart.
    """
    if not from_date:
        from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    ea = ExpenseAnalysis(identifier=account_id, user_uuid=current_user.user_uuid)
    if not ea.fetch_data(from_date, to_date):
        return {"status": "success", "data": []}

    if time_type == "daily":
        df = ea.get_daily_spend_data()
    elif time_type == "weekly":
        df = ea.get_weekly_spend_data()
    else:
        df = ea.get_monthly_spend_data()

    payload = []
    for _, row in df.iterrows():
        data_point = {
            "Date": row['Date'].isoformat() if hasattr(row['Date'], 'isoformat') else str(row['Date']),
            "Amount": round(float(row['Amount']), 2)
        }
        if 'category' in row:
            data_point["Category"] = row['category']
        elif 'Category' in row:
             data_point["Category"] = row['Category']
             
        if 'description' in row:
             data_point["description"] = row['description']
             
        payload.append(data_point)
        
    resolved_name = account_id
    from config import SessionLocal
    from models.database_models import Account, Bank
    with SessionLocal() as session:
        acc = session.query(Account).filter_by(account_id=account_id).first()
        if acc and acc.bank_uuid:
            bank = session.query(Bank).filter_by(bank_uuid=acc.bank_uuid).first()
            if bank and bank.bank_name:
                resolved_name = bank.bank_name

    return {"status": "success", "data": [{"bank_name": resolved_name, "data": payload}]}


@router.get("/expense-distribution")
async def get_expense_distribution(
    account_id: str = Query(..., description="Bank Account ID"),
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Dedicated zero-latency UI endpoint for the Expense Distribution pie/doughnut chart.
    """
    if not from_date:
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    ea = ExpenseAnalysis(identifier=account_id, user_uuid=current_user.user_uuid)
    if not ea.fetch_data(from_date, to_date):
         return {"status": "success", "data": []}
         
    df = ea.classified_data
    if df.empty or ('category' not in df.columns and 'Category' not in df.columns):
         return {"status": "success", "data": []}
         
    cat_col = 'category' if 'category' in df.columns else 'Category'
    
    # Group by category
    grouped = df.groupby(cat_col)['Amount'].sum().reset_index()
    payload = []
    for _, row in grouped.iterrows():
         payload.append({
             "category": row[cat_col],
             "amount": round(float(row['Amount']), 2)
         })
         
    resolved_name = account_id
    from config import SessionLocal
    from models.database_models import Account, Bank
    with SessionLocal() as session:
        acc = session.query(Account).filter_by(account_id=account_id).first()
        if acc and acc.bank_uuid:
            bank = session.query(Bank).filter_by(bank_uuid=acc.bank_uuid).first()
            if bank and bank.bank_name:
                resolved_name = bank.bank_name
                
    return {"status": "success", "data": [{"bank_name": resolved_name, "data": payload}]}


@router.get("/proactive-insights")
async def get_proactive_insights(current_user: User = Depends(get_current_user)):
    """
    Dedicated zero-latency UI endpoint for the Proactive Insights feed.
    """
    from config import SessionLocal
    from models.database_models import ProactiveInsight
    
    with SessionLocal() as session:
        insights = session.query(ProactiveInsight).filter(
            ProactiveInsight.user_uuid == current_user.user_uuid
        ).order_by(ProactiveInsight.created_at.desc()).limit(5).all()
        
        payload = []
        for insight in insights:
            payload.append({
                "id": insight.id,
                "text": insight.insight_text,
                "type": insight.insight_type,
                "urgency": insight.urgency_level,
                "created_at": insight.created_at.isoformat()
            })
            
    return {"status": "success", "data": payload}
