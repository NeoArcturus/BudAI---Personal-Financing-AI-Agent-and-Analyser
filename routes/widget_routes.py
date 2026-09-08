from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta

from models.database_models import User
from middleware.auth_middleware import get_current_user
from services.logger_setup import get_core_logger
from config import SessionLocal
from sqlalchemy import text
from models.database_models import Account, Bank, ProactiveInsight

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
    if not from_date:
        from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    bucket_str = "1 day"
    if time_type == "weekly":
        bucket_str = "1 week"
    elif time_type == "monthly":
        bucket_str = "1 month"

    query = text(f"""
        SELECT time_bucket('{bucket_str}', date) as "Date", sum(abs(amount)) as "Amount"
        FROM transactions
        WHERE user_uuid = :user_uuid 
          AND account_id = :account_id
          AND date >= :start_date 
          AND date <= :end_date
          AND amount < 0
        GROUP BY time_bucket('{bucket_str}', date)
        ORDER BY "Date" ASC
    """)
    
    payload = []
    resolved_name = account_id
    
    with SessionLocal() as session:
        acc = session.query(Account).filter_by(account_id=account_id).first()
        if acc and acc.bank_uuid:
            bank = session.query(Bank).filter_by(bank_uuid=acc.bank_uuid).first()
            if bank and bank.bank_name:
                resolved_name = bank.bank_name
                
        results = session.execute(query, {
            "user_uuid": current_user.user_uuid,
            "account_id": account_id,
            "start_date": from_date,
            "end_date": to_date
        }).fetchall()
        
        for r in results:
            if r[0] is not None:
                payload.append({
                    "Date": r[0].isoformat() if hasattr(r[0], 'isoformat') else str(r[0]),
                    "Amount": round(float(r[1]), 2) if r[1] is not None else 0.0
                })

    return {"status": "success", "data": [{"bank_name": resolved_name, "data": payload}]}


@router.get("/expense-distribution")
async def get_expense_distribution(
    account_id: str = Query(..., description="Bank Account ID"),
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    if not from_date:
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    query = text("""
        SELECT m.category, sum(abs(t.amount)) as total
        FROM transactions t
        LEFT JOIN merchant_knowledge m ON t.merchant_knowledge_uuid = m.knowledge_uuid
        WHERE t.user_uuid = :user_uuid 
          AND t.account_id = :account_id
          AND t.date >= :start_date 
          AND t.date <= :end_date
          AND t.amount < 0
          AND m.category IS NOT NULL
        GROUP BY m.category
        ORDER BY total DESC
    """)

    payload = []
    resolved_name = account_id
    
    with SessionLocal() as session:
        acc = session.query(Account).filter_by(account_id=account_id).first()
        if acc and acc.bank_uuid:
            bank = session.query(Bank).filter_by(bank_uuid=acc.bank_uuid).first()
            if bank and bank.bank_name:
                resolved_name = bank.bank_name
                
        results = session.execute(query, {
            "user_uuid": current_user.user_uuid,
            "account_id": account_id,
            "start_date": from_date,
            "end_date": to_date
        }).fetchall()
        
        for r in results:
            payload.append({
                "category": r[0],
                "amount": round(float(r[1]), 2) if r[1] is not None else 0.0
            })

    return {"status": "success", "data": [{"bank_name": resolved_name, "data": payload}]}


@router.get("/proactive-insights")
async def get_proactive_insights(current_user: User = Depends(get_current_user)):
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
