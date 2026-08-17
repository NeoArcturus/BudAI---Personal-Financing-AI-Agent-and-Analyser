from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from sqlmodel import Session, select
from config import get_db
from models.database_models import User
from middleware.auth_middleware import get_current_user
from pydantic import BaseModel
import uuid
import json
from datetime import datetime, timedelta
from sqlalchemy import func

router = APIRouter(prefix="/api/dashboard/widgets", tags=["Dashboard Widgets"])

class DataRequestPayload(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

@router.get("", response_model=Dict[str, List[str]])
def get_user_widgets(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.get(User, current_user.user_uuid)
    
    PERSONA_WIDGET_MAP = {
        "STUDENT": ["cashFlow", "expenseDistribution", "analyticsHabits", "aiChat"],
        "PROFESSIONAL": ["spendingTrend", "analyticsSubscriptions", "analyticsHealth", "aiChat"],
        "BUSINESS": ["cashFlow", "ledger", "analyticsAnomalies", "financialNews"],
        "RETIREE": ["commodityMarket", "analyticsRisk", "expenseDistribution", "aiChat"],
        "CREATIVE": ["spendingTrend", "analyticsHabits", "analyticsAnomalies", "aiChat"]
    }
    
    # Fallback to empty list or default if user has no persona yet
    widgets = []
    if user and user.persona:
        widgets = PERSONA_WIDGET_MAP.get(user.persona, PERSONA_WIDGET_MAP["PROFESSIONAL"])
        
    return {"widgets": widgets}

@router.post("/data")
def fetch_widget_data(payload: DataRequestPayload, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tool_name = payload.tool_name
    params = payload.parameters
    
    # Import inside function to avoid circular/init issues
    from models.database_models import Transaction, ProactiveInsight
    
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    if tool_name == "aggregate_financial_data":
        # Get daily spending for last 30 days
        stmt = select(
            func.date(Transaction.date).label("dt"),
            func.sum(func.abs(Transaction.amount))
        ).where(
            Transaction.user_uuid == current_user.user_uuid,
            Transaction.amount < 0,
            Transaction.date >= thirty_days_ago
        ).group_by(func.date(Transaction.date)).order_by(func.date(Transaction.date))
        
        rows = db.execute(stmt).all()
        labels = [str(r[0]) for r in rows]
        data = [float(r[1]) for r in rows]
        
        return {
            "chart_data": {
                "labels": labels,
                "datasets": [{"label": "Daily Spend", "data": data}]
            },
            "summary": f"Total spend over {len(rows)} active days."
        }
        
    elif tool_name == "expense_distribution":
        # Get category breakdown for last 30 days
        stmt = select(
            Transaction.category,
            func.sum(func.abs(Transaction.amount))
        ).where(
            Transaction.user_uuid == current_user.user_uuid,
            Transaction.amount < 0,
            Transaction.date >= thirty_days_ago,
            Transaction.category.isnot(None)
        ).group_by(Transaction.category).order_by(func.sum(func.abs(Transaction.amount)).desc()).limit(10)
        
        rows = db.execute(stmt).all()
        labels = [str(r[0]) for r in rows]
        data = [float(r[1]) for r in rows]
        
        return {
            "chart_data": {
                "labels": labels,
                "datasets": [{"label": "Category Spend", "data": data}]
            },
            "summary": "Top expense categories for the last 30 days."
        }
        
    elif tool_name == "proactive_insights":
        stmt = select(ProactiveInsight.insight_text, ProactiveInsight.insight_type).where(
            ProactiveInsight.user_uuid == current_user.user_uuid
        ).order_by(ProactiveInsight.created_at.desc()).limit(5)
        
        rows = db.execute(stmt).all()
        insights = [{"text": r.insight_text, "severity": r.insight_type} for r in rows]
        
        if not insights:
            insights = [{"text": "No current financial anomalies detected.", "severity": "low"}]
            
        return {
            "insights": insights
        }
        
    return {"error": f"Tool {tool_name} not natively mapped yet in dashboard data endpoint."}
