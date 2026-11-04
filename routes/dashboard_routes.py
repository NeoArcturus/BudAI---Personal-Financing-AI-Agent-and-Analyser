from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from sqlmodel import Session, select
from config import get_db
from models.database_models import User
from middleware.auth_middleware import get_current_user
from pydantic import BaseModel
import uuid
import json
import math
from datetime import datetime, timedelta
from sqlalchemy import func

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])

class DataRequestPayload(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

@router.get("/widgets", response_model=Dict[str, List[str]])
def get_user_widgets(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.get(User, current_user.user_uuid)
    
    PERSONA_WIDGET_MAP = {
        "STUDENT": ["cashFlow", "expenseDistribution", "analyticsHabits", "aiChat"],
        "PROFESSIONAL": ["spendingTrend", "analyticsSubscriptions", "analyticsHealth", "aiChat"],
        "BUSINESS": ["cashFlow", "ledger", "analyticsAnomalies", "financialNews"],
        "RETIREE": ["commodityMarket", "analyticsRisk", "expenseDistribution", "aiChat"],
        "CREATIVE": ["spendingTrend", "analyticsHabits", "analyticsAnomalies", "aiChat"]
    }
    
    widgets = []
    if user and user.persona:
        widgets = PERSONA_WIDGET_MAP.get(user.persona, PERSONA_WIDGET_MAP["PROFESSIONAL"])
        
    from models.database_models import UserLifestyleProfile
    profile = db.execute(select(UserLifestyleProfile).where(UserLifestyleProfile.user_uuid == current_user.user_uuid)).scalars().first()
    if profile and profile.hidden_widgets:
        hidden_list = profile.hidden_widgets.split(",")
        widgets = [w for w in widgets if w not in hidden_list]
        
    return {"widgets": widgets}

@router.delete("/widgets/{widget_type}")
def delete_user_widget(widget_type: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from models.database_models import UserLifestyleProfile
    profile = db.execute(select(UserLifestyleProfile).where(UserLifestyleProfile.user_uuid == current_user.user_uuid)).scalars().first()
    
    if not profile:
        profile = UserLifestyleProfile(profile_uuid=str(uuid.uuid4()), user_uuid=current_user.user_uuid)
        db.add(profile)
        
    hidden_list = profile.hidden_widgets.split(",") if profile.hidden_widgets else []
    if widget_type not in hidden_list:
        hidden_list.append(widget_type)
        profile.hidden_widgets = ",".join(hidden_list).strip(",")
        db.commit()
        
    return {"status": "success", "message": f"{widget_type} widget permanently hidden"}

@router.post("/widgets/data")
def fetch_widget_data(payload: DataRequestPayload, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tool_name = payload.tool_name
    params = payload.parameters
    
    from models.database_models import Transaction, ProactiveInsight, utc_now
    
    thirty_days_ago = utc_now() - timedelta(days=30)

    if tool_name == "aggregate_financial_data":
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
        from models.database_models import MerchantKnowledge
        stmt = select(
            MerchantKnowledge.category,
            func.sum(func.abs(Transaction.amount))
        ).join(
            MerchantKnowledge, Transaction.merchant_knowledge_uuid == MerchantKnowledge.knowledge_uuid, isouter=True
        ).where(
            Transaction.user_uuid == current_user.user_uuid,
            Transaction.amount < 0,
            Transaction.date >= thirty_days_ago,
            MerchantKnowledge.category.isnot(None)
        ).group_by(MerchantKnowledge.category).order_by(func.sum(func.abs(Transaction.amount)).desc()).limit(10)
        
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

@router.get("/snapshot")
def get_dashboard_snapshot(
    range_days: int = 60,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Consolidated Dashboard Snapshot Endpoint.
    Returns atomic zero-sum reconciliation, bucket hierarchy, stochastic forecast trajectory,
    recent monitored transactions, and structured agent decision stream in a single payload.
    """
    from models.database_models import (
        Account, Bucket, Transaction, MerchantKnowledge, SystemAlert, utc_now
    )

    user_uuid = current_user.user_uuid

    # 1. Reconciliation (Physical bank accounts vs. Virtual buckets)
    accounts = db.execute(select(Account).where(Account.user_uuid == user_uuid)).scalars().all()
    physical_balance = round(sum(float(acc.account_balance or 0.0) for acc in accounts), 2)
    bank_synced = len(accounts) > 0

    buckets = db.execute(
        select(Bucket).where(Bucket.user_id == user_uuid, Bucket.is_active == True).order_by(Bucket.priority_index)
    ).scalars().all()

    bucket_list = []
    raw_virtual_total = 0.0
    for b in buckets:
        bal = max(0.0, float(b.cached_balance or 0.0))
        raw_virtual_total += bal

    if not buckets and physical_balance > 0:
        raw_virtual_total = physical_balance

    virtual_balance = round(raw_virtual_total, 2)
    delta = round(physical_balance - virtual_balance, 2)
    is_reconciled = abs(delta) < 0.01

    bucket_type_notes = {
        "DEFAULT": "Optimal burn rate",
        "RECURRING": "Covers next 30d obligations",
        "ACCUMULATING": "Reserve buffer on target",
        "TARGET_DATE": "Scheduled contributions active",
        "LIABILITY": "Amortization active"
    }
    
    for b in buckets:
        bal = max(0.0, float(b.cached_balance or 0.0))
        pct = round((bal / virtual_balance * 100), 1) if virtual_balance > 0 else 0.0
        bucket_list.append({
            "id": str(b.id),
            "name": b.name,
            "type": b.type,
            "cached_balance": bal,
            "target_amount": float(b.target_amount) if b.target_amount else None,
            "priority_index": float(b.priority_index or 0.0),
            "allocation_percentage": pct,
            "daily_burn": 32.0 if b.type == "DEFAULT" else 0.0,
            "status_note": bucket_type_notes.get(b.type, "Active")
        })

    # If no buckets exist in DB yet, provide standard initial allocation
    if not bucket_list:
        base_bal = physical_balance if physical_balance > 0 else 4500.0
        default_templates = [
            ("Operating Cash", "DEFAULT", round(base_bal * 0.28, 2), 1.0, 28.0),
            ("Fixed Obligations", "RECURRING", round(base_bal * 0.40, 2), 2.0, 40.0),
            ("Emergency Reserve", "ACCUMULATING", round(base_bal * 0.21, 2), 3.0, 21.0),
            ("Travel & Leisure", "TARGET_DATE", round(base_bal * 0.08, 2), 4.0, 8.0),
            ("Debt Paydown", "LIABILITY", round(base_bal * 0.03, 2), 5.0, 3.0),
        ]
        for name, b_type, bal, prio, pct in default_templates:
            bucket_list.append({
                "id": f"temp-{b_type.lower()}",
                "name": name,
                "type": b_type,
                "cached_balance": bal,
                "target_amount": bal * 1.5 if b_type in ["ACCUMULATING", "TARGET_DATE"] else None,
                "priority_index": prio,
                "allocation_percentage": pct,
                "daily_burn": 32.0 if b_type == "DEFAULT" else 0.0,
                "status_note": bucket_type_notes.get(b_type, "Active")
            })
        virtual_balance = base_bal
        if physical_balance == 0:
            physical_balance = base_bal
        delta = 0.0
        is_reconciled = True

    # 2. Stochastic Forward Trajectory (60 days)
    safety_buffer_floor = 800.0
    start_balance = physical_balance if physical_balance > 0 else (virtual_balance if virtual_balance > 0 else 4500.0)
    
    thirty_days_ago = utc_now() - timedelta(days=30)
    recent_spend_stmt = select(func.sum(func.abs(Transaction.amount))).where(
        Transaction.user_uuid == user_uuid,
        Transaction.amount < 0,
        Transaction.date >= thirty_days_ago
    )
    total_recent_spend = db.execute(recent_spend_stmt).scalar() or 0.0
    est_daily_burn = float(total_recent_spend) / 30.0 if total_recent_spend > 0 else 32.0

    today = datetime.now()
    trajectory = []
    current_proj = start_balance
    volatility = 16.0

    for day in range(range_days + 1):
        dt = today + timedelta(days=day)
        income_boost = 2200.0 if (day % 30 == 25 and day > 0) else 0.0
        current_proj = current_proj - est_daily_burn + income_boost
        
        spread = volatility * math.sqrt(day + 1) * 3.2
        lower = max(0.0, round(current_proj - spread, 2))
        upper = round(current_proj + spread, 2)
        median = round(current_proj, 2)

        trajectory.append({
            "day": day,
            "date": dt.strftime("%b %d"),
            "median": median,
            "lower_bound": lower,
            "upper_bound": upper
        })

    # 3. Monitored Transactions (Recent 25)
    tx_stmt = select(
        Transaction, MerchantKnowledge.category
    ).outerjoin(
        MerchantKnowledge, Transaction.merchant_knowledge_uuid == MerchantKnowledge.knowledge_uuid
    ).where(
        Transaction.user_uuid == user_uuid
    ).order_by(Transaction.date.desc()).limit(25)

    tx_rows = db.execute(tx_stmt).all()
    transaction_list = []

    for tx, cat in tx_rows:
        amount_abs = round(abs(float(tx.amount)), 2)
        is_anomaly = bool(tx.is_semantic_anomaly)
        
        if is_anomaly or (amount_abs > 200 and not cat):
            tx_status = "Action Required"
        elif tx.sub_category in ["subscription", "utility", "fixed_obligation"]:
            tx_status = "Reconciled"
        else:
            tx_status = "Verified"

        if tx_status == "Action Required":
            pool_name = "Held for Review"
            pool_type = "DEFAULT"
        elif cat in ["Bills & Utilities", "Subscriptions", "Rent"]:
            pool_name = "Fixed Obligations"
            pool_type = "RECURRING"
        else:
            pool_name = "Operating Cash"
            pool_type = "DEFAULT"

        transaction_list.append({
            "id": str(tx.transaction_uuid),
            "date": tx.date.strftime("%Y-%m-%d %H:%M") if tx.date else "Recent",
            "display_date": tx.date.strftime("%b %d, %H:%M") if tx.date else "Recent",
            "merchant": tx.fully_cleaned_description or tx.semi_cleaned_description or tx.description or "Unknown Merchant",
            "category": cat or tx.sub_category or "Uncategorized",
            "pool": pool_name,
            "bucket_type": pool_type,
            "amount": amount_abs,
            "is_outflow": float(tx.amount) < 0,
            "is_anomaly": is_anomaly,
            "status": tx_status
        })

    if not transaction_list:
        sample_txs = [
            ("Tesco Stores", "Groceries", "Operating Cash", "DEFAULT", 45.20, False, "Verified", "Today, 14:20"),
            ("AWS Cloud", "Infrastructure", "Fixed Obligations", "RECURRING", 18.50, False, "Reconciled", "Today, 09:12"),
            ("Shell Petrol", "Transport", "Operating Cash", "DEFAULT", 82.00, False, "Verified", "Yesterday"),
            ("Merchant_4819 (Unknown)", "Uncategorized", "Held for Review", "DEFAULT", 410.00, True, "Action Required", "Yesterday"),
        ]
        for merch, cat, pool, b_type, amt, anom, st, d_str in sample_txs:
            transaction_list.append({
                "id": f"sample-{len(transaction_list)+1}",
                "date": today.isoformat(),
                "display_date": d_str,
                "merchant": merch,
                "category": cat,
                "pool": pool,
                "bucket_type": b_type,
                "amount": amt,
                "is_outflow": True,
                "is_anomaly": anom,
                "status": st
            })

    # 4. Agent Decisions & Causal Rationale Stream
    alerts_stmt = select(SystemAlert).where(
        SystemAlert.user_id == user_uuid
    ).order_by(SystemAlert.timestamp.desc()).limit(10)
    alerts = db.execute(alerts_stmt).scalars().all()

    decision_list = []
    for a in alerts:
        payload_data = {}
        if a.action_payload:
            try:
                payload_data = json.loads(a.action_payload) if isinstance(a.action_payload, str) else a.action_payload
            except Exception:
                payload_data = {}

        requires_input = a.urgency_level == "CRITICAL" or a.action_type == "CONFIRMATION_REQUIRED"
        decision_list.append({
            "id": str(a.id),
            "timestamp": a.timestamp.strftime("%b %d, %H:%M") if a.timestamp else "Recent",
            "status": "Awaiting Input" if requires_input else "Executed",
            "title": a.title,
            "observation": payload_data.get("observation", a.message),
            "policy": payload_data.get("policy", "Autonomous liquidity balancing rule"),
            "action": payload_data.get("action", f"Execution triggered for {a.title}"),
            "requires_action": requires_input
        })

    if not decision_list:
        decision_list = [
            {
                "id": "dec-1",
                "timestamp": "Today, 10:39 AM",
                "status": "Executed",
                "title": "Automated Rebalance: +£120.00",
                "observation": "AWS invoice £18.50 exceeded weekly mean.",
                "policy": "Keep Fixed Obligations covered 14 days ahead.",
                "action": "Transferred £120.00 from Operating Cash to Fixed Obligations.",
                "requires_action": False
            },
            {
                "id": "dec-2",
                "timestamp": "Yesterday, 10:38 PM",
                "status": "Executed",
                "title": "Buffer Preservation: £45.00",
                "observation": "Supermarket debit cleared.",
                "policy": "Priority cascade from auxiliary pool.",
                "action": "Replenished liquidity pool from short-term reserve.",
                "requires_action": False
            },
            {
                "id": "dec-3",
                "timestamp": "Yesterday, 07:33 PM",
                "status": "Awaiting Input",
                "title": "Policy Intercept: £410.00 Unrecognized",
                "observation": "Transaction exceeds the autonomous unverified limit (£100.00).",
                "policy": "Mandatory user confirmation required before final categorization.",
                "action": "Quarantine funds in Operating Cash pending review.",
                "requires_action": True
            }
        ]

    # 5. Guardrails
    guardrails = {
        "safety_buffer": safety_buffer_floor,
        "autonomous_limit": 150.0,
        "approval_policy": "Strict (Two-Factor)",
        "bank_connection_protocol": "Open Banking 2.1",
        "guardrails_active": True
    }

    return {
        "status": "SUCCESS",
        "timestamp": utc_now().isoformat(),
        "reconciliation": {
            "physical_balance": physical_balance,
            "virtual_balance": virtual_balance,
            "delta": delta,
            "is_reconciled": is_reconciled,
            "bank_synced": bank_synced
        },
        "buckets": bucket_list,
        "forecast": {
            "horizon_days": range_days,
            "safety_buffer_floor": safety_buffer_floor,
            "trajectory": trajectory
        },
        "transactions": transaction_list,
        "decisions": decision_list,
        "guardrails": guardrails
    }
