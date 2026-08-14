from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import select
from typing import List, Dict, Any
from config import SessionLocal
from models.database_models import UserLifestyleProfile, LifestyleCluster, Subscription, Transaction
import pandas as pd

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/persona", response_model=Dict[str, Any])
def get_user_persona(user_uuid: str, db: SessionLocal = Depends(get_db)):
    """Fetch the user's Macro-Persona and statistical benchmarks."""
    profile = db.execute(
        select(UserLifestyleProfile).where(UserLifestyleProfile.user_uuid == user_uuid)
    ).scalars().first()
    
    if not profile:
        return {
            "macro_persona": "PENDING_ANALYSIS",
            "benchmarks": {
                "housing": {"user_spend": 0, "average": 1000, "status": "UNKNOWN"},
                "transport": {"user_spend": 0, "average": 100, "status": "UNKNOWN"}
            }
        }
        
    # Dynamically calculate the user's actual category spending
    txs = db.execute(
        select(Transaction)
        .where(Transaction.user_uuid == user_uuid)
        .where(Transaction.amount < 0)
    ).scalars().all()
    
    category_spend = {}
    for tx in txs:
        cat = (tx.category or "uncategorized").lower()
        category_spend[cat] = category_spend.get(cat, 0.0) + abs(tx.amount)
        
    # Phase 4 (Roadmap): We will eventually query the Office for National Statistics JSON.
    # For now, we apply basic deterministic thresholds based on the persona.
    benchmarks = {}
    
    if profile.macro_persona == "STUDENT":
        target_housing, target_transport = 800, 50
    elif profile.macro_persona == "PROFESSIONAL":
        target_housing, target_transport = 1500, 200
    else:
        target_housing, target_transport = 1000, 100

    user_housing = category_spend.get("housing", 0.0)
    user_transport = category_spend.get("transport", 0.0)

    benchmarks["housing"] = {
        "user_spend": user_housing,
        "average": target_housing,
        "status": "OVER" if user_housing > target_housing else "UNDER"
    }
    benchmarks["transport"] = {
        "user_spend": user_transport,
        "average": target_transport,
        "status": "OVER" if user_transport > target_transport else "UNDER"
    }
    
    return {
        "macro_persona": profile.macro_persona,
        "benchmarks": benchmarks
    }

@router.get("/clusters", response_model=Dict[str, List[Dict[str, Any]]])
def get_lifestyle_clusters(user_uuid: str, db: SessionLocal = Depends(get_db)):
    """Fetch all HDBSCAN micro-lifestyles for the user."""
    clusters = db.execute(
        select(LifestyleCluster).where(LifestyleCluster.user_uuid == user_uuid)
    ).scalars().all()
    
    output = []
    for c in clusters:
        output.append({
            "name": c.micro_archetype or f"Cluster {c.hdbscan_cluster_id}",
            "total_spend": c.total_spend or 0.0,
            "transaction_count": c.transaction_count or 0
        })
        
    return {"clusters": output}

@router.get("/subscriptions", response_model=Dict[str, List[Dict[str, Any]]])
def get_upcoming_subscriptions(user_uuid: str, db: SessionLocal = Depends(get_db)):
    """Fetch detected recurring subscriptions and price hike flags."""
    from models.database_models import Bank, Account
    
    subs = db.execute(
        select(Subscription)
        .where(Subscription.user_uuid == user_uuid)
        .order_by(Subscription.next_expected_date.asc())
    ).scalars().all()
    
    output = []
    for s in subs:
        bank_name = None
        account_number = None
        sort_code = None
        
        if s.bank_uuid:
            bank = db.execute(select(Bank).where(Bank.bank_uuid == s.bank_uuid)).scalars().first()
            if bank:
                bank_name = bank.bank_name
                account = db.execute(
                    select(Account)
                    .where(Account.bank_uuid == s.bank_uuid)
                    .where(Account.user_uuid == user_uuid)
                ).scalars().first()
                if account:
                    account_number = account.account_number
                    sort_code = account.sort_code
                    
        output.append({
            "merchant_name": s.merchant_name,
            "expected_amount": s.expected_amount,
            "next_expected_date": s.next_expected_date.isoformat() if s.next_expected_date else None,
            "is_price_hike": s.is_price_hike,
            "predicted_frequency": s.predicted_frequency,
            "last_payment_date": s.last_payment_date.isoformat() if s.last_payment_date else None,
            "last_payment_amount": s.last_payment_amount,
            "bank_name": bank_name,
            "account_number": account_number,
            "sort_code": sort_code,
            "status": s.status
        })
        
    return {"subscriptions": output}

@router.get("/anomalies", response_model=Dict[str, List[Dict[str, Any]]])
def get_semantic_anomalies(user_uuid: str, db: SessionLocal = Depends(get_db)):
    """Fetch transactions flagged as semantic anomalies."""
    anomalies = db.execute(
        select(Transaction)
        .where(Transaction.user_uuid == user_uuid)
        .where(Transaction.is_semantic_anomaly == True)
        .order_by(Transaction.date.desc())
        .limit(10)
    ).scalars().all()
    
    output = []
    for a in anomalies:
        output.append({
            "transaction_uuid": a.transaction_uuid,
            "merchant": a.description,
            "amount": a.amount,
            "date": a.date.isoformat() if a.date else None,
            "reason": "Amount deviation from vector norm"
        })
        
    return {"anomalies": output}

@router.get("/impulse-vulnerability", response_model=Dict[str, Any])
def get_impulse_vulnerability(user_uuid: str, db: SessionLocal = Depends(get_db)):
    """Calculate the most vulnerable day of the week for discretionary spending."""
    txs = db.execute(
        select(Transaction)
        .where(Transaction.user_uuid == user_uuid)
        .where(Transaction.amount < 0)
    ).scalars().all()
    
    if not txs:
        return {"most_vulnerable_day": "Unknown", "historical_density_spend": 0.0}
        
    df = pd.DataFrame([{"amount": abs(t.amount), "day": t.date.weekday() if t.date else 0} for t in txs])
    if df.empty:
         return {"most_vulnerable_day": "Unknown", "historical_density_spend": 0.0}
         
    day_spend = df.groupby("day")["amount"].sum()
    worst_day_idx = day_spend.idxmax()
    worst_amount = float(day_spend.max())
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    return {
        "most_vulnerable_day": days[worst_day_idx],
        "historical_density_spend": worst_amount
    }
