import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, text
from config import SessionLocal
from middleware.auth_middleware import get_current_user
from models.database_models import User, Account, Bucket, BucketType, Transaction
from services.analytics.fx_service import get_cross_fx_rate
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

router = APIRouter(tags=["Dashboard"])

@router.get("/api/dashboard/snapshot")
@router.get("/api/snapshot")
def get_dashboard_snapshot(
    currency: Optional[str] = Query(None, description="Preferred display currency e.g. GBP, USD, EUR"),
    limit_transactions: int = Query(default=30, ge=1, le=100),
    limit_notifications: int = Query(default=15, ge=1, le=50),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Consolidated Single-Roundtrip Snapshot API.
    Extracts parameters from existing Account, Bucket, Transaction, and SystemAlert models
    to hydrate the entire unified dashboard in one single database transaction without extra classes.
    """
    with SessionLocal() as session:
        # 1. Accounts
        accounts_raw = session.query(Account).filter_by(user_uuid=current_user.user_uuid).all()
        account_list: List[Dict[str, Any]] = []
        is_syncing = False

        for acc in accounts_raw:
            raw_bal = float(acc.account_balance or 0.0)
            status_val = getattr(acc, "sync_status", "READY") or "READY"
            if status_val in ["SYNCING", "INITIALIZING"]:
                is_syncing = True

            account_list.append({
                "account_id": acc.account_id,
                "display_name": acc.display_name or "Bank Account",
                "account_number": acc.account_number,
                "sort_code": acc.sort_code,
                "account_balance": max(0.0, raw_bal),
                "currency": (acc.currency or "GBP").strip().upper(),
                "account_type": acc.account_type or "TRANSACTION",
                "sync_status": status_val,
                "last_synced_at": acc.last_synced_at.isoformat() if acc.last_synced_at else None
            })

        # Determine base currency
        if currency and currency.strip().isalpha():
            base_curr = currency.strip().upper()
        elif account_list:
            base_curr = account_list[0]["currency"]
        else:
            base_curr = "GBP"

        # 2. Virtual Buckets with Idempotent DEFAULT check
        default_bucket = session.query(Bucket).filter(
            Bucket.user_id == current_user.user_uuid,
            Bucket.type == BucketType.DEFAULT,
            Bucket.is_active == True
        ).first()

        if not default_bucket:
            default_bucket = Bucket(
                id=str(uuid.uuid4()),
                user_id=current_user.user_uuid,
                type=BucketType.DEFAULT,
                name="Unallocated Funds",
                priority_index=0.0,
                cached_balance=0.0,
                is_active=True
            )
            session.add(default_bucket)
            session.commit()

        buckets_raw = session.query(Bucket).filter(
            Bucket.user_id == current_user.user_uuid,
            Bucket.is_active == True
        ).order_by(
            case((Bucket.type == BucketType.DEFAULT, 0), else_=1),
            Bucket.priority_index
        ).all()

        bucket_list: List[Dict[str, Any]] = []
        for b in buckets_raw:
            b_bal = max(0.0, float(b.cached_balance or 0.0))
            bucket_list.append({
                "id": b.id,
                "name": b.name,
                "type": b.type.value if hasattr(b.type, "value") else str(b.type),
                "cached_balance": b_bal,
                "priority_index": float(b.priority_index or 0.0),
                "target_amount": float(b.target_amount) if b.target_amount is not None else None,
                "target_date": b.target_date.isoformat() if b.target_date else None,
                "is_active": b.is_active
            })

        # 3. FX Multi-Currency Portfolio Reconciliation
        by_currency = {}
        total_physical_reconciled = 0.0

        for acc in account_list:
            c_code = acc["currency"]
            bal = acc["account_balance"]
            if c_code not in by_currency:
                by_currency[c_code] = {"nominal": 0.0, "converted": 0.0, "rate": 1.0}
            by_currency[c_code]["nominal"] = round(by_currency[c_code]["nominal"] + bal, 2)

        for c_code, info in by_currency.items():
            if c_code == base_curr:
                rate = 1.0
            else:
                try:
                    fx_res = get_cross_fx_rate(from_currency=c_code, to_currency=base_curr, amount=1.0)
                    rate = float(fx_res.get("rate", 1.0))
                except Exception:
                    rate = 1.0
            info["rate"] = round(rate, 6)
            converted_val = round(info["nominal"] * rate, 2)
            info["converted"] = converted_val
            total_physical_reconciled += converted_val

        total_physical_reconciled = round(total_physical_reconciled, 2)
        total_virtual = round(sum(b["cached_balance"] for b in bucket_list), 2)
        discrepancy = round(total_physical_reconciled - total_virtual, 2)
        is_balanced = abs(discrepancy) < 0.01

        # 4. Recent Transactions from existing Transaction model
        txs_raw = (
            session.query(Transaction)
            .filter_by(user_uuid=current_user.user_uuid)
            .order_by(Transaction.date.desc())
            .limit(limit_transactions)
            .all()
        )

        tx_list: List[Dict[str, Any]] = []
        for tx in txs_raw:
            tx_list.append({
                "transaction_uuid": tx.transaction_uuid,
                "account_id": tx.account_id,
                "date": tx.date.isoformat() if tx.date else None,
                "amount": float(tx.amount or 0.0),
                "currency": tx.currency or "GBP",
                "description": tx.description or "",
                "category": tx.sub_category or getattr(tx, "category", "Uncategorized") or "Uncategorized",
                "is_pending": bool(tx.is_pending)
            })

        # 5. System Notifications from SystemAlert
        alerts_sql = text("""
            SELECT id, title, message, event_type, 
                   COALESCE(urgency_level, 1) as urgency_level,
                   COALESCE(status_code, '308-200') as status_code,
                   COALESCE(is_read, FALSE) as is_read,
                   COALESCE(is_dismissed, FALSE) as is_dismissed,
                   timestamp
            FROM system_alerts
            WHERE user_id = :user_id AND (is_dismissed IS FALSE OR is_dismissed IS NULL)
            ORDER BY timestamp DESC
            LIMIT :limit
        """)
        alerts_res = session.execute(alerts_sql, {"user_id": current_user.user_uuid, "limit": limit_notifications}).fetchall()

        alert_list: List[Dict[str, Any]] = []
        for r in alerts_res:
            alert_list.append({
                "id": str(r.id),
                "title": r.title,
                "message": r.message,
                "event_type": r.event_type,
                "urgency_level": int(r.urgency_level),
                "status_code": r.status_code,
                "is_read": bool(r.is_read),
                "is_dismissed": bool(r.is_dismissed),
                "timestamp": r.timestamp.isoformat() if r.timestamp else None
            })

        log_pipeline_data(
            logger=logger,
            pipeline="snapshot_engine",
            stage="get_dashboard_snapshot",
            data={
                "user_uuid": current_user.user_uuid,
                "accounts_count": len(account_list),
                "buckets_count": len(bucket_list),
                "transactions_count": len(tx_list),
                "notifications_count": len(alert_list),
                "total_wealth": total_physical_reconciled,
                "is_syncing": is_syncing
            },
            message=f"Consolidated snapshot extracted in 1 roundtrip for user {current_user.user_uuid}"
        )

        return {
            "user_uuid": current_user.user_uuid,
            "accounts": account_list,
            "buckets": bucket_list,
            "recent_transactions": tx_list,
            "notifications": alert_list,
            "total_wealth": total_physical_reconciled,
            "active_currency": base_curr,
            "is_syncing": is_syncing,
            "portfolio_reconciliation": {
                "base_currency": base_curr,
                "total_physical_reconciled": total_physical_reconciled,
                "total_virtual": total_virtual,
                "discrepancy": discrepancy,
                "is_balanced": is_balanced,
                "by_currency": by_currency
            }
        }
