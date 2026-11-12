import json
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy import case
from config import SessionLocal, redis_client
from models.database_models import Bucket, BucketType, Account
from services.analytics.fx_service import get_cross_fx_rate
from services.bucket_service import (
    get_or_create_unallocated_bucket,
    sync_unallocated_from_accounts,
    create_new_bucket,
    execute_bucket_transfer,
    update_bucket_details,
    delete_virtual_bucket,
    toggle_bucket_active_state
)
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


def get_user_buckets(user_uuid: str, currency: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves all virtual buckets and reconciles multi-currency balances against physical accounts.
    """
    try:
        with SessionLocal() as session:
            # Sync Unallocated bucket to live accounts
            sync_unallocated_from_accounts(session, user_uuid)

            # Query buckets ordered with UNALLOCATED first
            buckets = session.query(Bucket).filter(
                Bucket.user_id == user_uuid,
                Bucket.is_active == True
            ).order_by(
                case((Bucket.type == BucketType.UNALLOCATED, 0), else_=1),
                Bucket.priority
            ).all()

            bucket_list = []
            for b in buckets:
                b_dict = {
                    "id": b.id,
                    "name": b.name,
                    "type": b.type.value if hasattr(b.type, "value") else str(b.type),
                    "cached_balance": max(0.0, float(b.cached_balance or 0.0)),
                    "cached_balances": b.cached_balances or {},
                    "currency": b.currency or "GBP",
                    "priority": b.priority,
                    "target_amount": float(b.target_amount) if b.target_amount else None,
                    "target_date": b.target_date.isoformat() if b.target_date else None,
                    "associated_account_id": b.associated_account_id,
                    "associated_account_name": b.associated_account_name,
                    "allocated_amount": float(b.allocated_amount or 0.0),
                    "status_text": b.status_text,
                    "has_shortfall_warning": bool(b.has_shortfall_warning),
                    "shortfall_amount": float(b.shortfall_amount or 0.0)
                }
                bucket_list.append(b_dict)

            accounts = session.query(Account).filter_by(user_uuid=user_uuid).all()
            if currency and currency.strip().isalpha():
                base_curr = currency.strip().upper()
            elif accounts and accounts[0].currency:
                base_curr = accounts[0].currency.strip().upper()
            else:
                base_curr = "GBP"

            by_currency: Dict[str, Dict[str, float]] = {}
            total_physical_reconciled = 0.0

            for acc in accounts:
                c_code = (acc.currency or "GBP").strip().upper()
                bal = float(acc.account_balance or 0.0)
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
            total_virtual = round(sum(float(b.cached_balance or 0.0) for b in buckets), 2)
            discrepancy = round(total_physical_reconciled - total_virtual, 2)
            is_balanced = abs(discrepancy) < 0.01

            unallocated_bucket = next((b for b in buckets if (b.type.value if hasattr(b.type, "value") else str(b.type)) == BucketType.UNALLOCATED.value), None)

            return {
                "status": "success",
                "base_currency": base_curr,
                "unallocated_bucket_id": unallocated_bucket.id if unallocated_bucket else None,
                "unallocated_balance": round(float(unallocated_bucket.cached_balance or 0.0), 2) if unallocated_bucket else 0.0,
                "reconciliation": {
                    "total_physical_reconciled": total_physical_reconciled,
                    "total_virtual_reconciled": total_virtual,
                    "discrepancy": discrepancy,
                    "is_balanced": is_balanced,
                    "by_currency": by_currency
                },
                "buckets": bucket_list
            }
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Error fetching buckets for user {user_uuid}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Failed to retrieve virtual buckets.")


def create_bucket(
    user_uuid: str,
    bucket_type: str,
    name: str,
    target_amount: Optional[float] = None,
    target_date: Optional[datetime] = None,
    priority: int = 10,
    currency: str = "GBP"
) -> Dict[str, Any]:
    """Creates a new virtual bucket strictly of type BILLS or GOALS."""
    try:
        with SessionLocal() as session:
            new_bucket = create_new_bucket(
                session=session,
                user_uuid=user_uuid,
                bucket_type=bucket_type,
                name=name,
                target_amount=target_amount,
                target_date=target_date,
                priority=priority,
                currency=currency
            )
            session.commit()

            try:
                redis_client.publish(f"budai_events_{user_uuid}", json.dumps({
                    "event_type": "BUCKET_CREATED",
                    "bucket_id": new_bucket.id
                }))
            except Exception:
                pass

            return {"status": "success", "bucket_id": new_bucket.id}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Error creating bucket for user {user_uuid}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Internal server error")


def execute_transfer(
    user_uuid: str,
    source_bucket_id: str,
    target_bucket_id: str,
    amount: float
) -> Dict[str, Any]:
    """Executes a virtual transfer between buckets."""
    try:
        with SessionLocal() as session:
            transfer = execute_bucket_transfer(
                session=session,
                user_uuid=user_uuid,
                source_bucket_id=source_bucket_id,
                target_bucket_id=target_bucket_id,
                amount=amount
            )
            session.commit()

            try:
                redis_client.publish(f"budai_events_{user_uuid}", json.dumps({
                    "event_type": "VIRTUAL_TRANSFER",
                    "amount": amount,
                    "transfer_id": transfer.id
                }))
            except Exception:
                pass

            return {"status": "success", "message": "Transfer executed", "transfer_id": transfer.id}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Error executing transfer for user {user_uuid}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Internal server error")


def update_bucket(
    user_uuid: str,
    bucket_id: str,
    name: Optional[str] = None,
    target_amount: Optional[float] = None,
    target_date: Optional[datetime] = None,
    priority: Optional[int] = None
) -> Dict[str, Any]:
    """Updates bucket configuration."""
    try:
        with SessionLocal() as session:
            update_bucket_details(
                session=session,
                user_uuid=user_uuid,
                bucket_id=bucket_id,
                name=name,
                target_amount=target_amount,
                target_date=target_date,
                priority=priority
            )
            session.commit()

            try:
                redis_client.publish(f"budai_events_{user_uuid}", json.dumps({
                    "event_type": "BUCKET_UPDATED",
                    "bucket_id": bucket_id
                }))
            except Exception:
                pass

            return {"status": "success", "message": "Bucket updated"}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Error updating bucket {bucket_id}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Internal server error")


def delete_bucket(user_uuid: str, bucket_id: str) -> Dict[str, Any]:
    """Deletes/deactivates a virtual bucket and returns funds to Unallocated."""
    try:
        with SessionLocal() as session:
            delete_virtual_bucket(session=session, user_uuid=user_uuid, bucket_id=bucket_id)
            session.commit()

            try:
                redis_client.publish(f"budai_events_{user_uuid}", json.dumps({
                    "event_type": "BUCKET_DELETED",
                    "bucket_id": bucket_id
                }))
            except Exception:
                pass

            return {"status": "success", "message": "Bucket deleted successfully"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Error deleting bucket {bucket_id}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Internal server error")


def toggle_bucket_active(user_uuid: str, bucket_id: str, is_active: bool) -> Dict[str, Any]:
    """Toggles active status of a virtual bucket."""
    try:
        with SessionLocal() as session:
            toggle_bucket_active_state(session=session, user_uuid=user_uuid, bucket_id=bucket_id, is_active=is_active)
            session.commit()

            try:
                redis_client.publish(f"budai_events_{user_uuid}", json.dumps({
                    "event_type": "BUCKET_UPDATED",
                    "bucket_id": bucket_id
                }))
            except Exception:
                pass

            return {"status": "success", "message": f"Bucket active state set to {is_active}"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Error toggling bucket {bucket_id}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Internal server error")
