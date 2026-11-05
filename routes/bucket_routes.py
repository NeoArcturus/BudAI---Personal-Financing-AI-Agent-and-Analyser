from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from config import SessionLocal
from services.bucket_service import create_new_bucket
from middleware.auth_middleware import get_current_user
from models.database_models import User
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

router = APIRouter(prefix="/api/buckets", tags=["Buckets"])

class BucketCreateRequest(BaseModel):
    bucket_type: str
    name: str
    target_amount: Optional[float] = None
    target_date: Optional[datetime] = None
    priority_index: Optional[float] = 0.0

@router.post("/")
def create_bucket_endpoint(request: BucketCreateRequest, current_user: User = Depends(get_current_user)):
    try:
        with SessionLocal() as session:
            new_bucket = create_new_bucket(
                session=session,
                user_uuid=current_user.user_uuid,
                bucket_type=request.bucket_type,
                name=request.name,
                target_amount=request.target_amount,
                target_date=request.target_date,
                priority_index=request.priority_index
            )
            session.commit()
            return {"status": "success", "bucket_id": new_bucket.id}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/")
def get_user_buckets(
    currency: Optional[str] = Query(None, description="Target base currency for portfolio reconciliation, e.g. GBP, EUR, USD"),
    current_user: User = Depends(get_current_user)
):
    try:
        from models.database_models import Bucket, BucketType, Account
        from sqlalchemy import case
        from services.analytics.fx_service import get_cross_fx_rate
        import uuid

        with SessionLocal() as session:
            # 1. Idempotent check: Ensure canonical DEFAULT bucket exists
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

            # 2. Query buckets guaranteed with DEFAULT at index 0
            buckets = session.query(Bucket).filter(
                Bucket.user_id == current_user.user_uuid,
                Bucket.is_active == True
            ).order_by(
                case((Bucket.type == BucketType.DEFAULT, 0), else_=1),
                Bucket.priority_index
            ).all()

            bucket_list = []
            for b in buckets:
                b_dict = {c.name: getattr(b, c.name) for c in b.__table__.columns}
                if b_dict.get("cached_balance") is not None and b_dict["cached_balance"] < 0:
                    b_dict["cached_balance"] = 0.0
                bucket_list.append(b_dict)

            # 3. Multi-Currency Portfolio Reconciliation
            accounts = session.query(Account).filter_by(user_uuid=current_user.user_uuid).all()

            if currency and currency.strip().isalpha():
                base_curr = currency.strip().upper()
            elif accounts and accounts[0].currency:
                base_curr = accounts[0].currency.strip().upper()
            else:
                base_curr = "GBP"

            by_currency = {}
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

            default_id = default_bucket.id if default_bucket else (buckets[0].id if buckets else None)
            default_bal = round(float(default_bucket.cached_balance or 0.0), 2) if default_bucket else 0.0

            return {
                "status": "success",
                "base_currency": base_curr,
                "default_bucket_id": default_id,
                "default_bucket_balance": default_bal,
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
        logger.error(f"Error fetching buckets: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

class VirtualTransferRequest(BaseModel):
    source_bucket_id: Optional[str] = None
    target_bucket_id: str
    amount: float = Field(..., gt=0, description="Amount to transfer, must be strictly positive")

class BucketUpdateRequest(BaseModel):
    name: Optional[str] = None
    target_amount: Optional[float] = None
    target_date: Optional[datetime] = None
    priority_index: Optional[float] = None

@router.post("/transfer")
def execute_transfer_endpoint(request: VirtualTransferRequest, current_user: User = Depends(get_current_user)):
    from services.bucket_service import execute_bucket_transfer
    try:
        with SessionLocal() as session:
            execute_bucket_transfer(
                session=session,
                user_uuid=current_user.user_uuid,
                source_bucket_id=request.source_bucket_id,
                target_bucket_id=request.target_bucket_id,
                amount=request.amount
            )
            session.commit()
            
            # Publish to WebSocket
            from config import redis_client
            import json
            try:
                payload = {"event_type": "VIRTUAL_TRANSFER", "amount": request.amount}
                redis_client.publish(f"budai_events_{current_user.user_uuid}", json.dumps(payload))
            except:
                pass
                
            return {"status": "success", "message": "Transfer executed"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.patch("/{bucket_id}")
def update_bucket_endpoint(bucket_id: str, request: BucketUpdateRequest, current_user: User = Depends(get_current_user)):
    from services.bucket_service import update_bucket_details
    try:
        with SessionLocal() as session:
            updated = update_bucket_details(
                session=session,
                user_uuid=current_user.user_uuid,
                bucket_id=bucket_id,
                name=request.name,
                target_amount=request.target_amount,
                target_date=request.target_date,
                priority_index=request.priority_index
            )
            session.commit()
            
            # Publish to WebSocket
            from config import redis_client
            import json
            try:
                payload = {"event_type": "BUCKET_UPDATED", "bucket_id": bucket_id}
                redis_client.publish(f"budai_events_{current_user.user_uuid}", json.dumps(payload))
            except:
                pass
                
            return {"status": "success", "message": "Bucket updated"}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{bucket_id}")
def delete_bucket_endpoint(bucket_id: str, current_user: User = Depends(get_current_user)):
    from services.bucket_service import delete_virtual_bucket
    try:
        with SessionLocal() as session:
            delete_virtual_bucket(session, current_user.user_uuid, bucket_id)
            session.commit()
            
            # Publish to WebSocket
            from config import redis_client
            import json
            try:
                payload = {"event_type": "BUCKET_DELETED", "bucket_id": bucket_id}
                redis_client.publish(f"budai_events_{current_user.user_uuid}", json.dumps(payload))
            except:
                pass
                
            return {"status": "success", "message": "Bucket deleted successfully"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

class BucketActiveStateRequest(BaseModel):
    is_active: bool

@router.patch("/{bucket_id}/active")
def toggle_bucket_active_state_endpoint(bucket_id: str, request: BucketActiveStateRequest, current_user: User = Depends(get_current_user)):
    try:
        from models.database_models import Bucket
        with SessionLocal() as session:
            bucket = session.query(Bucket).filter_by(id=bucket_id, user_id=current_user.user_uuid).first()
            if not bucket:
                raise HTTPException(status_code=404, detail="Bucket not found.")
            
            if bucket.type == "DEFAULT" and not request.is_active:
                raise HTTPException(status_code=400, detail="Cannot deactivate the DEFAULT bucket.")
                
            bucket.is_active = request.is_active
            session.commit()
            
            from config import redis_client
            import json
            payload = {"event_type": "BUCKET_UPDATED"}
            redis_client.publish(f"budai_events_{current_user.user_uuid}", json.dumps(payload))
            
            return {"status": "success", "message": f"Bucket active state set to {request.is_active}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
