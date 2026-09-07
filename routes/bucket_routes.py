from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from config import SessionLocal
from services.bucket_service import create_new_bucket
from middleware.auth_middleware import get_current_user
from models.database_models import User

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
def get_user_buckets(current_user: User = Depends(get_current_user)):
    try:
        from models.database_models import Bucket
        with SessionLocal() as session:
            buckets = session.query(Bucket).filter(Bucket.user_id == current_user.user_uuid, Bucket.is_active == True).order_by(Bucket.priority_index).all()
            
            # The UI should never display negative numbers (physical overdrafts are handled 
            # gracefully as 0 available virtual money)
            bucket_list = []
            for b in buckets:
                b_dict = {c.name: getattr(b, c.name) for c in b.__table__.columns}
                if b_dict.get("cached_balance") is not None and b_dict["cached_balance"] < 0:
                    b_dict["cached_balance"] = 0.0
                bucket_list.append(b_dict)
                
            return {"status": "success", "buckets": bucket_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

class VirtualTransferRequest(BaseModel):
    source_bucket_id: str
    target_bucket_id: str
    amount: float

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
