from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from middleware.auth_middleware import get_current_user
from models.database_models import User
from controllers.bucket_controller import (
    get_user_buckets,
    create_bucket,
    execute_transfer,
    update_bucket,
    delete_bucket,
    toggle_bucket_active
)
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

router = APIRouter(prefix="/api/buckets", tags=["Buckets"])


class BucketCreateRequest(BaseModel):
    bucket_type: str
    name: str
    target_amount: Optional[float] = None
    target_date: Optional[datetime] = None
    priority: Optional[int] = 10
    priority_index: Optional[float] = None
    currency: Optional[str] = "GBP"


class VirtualTransferRequest(BaseModel):
    source_bucket_id: Optional[str] = None
    target_bucket_id: str
    amount: float = Field(..., gt=0, description="Amount to transfer, must be strictly positive")


class BucketUpdateRequest(BaseModel):
    name: Optional[str] = None
    target_amount: Optional[float] = None
    target_date: Optional[datetime] = None
    priority: Optional[int] = None
    priority_index: Optional[float] = None


class BucketActiveStateRequest(BaseModel):
    is_active: bool


@router.get("")
@router.get("/")
def get_user_buckets_route(
    currency: Optional[str] = Query(None, description="Preferred display currency e.g. GBP, USD, EUR"),
    base_currency: Optional[str] = Query(None, description="Alias for preferred currency"),
    current_user: User = Depends(get_current_user)
):
    target_curr = currency or base_currency
    return get_user_buckets(user_uuid=current_user.user_uuid, currency=target_curr)


@router.post("")
@router.post("/")
def create_bucket_route(
    request: BucketCreateRequest,
    current_user: User = Depends(get_current_user)
):
    priority_val = request.priority if request.priority is not None else (int(request.priority_index) if request.priority_index is not None else 10)
    return create_bucket(
        user_uuid=current_user.user_uuid,
        bucket_type=request.bucket_type,
        name=request.name,
        target_amount=request.target_amount,
        target_date=request.target_date,
        priority=priority_val,
        currency=request.currency or "GBP"
    )


@router.post("/transfer")
def execute_transfer_route(
    request: VirtualTransferRequest,
    current_user: User = Depends(get_current_user)
):
    return execute_transfer(
        user_uuid=current_user.user_uuid,
        source_bucket_id=request.source_bucket_id,
        target_bucket_id=request.target_bucket_id,
        amount=request.amount
    )


@router.patch("/{bucket_id}")
def update_bucket_route(
    bucket_id: str,
    request: BucketUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    priority_val = request.priority if request.priority is not None else (int(request.priority_index) if request.priority_index is not None else None)
    return update_bucket(
        user_uuid=current_user.user_uuid,
        bucket_id=bucket_id,
        name=request.name,
        target_amount=request.target_amount,
        target_date=request.target_date,
        priority=priority_val
    )


@router.delete("/{bucket_id}")
def delete_bucket_route(
    bucket_id: str,
    current_user: User = Depends(get_current_user)
):
    return delete_bucket(user_uuid=current_user.user_uuid, bucket_id=bucket_id)


@router.patch("/{bucket_id}/active")
def toggle_bucket_active_route(
    bucket_id: str,
    request: BucketActiveStateRequest,
    current_user: User = Depends(get_current_user)
):
    return toggle_bucket_active(
        user_uuid=current_user.user_uuid,
        bucket_id=bucket_id,
        is_active=request.is_active
    )
