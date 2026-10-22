import json
import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Session
from models.database_models import Bucket
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

def create_new_bucket(
    session: Session, 
    user_uuid: str, 
    bucket_type: str, 
    name: str, 
    target_amount: Optional[float] = None, 
    target_date: Optional[datetime] = None, 
    priority_index: float = 0.0
) -> Bucket:
    """
    Core service to create a new virtual bucket.
    Used by both the REST API (frontend) and the LLM (Agentic Tool).
    """
    valid_types = ["DEFAULT", "RECURRING", "ACCUMULATING", "TARGET_DATE", "LIABILITY"]
    if bucket_type not in valid_types:
        raise ValueError(f"Invalid bucket type. Must be one of: {valid_types}")
        
    new_bucket = Bucket(
        id=str(uuid.uuid4()),
        user_id=user_uuid,
        type=bucket_type,
        name=name,
        target_amount=target_amount,
        target_date=target_date,
        priority_index=priority_index,
        cached_balance=0.0
    )
    
    session.add(new_bucket)
    log_pipeline_data(
        logger=logger,
        pipeline="bucket_service",
        stage="create_bucket",
        data={
            "bucket_id": new_bucket.id,
            "user_uuid": user_uuid,
            "name": name,
            "type": bucket_type,
            "target_amount": target_amount,
            "priority_index": priority_index
        },
        message=f"Created new {bucket_type} bucket '{name}' for user {user_uuid}"
    )
    
    return new_bucket

def execute_bucket_transfer(
    session: Session,
    user_uuid: str,
    source_bucket_id: str,
    target_bucket_id: str,
    amount: float
):
    from models.database_models import VirtualTransfer
    # Verify ownership
    source = session.query(Bucket).filter_by(id=source_bucket_id, user_id=user_uuid).first()
    target = session.query(Bucket).filter_by(id=target_bucket_id, user_id=user_uuid).first()
    
    if not source or not target:
        raise ValueError("Invalid source or target bucket.")
        
    if source.cached_balance < amount:
        raise ValueError(f"Insufficient funds. The source bucket only has £{source.cached_balance:.2f} available.")
        
    transfer = VirtualTransfer(
        id=str(uuid.uuid4()),
        source_bucket_id=source_bucket_id,
        target_bucket_id=target_bucket_id,
        amount=amount
    )
    session.add(transfer)
    # Note: PL/pgSQL database trigger automatically updates the cached_balance 
    # of both buckets when this insert commits.
    
    log_pipeline_data(
        logger=logger,
        pipeline="bucket_service",
        stage="execute_transfer",
        data={
            "transfer_id": transfer.id,
            "user_uuid": user_uuid,
            "source_bucket": {"id": source.id, "name": source.name, "balance_before": source.cached_balance},
            "target_bucket": {"id": target.id, "name": target.name, "balance_before": target.cached_balance},
            "amount": amount
        },
        message=f"Virtual transfer of £{amount} staged from '{source.name}' to '{target.name}'"
    )
    return transfer

def update_bucket_details(
    session: Session,
    user_uuid: str,
    bucket_id: str,
    name: Optional[str] = None,
    target_amount: Optional[float] = None,
    target_date: Optional[datetime] = None,
    priority_index: Optional[float] = None
) -> Bucket:
    bucket = session.query(Bucket).filter_by(id=bucket_id, user_id=user_uuid).first()
    if not bucket:
        raise ValueError("Bucket not found.")
        
    if name is not None:
        bucket.name = name
    if target_amount is not None:
        bucket.target_amount = target_amount
    if target_date is not None:
        bucket.target_date = target_date
    if priority_index is not None:
        bucket.priority_index = priority_index
        
    bucket.updated_at = datetime.now(timezone.utc)
        
    log_pipeline_data(
        logger=logger,
        pipeline="bucket_service",
        stage="update_bucket",
        data={
            "bucket_id": bucket_id,
            "user_uuid": user_uuid,
            "name": bucket.name,
            "target_amount": bucket.target_amount,
            "priority_index": bucket.priority_index
        },
        message=f"Updated bucket '{bucket.name}' ({bucket_id}) for user {user_uuid}"
    )
    return bucket

def delete_virtual_bucket(session: Session, user_uuid: str, bucket_id: str):
    bucket = session.query(Bucket).filter_by(id=bucket_id, user_id=user_uuid).first()
    if not bucket:
        raise ValueError("Bucket not found or already deleted.")
        
    if bucket.type == "DEFAULT":
        raise ValueError("Cannot delete the DEFAULT bucket.")
        
    # Zero-Sum Safety: Transfer remaining funds back to DEFAULT bucket
    amount_to_recover = bucket.cached_balance
    if amount_to_recover > 0:
        default_bucket = session.query(Bucket).filter_by(user_id=user_uuid, type="DEFAULT").first()
        if not default_bucket:
            raise ValueError("System error: No DEFAULT bucket found for user.")
            
        # Manually update the DEFAULT bucket's balance to recover the funds
        default_bucket.cached_balance += amount_to_recover
        bucket.cached_balance = 0.0
        session.flush()
        
    from models.database_models import VirtualTransfer
    # Delete historical transfers to satisfy foreign key constraints before hard deleting
    session.query(VirtualTransfer).filter(
        (VirtualTransfer.source_bucket_id == bucket.id) | 
        (VirtualTransfer.target_bucket_id == bucket.id)
    ).delete(synchronize_session=False)
    
    # Now hard delete the bucket
    session.delete(bucket)
    
    log_pipeline_data(
        logger=logger,
        pipeline="bucket_service",
        stage="delete_bucket",
        data={
            "bucket_id": bucket_id,
            "user_uuid": user_uuid,
            "recovered_to_default": amount_to_recover
        },
        message=f"Bucket {bucket_id} removed, £{amount_to_recover} returned to DEFAULT"
    )
    return True

