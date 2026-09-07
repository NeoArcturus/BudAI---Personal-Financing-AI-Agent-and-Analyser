import json
import uuid
from sqlmodel import Session, select
from models.database_models import Bucket, VirtualTransfer, SystemAlert, Account
from services.logger_setup import get_core_logger
from sqlalchemy import text
from datetime import datetime

logger = get_core_logger(__name__)

def perform_physical_delta_sync(session: Session, user_uuid: str):
    """
    Step 1 & 2 of Phase 2: Physical Delta Sync
    Compares the total physical bank balance to the total virtual buckets balance.
    If there is a difference (e.g. unbudgeted physical spend), deducts from DEFAULT bucket.
    """
    logger.info(json.dumps({"message": f"Starting Physical Delta Sync for user {user_uuid}", "status_code": 200}))
    try:
        accounts = session.execute(select(Account).where(Account.user_uuid == user_uuid)).scalars().all()
        physical_total = sum(acc.account_balance for acc in accounts if acc.account_balance is not None)
        
        buckets = session.execute(select(Bucket).where(Bucket.user_id == user_uuid)).scalars().all()
        virtual_total = sum(b.cached_balance for b in buckets)
        
        delta = physical_total - virtual_total
        
        if abs(delta) < 0.01:
            logger.info(json.dumps({"message": "Physical and Virtual balances perfectly aligned.", "status_code": 200}))
            return {"status": "aligned", "delta": 0}
            
        default_bucket = next((b for b in buckets if b.type == "DEFAULT"), None)
        if not default_bucket:
            logger.error(json.dumps({"message": "Critical Error: No DEFAULT bucket found for user.", "status_code": 500}))
            return {"status": "error", "message": "No DEFAULT bucket"}
            
        logger.info(json.dumps({"message": f"Delta detected: {delta}. Aligning DEFAULT bucket...", "status_code": 200}))
        
        cascaded_from = []
        if delta > 0:
            # Income: Safely add to DEFAULT
            session.execute(text("UPDATE buckets SET cached_balance = cached_balance + :delta WHERE id = :b_id"), {"delta": delta, "b_id": default_bucket.id})
        else:
            # Expense: We must ensure DEFAULT doesn't go negative during the deduction
            spend = abs(delta)
            if default_bucket.cached_balance < spend:
                deficit = spend - default_bucket.cached_balance
                logger.warning(json.dumps({"message": f"DEFAULT bucket short by £{deficit}. Triggering In-Memory Cascade.", "status_code": 400}))
                cascaded_from = run_priority_cascade(session, user_uuid, default_bucket, deficit)
            
            # Now that DEFAULT has been padded by the cascade (or had enough to begin with), deduct the spend
            session.execute(text("UPDATE buckets SET cached_balance = cached_balance - :spend WHERE id = :b_id"), {"spend": spend, "b_id": default_bucket.id})
            
        session.commit()
        return {"status": "synced", "delta": delta, "cascaded_from": cascaded_from}
        
    except Exception as e:
        session.rollback()
        logger.error(json.dumps({"message": f"Delta Sync failed: {e}", "status_code": 500}), exc_info=True)
        return {"status": "error"}

def run_priority_cascade(session: Session, user_uuid: str, default_bucket: Bucket, deficit: float):
    """
    Step 3 of Phase 2: Priority Cascade (The Overdraft Engine)
    Moves money from lowest priority buckets to DEFAULT so it can safely absorb a physical deficit.
    """
    cascaded_buckets = []
    try:
        other_buckets = session.execute(
            select(Bucket)
            .where(Bucket.user_id == user_uuid)
            .where(Bucket.type != "DEFAULT")
            .where(Bucket.cached_balance > 0)
            .order_by(Bucket.priority_index.desc())
        ).scalars().all()
        
        for bucket in other_buckets:
            if deficit <= 0.01:
                break
                
            drain_amount = min(bucket.cached_balance, deficit)
            
            transfer = VirtualTransfer(
                id=str(uuid.uuid4()),
                source_bucket_id=bucket.id,
                target_bucket_id=default_bucket.id,
                amount=drain_amount
            )
            session.add(transfer)
            session.flush() # DB trigger immediately moves money to DEFAULT in this transaction
            
            deficit -= drain_amount
            cascaded_buckets.append(bucket.name)
            logger.info(json.dumps({"message": f"Cascaded £{drain_amount} from {bucket.name} to DEFAULT", "status_code": 200}))

        if deficit > 0.01:
            logger.error(json.dumps({"message": f"CRITICAL: User is physically overdrawn by £{deficit}.", "status_code": 500}))
            alert = SystemAlert(
                id=str(uuid.uuid4()),
                user_id=user_uuid,
                event_type="CRITICAL_OVERDRAFT",
                message=f"CRITICAL: Your physical accounts are overdrawn. We drained all virtual buckets and you are still short by £{deficit:.2f}."
            )
            session.add(alert)
            # We still add the deficit as a negative to DEFAULT if literally everything is drained 
            # WAIT! The DB constraint prevents negative balances!
            # If the user is PHYSICALLY overdrawn (bank account is literally negative), 
            # we MUST allow the system to reflect this. BUT the DB constraint forbids it!
            # If physical bank is -10, total virtual MUST be -10 to match.
            # If we enforce cached_balance >= 0 on all buckets, the DB literally cannot represent a real-life physical overdraft!
        
        return cascaded_buckets
        
    except Exception as e:
        session.rollback()
        logger.error(json.dumps({"message": f"Priority Cascade failed: {e}", "status_code": 500}), exc_info=True)
        return cascaded_buckets
