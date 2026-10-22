import json
import uuid
from sqlmodel import Session, select
from models.database_models import Bucket, VirtualTransfer, SystemAlert, Account
from services.logger_setup import get_core_logger, log_pipeline_data
from sqlalchemy import text

logger = get_core_logger("delta_sync")

def perform_physical_delta_sync(session: Session, user_uuid: str):
    """
    Step 1 & 2 of Phase 2: Physical Delta Sync
    Compares the total physical bank balance to the total virtual buckets balance.
    If there is a difference (e.g. unbudgeted physical spend), deducts from DEFAULT bucket.
    """
    try:
        accounts = session.execute(select(Account).where(Account.user_uuid == user_uuid)).scalars().all()
        physical_total = sum(acc.account_balance for acc in accounts if acc.account_balance is not None)
        
        buckets = session.execute(select(Bucket).where(Bucket.user_id == user_uuid)).scalars().all()
        virtual_total = sum(b.cached_balance for b in buckets)
        
        delta = physical_total - virtual_total
        
        log_pipeline_data(
            logger,
            pipeline="physical_delta_sync",
            stage="balance_comparison",
            data={
                "user_uuid": user_uuid,
                "physical_total": round(physical_total, 2),
                "virtual_total": round(virtual_total, 2),
                "delta": round(delta, 2),
                "connected_accounts": len(accounts),
                "virtual_buckets": len(buckets)
            },
            message=f"Delta sync evaluated: Physical=£{physical_total:.2f}, Virtual=£{virtual_total:.2f}, Delta=£{delta:.2f}"
        )
        
        if abs(delta) < 0.01:
            return {"status": "aligned", "delta": 0}
            
        default_bucket = next((b for b in buckets if b.type == "DEFAULT"), None)
        if not default_bucket:
            logger.error(
                "Critical Error: No DEFAULT bucket found for user",
                extra={"user_uuid": user_uuid, "status_code": 500}
            )
            return {"status": "error", "message": "No DEFAULT bucket"}
            
        cascaded_from = []
        if delta > 0:
            # Income: Safely add to DEFAULT
            session.execute(
                text("UPDATE buckets SET cached_balance = cached_balance + :delta WHERE id = :b_id"),
                {"delta": delta, "b_id": default_bucket.id}
            )
        else:
            # Expense: We must ensure DEFAULT doesn't go negative during the deduction
            spend = abs(delta)
            if default_bucket.cached_balance < spend:
                deficit = spend - default_bucket.cached_balance
                cascaded_from = run_priority_cascade(session, user_uuid, default_bucket, deficit)
            
            # Now that DEFAULT has been padded by the cascade (or had enough to begin with), deduct the spend
            session.execute(
                text("UPDATE buckets SET cached_balance = cached_balance - :spend WHERE id = :b_id"),
                {"spend": spend, "b_id": default_bucket.id}
            )
            
        session.commit()

        log_pipeline_data(
            logger,
            pipeline="physical_delta_sync",
            stage="sync_complete",
            data={
                "user_uuid": user_uuid,
                "delta": round(delta, 2),
                "cascaded_from": cascaded_from,
                "final_default_balance": round(default_bucket.cached_balance, 2)
            },
            message=f"Physical delta sync successfully reconciled: Delta £{delta:.2f}"
        )
        return {"status": "synced", "delta": delta, "cascaded_from": cascaded_from}
        
    except Exception as e:
        session.rollback()
        logger.error(
            f"Delta Sync failed: {e}",
            extra={"user_uuid": user_uuid, "status_code": 500},
            exc_info=True
        )
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

        log_pipeline_data(
            logger,
            pipeline="priority_cascade",
            stage="waterfall_initiated",
            data={
                "user_uuid": user_uuid,
                "deficit": round(deficit, 2),
                "candidate_buckets": [{"name": b.name, "balance": b.cached_balance, "priority": b.priority_index} for b in other_buckets]
            },
            message=f"Initiating priority cascade to cover deficit of £{deficit:.2f}"
        )
        
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
            
            log_pipeline_data(
                logger,
                pipeline="priority_cascade",
                stage="bucket_drained",
                data={
                    "source_bucket": bucket.name,
                    "target_bucket": "DEFAULT",
                    "amount": round(drain_amount, 2),
                    "remaining_deficit": round(deficit, 2)
                },
                message=f"Cascaded £{drain_amount:.2f} from '{bucket.name}' to DEFAULT"
            )

        if deficit > 0.01:
            logger.error(
                f"CRITICAL: User is physically overdrawn by £{deficit:.2f}",
                extra={"user_uuid": user_uuid, "deficit": round(deficit, 2), "status_code": 500}
            )
            alert = SystemAlert(
                id=str(uuid.uuid4()),
                user_id=user_uuid,
                event_type="CRITICAL_OVERDRAFT",
                message=f"CRITICAL: Your physical accounts are overdrawn. We drained all virtual buckets and you are still short by £{deficit:.2f}."
            )
            session.add(alert)
        
        return cascaded_buckets
        
    except Exception as e:
        session.rollback()
        logger.error(
            f"Priority Cascade failed: {e}",
            extra={"user_uuid": user_uuid, "status_code": 500},
            exc_info=True
        )
        return cascaded_buckets
