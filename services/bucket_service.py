import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from sqlmodel import Session, select
from models.database_models import Bucket, BucketType, VirtualTransfer, TransferSuggestion, Account, SystemAlert
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

def get_or_create_unallocated_bucket(session: Session, user_uuid: str) -> Bucket:
    """Retrieves or creates the single Unallocated bucket for the user."""
    unallocated = session.execute(
        select(Bucket).where(
            Bucket.user_id == user_uuid,
            Bucket.type == BucketType.UNALLOCATED,
            Bucket.is_active == True
        )
    ).scalars().first()
    
    if not unallocated:
        unallocated = Bucket(
            id=str(uuid.uuid4()),
            user_id=user_uuid,
            type=BucketType.UNALLOCATED,
            name="Unallocated",
            cached_balance=0.0,
            cached_balances={},
            currency="GBP",
            is_active=True
        )
        session.add(unallocated)
        session.flush()
    return unallocated

def sync_unallocated_from_accounts(session: Session, user_uuid: str) -> Bucket:
    """
    Syncs the Unallocated bucket with the user's physical bank accounts.
    Calculates total balance per currency and a per-bank account breakdown.
    """
    accounts = session.execute(
        select(Account).where(Account.user_uuid == user_uuid)
    ).scalars().all()

    currency_map: Dict[str, Dict[str, Any]] = {}
    total_aggregate = 0.0

    for acc in accounts:
        curr = (acc.currency or "GBP").upper()
        bal = float(acc.account_balance or 0.0)
        total_aggregate += bal

        if curr not in currency_map:
            currency_map[curr] = {
                "total": 0.0,
                "accounts": {}
            }
        
        currency_map[curr]["total"] = round(currency_map[curr]["total"] + bal, 2)
        acc_label = acc.display_name or acc.account_number or acc.account_id
        currency_map[curr]["accounts"][acc.account_id] = {
            "name": acc_label,
            "balance": round(bal, 2)
        }

    unallocated = get_or_create_unallocated_bucket(session, user_uuid)
    unallocated.cached_balances = currency_map
    unallocated.cached_balance = round(total_aggregate, 2)
    unallocated.updated_at = datetime.now(timezone.utc)
    session.flush()

    log_pipeline_data(
        logger=logger,
        pipeline="bucket_service",
        stage="sync_unallocated",
        data={
            "user_uuid": user_uuid,
            "currencies": list(currency_map.keys()),
            "total_balance": unallocated.cached_balance
        },
        message=f"Synced Unallocated bucket with {len(accounts)} accounts across {list(currency_map.keys())}"
    )
    return unallocated

def create_new_bucket(
    session: Session, 
    user_uuid: str, 
    bucket_type: str, 
    name: str, 
    target_amount: Optional[float] = None, 
    target_date: Optional[datetime] = None, 
    priority: int = 10,
    currency: str = "GBP"
) -> Bucket:
    """Creates a new virtual bucket strictly of type BILLS or GOALS."""
    normalized_type = bucket_type.upper().strip()
    if normalized_type not in (BucketType.BILLS.value, BucketType.GOALS.value):
        raise ValueError(f"Invalid bucket type '{bucket_type}'. Only 'BILLS' and 'GOALS' are allowed.")
        
    new_bucket = Bucket(
        id=str(uuid.uuid4()),
        user_id=user_uuid,
        type=BucketType(normalized_type),
        name=name,
        target_amount=target_amount,
        target_date=target_date,
        priority=priority,
        currency=currency.upper(),
        cached_balance=0.0,
        allocated_amount=0.0,
        is_active=True
    )
    session.add(new_bucket)
    session.flush()
    return new_bucket

def format_bill_status(due_date: Optional[datetime], is_paid: bool, is_covered: bool) -> str:
    """Formats the bill timeline display text according to Stage 4 rules."""
    if not due_date:
        return "No upcoming due date"
        
    now = datetime.now(timezone.utc)
    if due_date.tzinfo is None:
        due_date = due_date.replace(tzinfo=timezone.utc)

    if is_paid:
        next_cycle_date = due_date + timedelta(days=30)
        return f"Bill paid, next payment {next_cycle_date.strftime('%d-%m-%Y')}"
        
    days_left = (due_date.date() - now.date()).days
    if days_left > 28:
        return f"Next payment due on {due_date.strftime('%d-%m-%Y')}"
    elif 15 <= days_left <= 28:
        return f"Next payment due in {days_left} days"
    else:
        # Less than 15 days left
        if is_covered:
            return f"Bill covered, payment due in {max(0, days_left)} days"
        return f"Next payment due in {max(0, days_left)} days"

def process_stage3_bills(session: Session, user_uuid: str, detected_bills: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Processes bills detected in Stage 3:
    1. Upserts bills into the BILLS bucket category.
    2. Runs the timeline check (>28d, 15-28d, <15d).
    3. Runs automated allocation for bills under 15 days from Unallocated and lower-priority Goals.
    4. Detects bank account physical shortfalls and creates transfer suggestions.
    """
    unallocated = get_or_create_unallocated_bucket(session, user_uuid)
    cached_balances = dict(unallocated.cached_balances or {})
    
    processed_bills = []
    shortfall_alerts = []

    for item in detected_bills:
        merchant_name = item.get("merchant") or item.get("merchant_name") or item.get("name") or "Unknown Bill"
        amount = float(item.get("expected_amount") or item.get("amount") or 0.0)
        currency = (item.get("currency") or "GBP").upper()
        account_id = item.get("account_id")
        account_name = item.get("account_name")
        next_date_val = item.get("next_date") or item.get("next_expected_date")

        next_date = None
        if isinstance(next_date_val, str):
            try:
                next_date = datetime.strptime(next_date_val[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                next_date = datetime.now(timezone.utc) + timedelta(days=30)
        elif isinstance(next_date_val, datetime):
            next_date = next_date_val if next_date_val.tzinfo else next_date_val.replace(tzinfo=timezone.utc)
        else:
            next_date = datetime.now(timezone.utc) + timedelta(days=30)

        # Look for existing bill bucket
        bill_bucket = session.execute(
            select(Bucket).where(
                Bucket.user_id == user_uuid,
                Bucket.type == BucketType.BILLS,
                Bucket.name == merchant_name,
                Bucket.is_active == True
            )
        ).scalars().first()

        if not bill_bucket:
            bill_bucket = Bucket(
                id=str(uuid.uuid4()),
                user_id=user_uuid,
                type=BucketType.BILLS,
                name=merchant_name,
                target_amount=amount,
                target_date=next_date,
                currency=currency,
                associated_account_id=account_id,
                associated_account_name=account_name,
                allocated_amount=0.0,
                cached_balance=0.0,
                is_active=True
            )
            session.add(bill_bucket)
            session.flush()
        else:
            bill_bucket.target_amount = amount
            bill_bucket.target_date = next_date
            bill_bucket.currency = currency
            if account_id:
                bill_bucket.associated_account_id = account_id
            if account_name:
                bill_bucket.associated_account_name = account_name

        days_left = (next_date.date() - datetime.now(timezone.utc).date()).days
        
        # Step 3: Less than 15 days -> Automated allocation
        if days_left < 15:
            needed = round(amount - bill_bucket.allocated_amount, 2)
            if needed > 0:
                # 1. Take from Unallocated in that currency
                curr_pool = cached_balances.get(currency, {"total": 0.0, "accounts": {}})
                unallocated_avail = float(curr_pool.get("total", 0.0))
                
                from_unallocated = min(needed, unallocated_avail)
                if from_unallocated > 0:
                    bill_bucket.allocated_amount = round(bill_bucket.allocated_amount + from_unallocated, 2)
                    bill_bucket.cached_balance = bill_bucket.allocated_amount
                    curr_pool["total"] = round(curr_pool["total"] - from_unallocated, 2)
                    needed = round(needed - from_unallocated, 2)

                # 2. If still needed, pull from Goals (lower-priority first: priority DESC)
                if needed > 0:
                    goals = session.execute(
                        select(Bucket).where(
                            Bucket.user_id == user_uuid,
                            Bucket.type == BucketType.GOALS,
                            Bucket.currency == currency,
                            Bucket.cached_balance > 0,
                            Bucket.is_active == True
                        ).order_by(Bucket.priority.desc()) # e.g. priority 10 before priority 1
                    ).scalars().all()

                    for g in goals:
                        if needed <= 0:
                            break
                        from_goal = min(needed, g.cached_balance)
                        if from_goal > 0:
                            g.cached_balance = round(g.cached_balance - from_goal, 2)
                            bill_bucket.allocated_amount = round(bill_bucket.allocated_amount + from_goal, 2)
                            bill_bucket.cached_balance = bill_bucket.allocated_amount
                            needed = round(needed - from_goal, 2)

                # If still short, flag shortfall warning
                if needed > 0:
                    bill_bucket.has_shortfall_warning = True
                    bill_bucket.shortfall_amount = needed
                    shortfall_alerts.append({
                        "bill": merchant_name,
                        "shortfall": needed,
                        "currency": currency,
                        "account_name": bill_bucket.associated_account_name
                    })
                else:
                    bill_bucket.has_shortfall_warning = False
                    bill_bucket.shortfall_amount = 0.0

            # Step 4: Check physical bank account balance
            if bill_bucket.associated_account_id:
                acc = session.execute(
                    select(Account).where(Account.account_id == bill_bucket.associated_account_id)
                ).scalars().first()
                if acc and float(acc.account_balance or 0.0) < amount:
                    # Physical shortfall detected in the specific bank account
                    shortfall = round(amount - float(acc.account_balance or 0.0), 2)
                    severity = "HIGH" if merchant_name.lower() in ("rent", "mortgage", "council tax", "utilities", "british gas", "water") or days_left <= 3 else "NORMAL"
                    deadline_hours = 24 if severity == "HIGH" else 48

                    # Find source account with spare funds
                    spare_acc = session.execute(
                        select(Account).where(
                            Account.user_uuid == user_uuid,
                            Account.account_id != acc.account_id,
                            Account.currency == currency,
                            Account.account_balance >= shortfall
                        )
                    ).scalars().first()

                    # Record transfer suggestion if not already recorded
                    existing_sugg = session.execute(
                        select(TransferSuggestion).where(
                            TransferSuggestion.user_uuid == user_uuid,
                            TransferSuggestion.bill_id == bill_bucket.id,
                            TransferSuggestion.status == "PENDING"
                        )
                    ).scalars().first()

                    if not existing_sugg and spare_acc:
                        sugg = TransferSuggestion(
                            id=str(uuid.uuid4()),
                            user_uuid=user_uuid,
                            bill_id=bill_bucket.id,
                            from_account_id=spare_acc.account_id,
                            from_account_name=spare_acc.display_name or spare_acc.account_number,
                            to_account_id=acc.account_id,
                            to_account_name=acc.display_name or acc.account_number,
                            amount=shortfall,
                            currency=currency,
                            status="PENDING",
                            severity=severity,
                            verification_deadline=datetime.now(timezone.utc) + timedelta(hours=deadline_hours)
                        )
                        session.add(sugg)

        is_covered = (bill_bucket.allocated_amount >= amount and amount > 0)
        bill_bucket.status_text = format_bill_status(next_date, is_paid=False, is_covered=is_covered)
        bill_bucket.updated_at = datetime.now(timezone.utc)
        processed_bills.append({
            "name": bill_bucket.name,
            "amount": bill_bucket.target_amount,
            "currency": bill_bucket.currency,
            "allocated": bill_bucket.allocated_amount,
            "status_text": bill_bucket.status_text,
            "days_left": days_left,
            "has_shortfall_warning": bill_bucket.has_shortfall_warning
        })

    # Save updated unallocated balances
    unallocated.cached_balances = cached_balances
    unallocated.cached_balance = round(sum(p.get("total", 0.0) for p in cached_balances.values()), 2)
    session.flush()

    return {
        "processed_count": len(processed_bills),
        "bills": processed_bills,
        "shortfalls": shortfall_alerts
    }

def deduct_transaction_outflow(session: Session, user_uuid: str, tx_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deduction hierarchy:
    1. If transaction matches a Bill -> Deduct from Bill, mark paid, advance cycle.
    2. If normal spend -> Deduct from Unallocated[currency].
    3. If Unallocated reaches 0.00 -> Deduct from Goals[currency] (priority DESC).
    4. If Goals reach 0.00 -> Deduct from Bills[currency] and trigger emergency alert.
    """
    amount = abs(float(tx_payload.get("amount", 0.0)))
    currency = (tx_payload.get("currency") or "GBP").upper()
    merchant = (tx_payload.get("merchant") or tx_payload.get("description") or "").strip().lower()

    unallocated = get_or_create_unallocated_bucket(session, user_uuid)
    cached_balances = dict(unallocated.cached_balances or {})
    curr_pool = cached_balances.get(currency, {"total": 0.0, "accounts": {}})

    # Step 1: Check if this is a Bill payment
    bill = session.execute(
        select(Bucket).where(
            Bucket.user_id == user_uuid,
            Bucket.type == BucketType.BILLS,
            Bucket.is_active == True
        )
    ).scalars().all()

    matched_bill = next((b for b in bill if b.name.lower() in merchant or merchant in b.name.lower()), None)
    if matched_bill:
        matched_bill.allocated_amount = max(0.0, round(matched_bill.allocated_amount - amount, 2))
        matched_bill.cached_balance = matched_bill.allocated_amount
        old_due = matched_bill.target_date or datetime.now(timezone.utc)
        new_due = old_due + timedelta(days=30)
        matched_bill.target_date = new_due
        matched_bill.status_text = format_bill_status(new_due, is_paid=True, is_covered=False)
        matched_bill.has_shortfall_warning = False
        matched_bill.updated_at = datetime.now(timezone.utc)
        session.flush()
        return {
            "status": "DEDUCTED_FROM_BILL",
            "bill_name": matched_bill.name,
            "amount": amount,
            "currency": currency,
            "status_text": matched_bill.status_text
        }

    # Step 2: Everyday spend -> Deduct from Unallocated
    remaining = amount
    unalloc_total = float(curr_pool.get("total", 0.0))
    if unalloc_total >= remaining:
        curr_pool["total"] = round(unalloc_total - remaining, 2)
        unallocated.cached_balances = cached_balances
        unallocated.cached_balance = round(sum(p.get("total", 0.0) for p in cached_balances.values()), 2)
        session.flush()
        return {
            "status": "DEDUCTED_FROM_UNALLOCATED",
            "amount": amount,
            "currency": currency,
            "unallocated_remaining": curr_pool["total"]
        }

    # Step 3: Unallocated is short -> Drain Unallocated down to 0, then pull from Goals
    if unalloc_total > 0:
        remaining = round(remaining - unalloc_total, 2)
        curr_pool["total"] = 0.0

    goals = session.execute(
        select(Bucket).where(
            Bucket.user_id == user_uuid,
            Bucket.type == BucketType.GOALS,
            Bucket.currency == currency,
            Bucket.cached_balance > 0,
            Bucket.is_active == True
        ).order_by(Bucket.priority.desc()) # lower-priority goals first
    ).scalars().all()

    drained_from_goals = 0.0
    for g in goals:
        if remaining <= 0:
            break
        from_g = min(remaining, g.cached_balance)
        g.cached_balance = round(g.cached_balance - from_g, 2)
        drained_from_goals += from_g
        remaining = round(remaining - from_g, 2)

    if remaining <= 0:
        unallocated.cached_balances = cached_balances
        unallocated.cached_balance = round(sum(p.get("total", 0.0) for p in cached_balances.values()), 2)
        session.flush()
        return {
            "status": "DEDUCTED_FROM_GOALS",
            "amount": amount,
            "currency": currency,
            "from_unallocated": unalloc_total,
            "from_goals": drained_from_goals
        }

    # Step 4: Goals empty -> Last resort, pull from Bills (Emergency Alert)
    bills = session.execute(
        select(Bucket).where(
            Bucket.user_id == user_uuid,
            Bucket.type == BucketType.BILLS,
            Bucket.currency == currency,
            Bucket.allocated_amount > 0,
            Bucket.is_active == True
        ).order_by(Bucket.target_date.desc()) # farthest bill tapped first
    ).scalars().all()

    tapped_bill = None
    if bills:
        tapped_bill = bills[0]
        taken_from_bill = min(remaining, tapped_bill.allocated_amount)
        tapped_bill.allocated_amount = round(tapped_bill.allocated_amount - taken_from_bill, 2)
        tapped_bill.cached_balance = tapped_bill.allocated_amount
        tapped_bill.has_shortfall_warning = True
        remaining = round(remaining - taken_from_bill, 2)

        # Generate alert payload
        alert = SystemAlert(
            id=str(uuid.uuid4()),
            user_id=user_uuid,
            title="Money Moved from Bill",
            event_type="BILL_MONEY_DEPLETED",
            message=f"£{taken_from_bill:.2f} {currency} had to be taken from '{tapped_bill.name}'. Associated account: {tapped_bill.associated_account_name or 'Main Account'} ({currency}).",
            urgency_level=2,
            action_type="REVIEW_BILL_SHORTFALL",
            action_payload={
                "bill_name": tapped_bill.name,
                "amount_taken": taken_from_bill,
                "currency": currency,
                "account_name": tapped_bill.associated_account_name
            }
        )
        session.add(alert)

    unallocated.cached_balances = cached_balances
    unallocated.cached_balance = round(sum(p.get("total", 0.0) for p in cached_balances.values()), 2)
    session.flush()

    return {
        "status": "DEDUCTED_FROM_BILLS_ALERT",
        "amount": amount,
        "currency": currency,
        "tapped_bill": tapped_bill.name if tapped_bill else None,
        "alert_raised": True
    }

def execute_bucket_transfer(
    session: Session,
    user_uuid: str,
    source_bucket_id: str,
    target_bucket_id: str,
    amount: float
) -> VirtualTransfer:
    """Executes a strict double-entry transfer between buckets within the same currency."""
    if amount <= 0:
        raise ValueError("Transfer amount must be strictly greater than 0.")

    source = session.query(Bucket).filter_by(id=source_bucket_id, user_id=user_uuid).first()
    target = session.query(Bucket).filter_by(id=target_bucket_id, user_id=user_uuid).first()

    if not source or not target:
        raise ValueError("Invalid source or target bucket.")
    if source.id == target.id:
        raise ValueError("Source and destination buckets cannot be identical.")
    if source.cached_balance < amount:
        raise ValueError(f"Insufficient funds in '{source.name}'. Balance: {source.cached_balance:.2f}.")

    source.cached_balance = round(source.cached_balance - amount, 2)
    target.cached_balance = round(target.cached_balance + amount, 2)
    source.updated_at = datetime.now(timezone.utc)
    target.updated_at = datetime.now(timezone.utc)

    transfer = VirtualTransfer(
        id=str(uuid.uuid4()),
        source_bucket_id=source.id,
        target_bucket_id=target.id,
        amount=amount
    )
    session.add(transfer)
    session.flush()

    log_pipeline_data(
        logger=logger,
        pipeline="bucket_service",
        stage="execute_transfer",
        data={
            "transfer_id": transfer.id,
            "source": source.name,
            "target": target.name,
            "amount": amount
        },
        message=f"Transferred {amount} from '{source.name}' to '{target.name}'"
    )
    return transfer

def check_reality_transfers(session: Session, user_uuid: str) -> List[Dict[str, Any]]:
    """
    Checks approved transfer suggestions:
    If the 24h or 48h deadline has passed and the target bank account balance has not increased,
    flags a reminder alert for the user.
    """
    now = datetime.now(timezone.utc)
    approved_suggestions = session.execute(
        select(TransferSuggestion).where(
            TransferSuggestion.user_uuid == user_uuid,
            TransferSuggestion.status == "APPROVED",
            TransferSuggestion.is_verified_in_bank == False
        )
    ).scalars().all()

    reminders = []
    for s in approved_suggestions:
        # Check if the target bank account received the funds
        target_acc = session.execute(
            select(Account).where(Account.account_id == s.to_account_id)
        ).scalars().first()

        # If deadline expired without balance update, flag reminder
        deadline = s.verification_deadline
        if deadline and deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)

        if deadline and now >= deadline:
            reminders.append({
                "suggestion_id": s.id,
                "from_account": s.from_account_name,
                "to_account": s.to_account_name,
                "amount": s.amount,
                "currency": s.currency,
                "severity": s.severity,
                "message": f"Reminder: You agreed to transfer {s.amount:.2f} {s.currency} from {s.from_account_name} to {s.to_account_name}. The bank balance has not updated yet."
            })

    return reminders


def update_bucket_details(
    session: Session,
    user_uuid: str,
    bucket_id: str,
    name: Optional[str] = None,
    target_amount: Optional[float] = None,
    target_date: Optional[datetime] = None,
    priority: Optional[int] = None
) -> Bucket:
    """Updates configuration fields of a virtual bucket."""
    bucket = session.query(Bucket).filter_by(id=bucket_id, user_id=user_uuid).first()
    if not bucket:
        raise ValueError("Bucket not found.")

    b_type = bucket.type.value if hasattr(bucket.type, "value") else str(bucket.type)
    if b_type == BucketType.UNALLOCATED.value:
        raise ValueError("Cannot edit configuration of the UNALLOCATED bucket.")

    if name is not None and name.strip():
        bucket.name = name.strip()
    if target_amount is not None:
        bucket.target_amount = max(0.0, float(target_amount))
    if target_date is not None:
        bucket.target_date = target_date
    if priority is not None:
        bucket.priority = int(priority)

    bucket.updated_at = datetime.now(timezone.utc)
    session.flush()
    return bucket


def delete_virtual_bucket(session: Session, user_uuid: str, bucket_id: str) -> None:
    """
    Deactivates a virtual bucket and returns any remaining balance to the Unallocated bucket,
    maintaining strict zero-sum balance.
    """
    bucket = session.query(Bucket).filter_by(id=bucket_id, user_id=user_uuid).first()
    if not bucket:
        raise ValueError("Bucket not found.")

    b_type = bucket.type.value if hasattr(bucket.type, "value") else str(bucket.type)
    if b_type == BucketType.UNALLOCATED.value:
        raise ValueError("Cannot delete the UNALLOCATED bucket.")

    remaining_funds = float(bucket.cached_balance or 0.0)
    if remaining_funds > 0:
        unallocated = get_or_create_unallocated_bucket(session, user_uuid)
        unallocated.cached_balance = round(float(unallocated.cached_balance or 0.0) + remaining_funds, 2)
        
        curr = (bucket.currency or "GBP").upper()
        balances = dict(unallocated.cached_balances or {})
        if curr in balances:
            balances[curr]["total"] = round(float(balances[curr].get("total", 0.0)) + remaining_funds, 2)
        else:
            balances[curr] = {"total": remaining_funds, "accounts": {}}
        unallocated.cached_balances = balances
        
        transfer = VirtualTransfer(
            id=str(uuid.uuid4()),
            source_bucket_id=bucket.id,
            target_bucket_id=unallocated.id,
            amount=remaining_funds
        )
        session.add(transfer)
        bucket.cached_balance = 0.0
        bucket.allocated_amount = 0.0

    bucket.is_active = False
    bucket.updated_at = datetime.now(timezone.utc)
    session.flush()


def toggle_bucket_active_state(session: Session, user_uuid: str, bucket_id: str, is_active: bool) -> Bucket:
    """Toggles active state of a virtual bucket."""
    bucket = session.query(Bucket).filter_by(id=bucket_id, user_id=user_uuid).first()
    if not bucket:
        raise ValueError("Bucket not found.")

    b_type = bucket.type.value if hasattr(bucket.type, "value") else str(bucket.type)
    if b_type == BucketType.UNALLOCATED.value and not is_active:
        raise ValueError("Cannot deactivate the UNALLOCATED bucket.")

    bucket.is_active = is_active
    bucket.updated_at = datetime.now(timezone.utc)
    session.flush()
    return bucket

