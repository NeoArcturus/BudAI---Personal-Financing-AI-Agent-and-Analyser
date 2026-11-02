from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from config import SessionLocal
from services.notifications.alert_dispatcher import get_alert_dispatcher
from models.status_codes import PipelineStatus
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class CashflowCollisionEngine:
    """
    Cashflow Deficit & Liability Collision Engine.
    Intersects upcoming 7-to-14-day bill and subscription commitments against
    the available liquid balance in DEFAULT and RECURRING buckets.
    Triggers critical overdraft protection alerts if a collision is detected.
    """

    def __init__(self, forecast_window_days: int = 7):
        self.window_days = forecast_window_days
        self.dispatcher = get_alert_dispatcher()

    def evaluate_liability_collision(self, user_uuid: str) -> Dict[str, Any]:
        """
        Scans upcoming recurring liabilities and compares with current liquid bucket capital.
        """
        now = datetime.utcnow()
        horizon_end = now + timedelta(days=self.window_days)

        with SessionLocal() as session:
            # 1. Fetch current liquid bucket balances
            bucket_query = text("""
                SELECT type, cached_balance
                FROM buckets
                WHERE user_id = :user_uuid;
            """)
            buckets = session.execute(bucket_query, {"user_uuid": user_uuid}).fetchall()

            default_balance = 0.0
            recurring_balance = 0.0
            for b in buckets:
                b_type = str(r_type := b[0]).upper()
                bal = float(b[1]) if b[1] else 0.0
                if b_type == "DEFAULT":
                    default_balance = bal
                elif b_type == "RECURRING":
                    recurring_balance = bal

            total_liquid = default_balance + recurring_balance

            # 2. Fetch detected subscriptions/liabilities due within the window
            sub_query = text("""
                SELECT merchant_name, 
                       COALESCE(expected_amount, last_payment_amount, 0.0) as amount,
                       predicted_frequency, next_expected_date
                FROM subscriptions
                WHERE user_uuid = :user_uuid
                  AND status != 'cancelled'
                  AND next_expected_date IS NOT NULL
                  AND next_expected_date <= :horizon_end;
            """)
            subs = session.execute(sub_query, {
                "user_uuid": user_uuid,
                "horizon_end": horizon_end
            }).fetchall()

            upcoming_bills = []
            total_due = 0.0
            for s in subs:
                name = s[0]
                amt = float(s[1]) if s[1] else 0.0
                freq = s[2] or "monthly"
                due_date = s[3]
                total_due += amt
                upcoming_bills.append({
                    "merchant": name,
                    "amount": round(amt, 2),
                    "due_date": due_date.strftime("%Y-%m-%d") if hasattr(due_date, "strftime") else str(due_date),
                    "frequency": freq
                })

            deficit = total_due - total_liquid

            logger.info(
                f"Collision Check for {user_uuid}: Liquid=£{total_liquid:.2f} "
                f"(Default=£{default_balance:.2f}, Recurring=£{recurring_balance:.2f}), "
                f"Due in {self.window_days}d=£{total_due:.2f}, Deficit=£{deficit:.2f}"
            )

            if deficit > 0.0 and len(upcoming_bills) > 0:
                bill_names = ", ".join([b["merchant"] for b in upcoming_bills[:3]])
                message = (
                    f"Cashflow Collision Warning: £{total_due:.2f} in liabilities due in the next "
                    f"{self.window_days} days ({bill_names}), but your liquid balance is £{total_liquid:.2f}. "
                    f"Projected shortfall: £{deficit:.2f}."
                )

                self.dispatcher.dispatch(
                    user_uuid=user_uuid,
                    title="Upcoming Bill Overdraft Collision",
                    message=message,
                    event_type="CASHFLOW_COLLISION",
                    urgency=5,  # Critical Tier 5 - never throttled
                    status_code=PipelineStatus.CASHFLOW_COLLISION_DEFICIT.value,
                    action_type="VIEW_BUCKETS",
                    action_payload={
                        "total_due": round(total_due, 2),
                        "total_liquid": round(total_liquid, 2),
                        "deficit": round(deficit, 2),
                        "window_days": self.window_days,
                        "bills": upcoming_bills
                    }
                )

                return {
                    "status": "collision_detected",
                    "total_liquid": round(total_liquid, 2),
                    "total_due": round(total_due, 2),
                    "deficit": round(deficit, 2),
                    "bills_count": len(upcoming_bills),
                    "alert_fired": True
                }

            return {
                "status": "solvent",
                "total_liquid": round(total_liquid, 2),
                "total_due": round(total_due, 2),
                "deficit": 0.0,
                "bills_count": len(upcoming_bills),
                "alert_fired": False
            }

_collision_engine_instance: Optional[CashflowCollisionEngine] = None

def get_cashflow_collision_engine() -> CashflowCollisionEngine:
    global _collision_engine_instance
    if _collision_engine_instance is None:
        _collision_engine_instance = CashflowCollisionEngine()
    return _collision_engine_instance
