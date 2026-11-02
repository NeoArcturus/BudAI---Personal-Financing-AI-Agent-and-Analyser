import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy import text
from config import SessionLocal
from services.notifications.alert_dispatcher import get_alert_dispatcher
from services.analytics.ons_client import get_ons_client
from models.status_codes import PipelineStatus
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class SurgeDetector:
    """
    Statistical Daily Spend Surge & Velocity Engine.
    Maintains a 30-day rolling daily spend baseline (mean and std-dev) per user.
    Fires high-signal SURGE_WARNING alerts when today's spending breaches Z >= 2.0σ.
    """

    def __init__(self, z_threshold: float = 2.0):
        self.z_threshold = z_threshold
        self.dispatcher = get_alert_dispatcher()

    def check_daily_surge(self, user_uuid: str) -> Dict[str, Any]:
        """
        Calculates today's cumulative spending against the 30-day baseline.
        If Z >= threshold, dispatches a SystemAlert.
        """
        now = datetime.utcnow()
        today_start = datetime(now.year, now.month, now.day)
        thirty_days_ago = today_start - timedelta(days=30)

        with SessionLocal() as session:
            # 1. Fetch daily totals for the last 30 days (excluding today)
            baseline_query = text("""
                SELECT DATE(date) as day, SUM(ABS(amount)) as total
                FROM transactions
                WHERE user_uuid = :user_uuid
                  AND amount < 0
                  AND date >= :start_date
                  AND date < :today_start
                GROUP BY DATE(date);
            """)
            rows = session.execute(baseline_query, {
                "user_uuid": user_uuid,
                "start_date": thirty_days_ago,
                "today_start": today_start
            }).fetchall()

            daily_totals = [float(r[1]) for r in rows] if rows else []

            # 2. Fetch today's accumulated spend and top spending driver
            today_query = text("""
                SELECT SUM(ABS(amount)) as today_total,
                       COUNT(transaction_uuid) as tx_count
                FROM transactions
                WHERE user_uuid = :user_uuid
                  AND amount < 0
                  AND date >= :today_start;
            """)
            today_res = session.execute(today_query, {
                "user_uuid": user_uuid,
                "today_start": today_start
            }).fetchone()

            today_spend = float(today_res[0]) if today_res and today_res[0] else 0.0
            today_tx_count = int(today_res[1]) if today_res and today_res[1] else 0

            if today_spend <= 0.0 or len(daily_totals) < 5:
                # Insufficient baseline history or no spend today
                return {
                    "status": "normal",
                    "today_spend": today_spend,
                    "z_score": 0.0,
                    "alert_fired": False
                }

            # 3. Compute baseline statistics
            mean_spend = sum(daily_totals) / len(daily_totals)
            variance = sum((x - mean_spend) ** 2 for x in daily_totals) / len(daily_totals)
            std_dev = math.sqrt(variance)

            # Avoid division by zero
            if std_dev < 1.0:
                std_dev = max(mean_spend * 0.2, 5.0)

            z_score = (today_spend - mean_spend) / std_dev

            logger.info(
                f"Surge Check for {user_uuid}: Today=£{today_spend:.2f}, "
                f"Mean=£{mean_spend:.2f}, Std=£{std_dev:.2f}, Z={z_score:.2f}"
            )

            if z_score >= self.z_threshold:
                # Identify the single largest merchant driving today's surge
                top_merchant_query = text("""
                    SELECT COALESCE(mk.clean_merchant_name, t.semi_cleaned_description, 'Unknown Merchant') as merchant,
                           ABS(t.amount) as amount,
                           COALESCE(mk.category, 'General') as category,
                           t.tags
                    FROM transactions t
                    LEFT JOIN merchant_knowledge mk ON t.merchant_knowledge_uuid = mk.knowledge_uuid
                    WHERE t.user_uuid = :user_uuid
                      AND t.amount < 0
                      AND t.date >= :today_start
                    ORDER BY ABS(t.amount) DESC
                    LIMIT 1;
                """)
                top_m = session.execute(top_merchant_query, {
                    "user_uuid": user_uuid,
                    "today_start": today_start
                }).fetchone()

                top_merchant_name = top_m[0] if top_m else "Recent Card Transactions"
                top_merchant_amt = float(top_m[1]) if top_m else 0.0
                top_merchant_cat = top_m[2] if top_m else "General"

                # Check ONS macroeconomic inflation context
                ons_inflation = get_ons_client().get_category_inflation(top_merchant_cat)

                multiplier = today_spend / mean_spend if mean_spend > 0 else 2.0
                macro_note = f" (ONS {top_merchant_cat} inflation: +{ons_inflation:.1f}% YoY)" if ons_inflation > 0 else ""
                message = (
                    f"Today's spending (£{today_spend:.2f}) is {multiplier:.1f}x above your daily baseline "
                    f"(£{mean_spend:.2f}). Largest outflow: £{top_merchant_amt:.2f} at {top_merchant_name}{macro_note}."
                )

                self.dispatcher.dispatch(
                    user_uuid=user_uuid,
                    title="Daily Expense Surge Detected",
                    message=message,
                    event_type="SURGE_WARNING",
                    urgency=4,
                    status_code=PipelineStatus.SPEND_SURGE_DETECTED.value,
                    action_type="VIEW_TRANSACTIONS",
                    action_payload={
                        "today_spend": today_spend,
                        "mean_spend": mean_spend,
                        "z_score": round(z_score, 2),
                        "top_merchant": top_merchant_name,
                        "category": top_merchant_cat,
                        "macro_inflation_yoy": ons_inflation
                    }
                )

                return {
                    "status": "surge_detected",
                    "today_spend": today_spend,
                    "mean_spend": mean_spend,
                    "z_score": round(z_score, 2),
                    "alert_fired": True
                }

            return {
                "status": "normal",
                "today_spend": today_spend,
                "mean_spend": mean_spend,
                "z_score": round(z_score, 2),
                "alert_fired": False
            }

_surge_detector_instance: Optional[SurgeDetector] = None

def get_surge_detector() -> SurgeDetector:
    global _surge_detector_instance
    if _surge_detector_instance is None:
        _surge_detector_instance = SurgeDetector()
    return _surge_detector_instance
