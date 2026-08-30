import json
from datetime import datetime, timedelta
from config import SessionLocal
from models.database_models import Subscription
from sqlalchemy import text
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def expire_stale_subscriptions():
    """
    Runs in the background (e.g. nightly) to check all active subscriptions.
    If the next_expected_date is more than 100 days in the past, it marks 
    the subscription status as 'expired'.
    """
    logger.info(json.dumps({"message": f"Starting stale subscription sweeper job.", "status_code": 200}))
    try:
        threshold_date = datetime.utcnow() - timedelta(days=100)
        
        with SessionLocal() as session:
            result = session.execute(
                text("""
                    UPDATE subscriptions 
                    SET status = 'expired', last_updated = :now 
                    WHERE status = 'active' AND next_expected_date < :threshold
                """),
                {"now": datetime.utcnow(), "threshold": threshold_date}
            )
            count = result.rowcount
            session.commit()
            logger.info(json.dumps({"message": f"Subscription sweeper completed. Marked {count} subscriptions as expired.", "status_code": 200}))
    except Exception as e:
        logger.error(json.dumps({"message": f"Error sweeping stale subscriptions: {e}", "status_code": 500}))
