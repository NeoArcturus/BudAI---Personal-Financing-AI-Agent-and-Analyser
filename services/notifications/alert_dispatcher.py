import json
import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
from config import SessionLocal, redis_client
from models.database_models import SystemAlert
from models.status_codes import PipelineStatus
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class AlertDispatcher:
    """
    Central notification dispatcher providing:
    - Persistent storage in Neon PostgreSQL
    - Real-time WebSocket event broadcasting over Upstash Redis Pub/Sub
    - Anti-spam sliding window rate limiting (max 2 non-critical alerts per 24 hours)
    - 48-hour fingerprint deduplication
    """

    def __init__(self):
        self.redis = redis_client

    def _should_throttle(self, user_uuid: str, urgency: int, fingerprint: str) -> bool:
        """
        Anti-Spam Guardrail:
        - Critical (urgency >= 5) is never throttled.
        - Identical fingerprints are muted for 48 hours.
        - Non-critical alerts are throttled to 2 per 24 hours.
        """
        if urgency >= 5:
            return False

        if not self.redis:
            return False

        try:
            # 1. Deduplication by fingerprint
            fp_key = f"alert_fp:{user_uuid}:{fingerprint}"
            if self.redis.get(fp_key):
                logger.info(f"Alert throttled: Duplicate fingerprint for user {user_uuid}")
                return True

            # 2. Daily volume ceiling (Max 2 non-critical alerts per day)
            today = datetime.utcnow().strftime('%Y-%m-%d')
            daily_count_key = f"alert_count:{user_uuid}:{today}"
            current_count = self.redis.incr(daily_count_key)
            if current_count == 1:
                self.redis.expire(daily_count_key, 86400)

            if current_count > 2:
                logger.info(f"Alert throttled: User {user_uuid} reached daily quota ({current_count}/2)")
                return True

            # Mark fingerprint as seen for 48 hours
            self.redis.setex(fp_key, 172800, "1")
            return False
        except Exception as e:
            logger.warning(f"Redis throttling check failed (allowing alert): {e}")
            return False

    def dispatch(
        self,
        user_uuid: str,
        message: str,
        event_type: str,
        title: Optional[str] = None,
        urgency: int = 2,
        action_type: Optional[str] = None,
        action_payload: Optional[Dict[str, Any]] = None,
        status_code: Optional[str] = None
    ) -> Optional[SystemAlert]:
        """
        Dispatches an alert through DB persistence and real-time WebSocket Pub/Sub.
        """
        fingerprint = hashlib.md5(f"{event_type}:{title}:{message[:50]}".encode()).hexdigest()

        if self._should_throttle(user_uuid, urgency, fingerprint):
            return None

        alert_id = str(uuid.uuid4())
        effective_title = title or event_type.replace("_", " ").title()

        # Map to custom BudAI compound status codes if not explicitly provided
        effective_status_code = status_code
        if not effective_status_code:
            if event_type == "SURGE_WARNING":
                effective_status_code = PipelineStatus.SPEND_SURGE_DETECTED.value
            elif event_type == "CASHFLOW_DEFICIT":
                effective_status_code = PipelineStatus.CASHFLOW_COLLISION_DEFICIT.value
            elif event_type == "SUBSCRIPTION_PRICE_HIKE":
                effective_status_code = PipelineStatus.SUBSCRIPTION_PRICE_HIKE.value
            elif event_type == "BUDGET_OVERRUN":
                effective_status_code = PipelineStatus.BUDGET_OVERRUN.value
            else:
                effective_status_code = PipelineStatus.ALERT_DISPATCHED.value

        now_ts = datetime.utcnow()
        alert = SystemAlert(
            id=alert_id,
            user_id=user_uuid,
            title=effective_title,
            message=message,
            event_type=event_type,
            urgency_level=urgency,
            status_code=effective_status_code,
            action_type=action_type,
            action_payload=action_payload,
            is_read=False,
            is_dismissed=False,
            timestamp=now_ts
        )

        # 1. Persist to Neon PostgreSQL
        try:
            with SessionLocal() as session:
                session.add(alert)
                session.commit()
                session.refresh(alert)
                session.expunge(alert)
        except Exception as e:
            logger.error(f"Failed to persist SystemAlert to database: {e}")
            return None

        # 2. Broadcast via Redis Pub/Sub to active WebSocket connections
        if self.redis:
            try:
                ws_payload = {
                    "event_type": "NEW_ALERT",
                    "alert": {
                        "id": alert_id,
                        "title": effective_title,
                        "message": message,
                        "event_type": event_type,
                        "urgency": urgency,
                        "status_code": effective_status_code,
                        "action_type": action_type,
                        "action_payload": action_payload,
                        "timestamp": now_ts.isoformat()
                    }
                }
                self.redis.publish(f"budai_events_{user_uuid}", json.dumps(ws_payload))
            except Exception as e:
                logger.error(f"Failed to publish notification to Redis PubSub: {e}")

        logger.info(f"Dispatched SystemAlert [{alert_id}] ({effective_status_code}) to user {user_uuid} (Urgency={urgency})")
        return alert

_dispatcher_instance: Optional[AlertDispatcher] = None

def get_alert_dispatcher() -> AlertDispatcher:
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = AlertDispatcher()
    return _dispatcher_instance
