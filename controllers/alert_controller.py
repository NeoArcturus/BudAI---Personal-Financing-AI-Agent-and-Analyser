import json
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from fastapi import HTTPException
from sqlalchemy import text
from sqlmodel import select
from config import SessionLocal, redis_client
from models.database_models import SystemAlert, TransferSuggestion
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


def get_notifications(user_uuid: str, limit: int = 20, include_dismissed: bool = False) -> List[Dict[str, Any]]:
    """
    Fetches notifications and alerts for the user.
    """
    try:
        with SessionLocal() as session:
            filter_clause = "WHERE user_id = :user_id"
            if not include_dismissed:
                filter_clause += " AND (is_dismissed IS FALSE OR is_dismissed IS NULL)"

            query = text(f"""
                SELECT id, title, message, event_type, 
                       COALESCE(urgency_level, 1) as urgency_level,
                       COALESCE(status_code, '308-200') as status_code,
                       action_type, action_payload,
                       COALESCE(is_read, FALSE) as is_read,
                       COALESCE(is_dismissed, FALSE) as is_dismissed,
                       timestamp
                FROM system_alerts
                {filter_clause}
                ORDER BY timestamp DESC
                LIMIT :limit;
            """)

            rows = session.execute(query, {
                "user_id": user_uuid,
                "limit": limit
            }).fetchall()

            alerts = []
            for r in rows:
                alerts.append({
                    "id": str(r[0]),
                    "title": r[1],
                    "message": r[2],
                    "event_type": r[3],
                    "urgency_level": int(r[4]),
                    "status_code": str(r[5]),
                    "action_type": r[6],
                    "action_payload": r[7] if isinstance(r[7], dict) else None,
                    "is_read": bool(r[8]),
                    "is_dismissed": bool(r[9]),
                    "timestamp": r[10].isoformat() if r[10] else None
                })
            return alerts
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Failed to fetch notifications for user {user_uuid}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Failed to load notifications")


def mark_alert_read(user_uuid: str, alert_id: str) -> Dict[str, Any]:
    """Marks a single alert as read."""
    try:
        with SessionLocal() as session:
            res = session.execute(
                text("""
                    UPDATE system_alerts 
                    SET is_read = TRUE 
                    WHERE id = :id AND user_id = :user_id
                """),
                {"id": alert_id, "user_id": user_uuid}
            )
            session.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Alert not found")
            return {"status": "success", "message": "Alert marked as read"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Failed to mark alert {alert_id} as read: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Database update failed")


def dismiss_alert(user_uuid: str, alert_id: str) -> Dict[str, Any]:
    """Dismisses an alert."""
    try:
        with SessionLocal() as session:
            res = session.execute(
                text("""
                    UPDATE system_alerts 
                    SET is_dismissed = TRUE, is_read = TRUE
                    WHERE id = :id AND user_id = :user_id
                """),
                {"id": alert_id, "user_id": user_uuid}
            )
            session.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Alert not found")
            return {"status": "success", "message": "Alert dismissed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Failed to dismiss alert {alert_id}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Database update failed")


def mark_all_alerts_read(user_uuid: str) -> Dict[str, Any]:
    """Marks all user alerts as read."""
    try:
        with SessionLocal() as session:
            session.execute(
                text("""
                    UPDATE system_alerts 
                    SET is_read = TRUE 
                    WHERE user_id = :user_id AND (is_read IS FALSE OR is_read IS NULL)
                """),
                {"user_id": user_uuid}
            )
            session.commit()
            return {"status": "success", "message": "All alerts marked as read"}
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Failed to mark all alerts as read for user {user_uuid}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Database update failed")


def get_pending_suggestions(user_uuid: str) -> List[Dict[str, Any]]:
    """Retrieves pending transfer suggestions for physical bank accounts."""
    try:
        with SessionLocal() as session:
            suggestions = session.execute(
                select(TransferSuggestion).where(
                    TransferSuggestion.user_uuid == user_uuid,
                    TransferSuggestion.status == "PENDING"
                )
            ).scalars().all()

            results = []
            for s in suggestions:
                results.append({
                    "id": s.id,
                    "bill_id": s.bill_id,
                    "from_account_id": s.from_account_id,
                    "from_account_name": s.from_account_name,
                    "to_account_id": s.to_account_id,
                    "to_account_name": s.to_account_name,
                    "amount": float(s.amount),
                    "currency": s.currency,
                    "status": s.status,
                    "severity": s.severity,
                    "verification_deadline": s.verification_deadline.isoformat() if s.verification_deadline else None,
                    "created_at": s.created_at.isoformat() if s.created_at else None
                })
            return results
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Failed to get suggestions for user {user_uuid}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Failed to load transfer suggestions")


def respond_to_suggestion(user_uuid: str, suggestion_id: str, approved: bool, reason: Optional[str] = None) -> Dict[str, Any]:
    """
    Handles user response to a physical bank transfer suggestion.
    """
    try:
        with SessionLocal() as session:
            sugg = session.execute(
                select(TransferSuggestion).where(
                    TransferSuggestion.id == suggestion_id,
                    TransferSuggestion.user_uuid == user_uuid
                )
            ).scalars().first()

            if not sugg:
                raise HTTPException(status_code=404, detail="Transfer suggestion not found")

            if sugg.status != "PENDING":
                raise HTTPException(status_code=400, detail=f"Suggestion is already {sugg.status}")

            if approved:
                sugg.status = "APPROVED"
                sugg.approved_at = datetime.now(timezone.utc)
                message = f"Transfer approved. Please move £{sugg.amount:.2f} {sugg.currency} from {sugg.from_account_name} to {sugg.to_account_name} in your banking app."
            else:
                sugg.status = "REJECTED"
                sugg.rejection_reason = reason or "User rejected transfer"
                message = "Transfer suggestion rejected."

            session.commit()

            try:
                redis_client.publish(f"budai_events_{user_uuid}", json.dumps({
                    "event_type": "TRANSFER_SUGGESTION_RESOLVED",
                    "suggestion_id": suggestion_id,
                    "status": sugg.status
                }))
            except Exception:
                pass

            return {"status": "success", "message": message, "new_status": sugg.status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Failed to respond to suggestion {suggestion_id}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Failed to process suggestion response")
