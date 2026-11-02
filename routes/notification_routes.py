from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import text
from config import SessionLocal
from models.database_models import User
from middleware.auth_middleware import get_current_user
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])

class AlertResponse(BaseModel):
    id: str
    title: Optional[str] = None
    message: str
    event_type: str
    urgency_level: int
    status_code: Optional[str] = None
    action_type: Optional[str] = None
    action_payload: Optional[Dict[str, Any]] = None
    is_read: bool
    is_dismissed: bool
    timestamp: datetime

@router.get("", response_model=List[AlertResponse])
async def get_notifications(
    limit: int = Query(default=20, ge=1, le=100),
    include_dismissed: bool = False,
    current_user: User = Depends(get_current_user)
):
    """
    Fetches notification alerts for the authenticated user ordered by timestamp desc.
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
                "user_id": current_user.user_uuid,
                "limit": limit
            }).fetchall()

            alerts = []
            for r in rows:
                alerts.append(AlertResponse(
                    id=str(r[0]),
                    title=r[1],
                    message=r[2],
                    event_type=r[3],
                    urgency_level=int(r[4]),
                    status_code=str(r[5]),
                    action_type=r[6],
                    action_payload=r[7] if isinstance(r[7], dict) else None,
                    is_read=bool(r[8]),
                    is_dismissed=bool(r[9]),
                    timestamp=r[10]
                ))
            return alerts
    except Exception as e:
        logger.error(f"Failed to fetch notifications for user {current_user.user_uuid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load notifications")

@router.patch("/{alert_id}/read", status_code=status.HTTP_200_OK)
async def mark_alert_read(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Marks a single notification as read.
    """
    try:
        with SessionLocal() as session:
            res = session.execute(
                text("""
                    UPDATE system_alerts 
                    SET is_read = TRUE 
                    WHERE id = :id AND user_id = :user_id
                """),
                {"id": alert_id, "user_id": current_user.user_uuid}
            )
            session.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Alert not found")
            return {"status": "success", "message": "Alert marked as read"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark alert {alert_id} as read: {e}")
        raise HTTPException(status_code=500, detail="Database update failed")

@router.post("/{alert_id}/dismiss", status_code=status.HTTP_200_OK)
async def dismiss_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Dismisses a notification from the primary drawer view.
    """
    try:
        with SessionLocal() as session:
            res = session.execute(
                text("""
                    UPDATE system_alerts 
                    SET is_dismissed = TRUE, is_read = TRUE
                    WHERE id = :id AND user_id = :user_id
                """),
                {"id": alert_id, "user_id": current_user.user_uuid}
            )
            session.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Alert not found")
            return {"status": "success", "message": "Alert dismissed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to dismiss alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Database update failed")

@router.post("/read-all", status_code=status.HTTP_200_OK)
async def mark_all_alerts_read(
    current_user: User = Depends(get_current_user)
):
    """
    Marks all notifications for the authenticated user as read.
    """
    try:
        with SessionLocal() as session:
            session.execute(
                text("""
                    UPDATE system_alerts 
                    SET is_read = TRUE 
                    WHERE user_id = :user_id AND is_read IS FALSE
                """),
                {"user_id": current_user.user_uuid}
            )
            session.commit()
            return {"status": "success", "message": "All alerts marked as read"}
    except Exception as e:
        logger.error(f"Failed to mark all alerts as read for {current_user.user_uuid}: {e}")
        raise HTTPException(status_code=500, detail="Database update failed")
