from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel
from middleware.auth_middleware import get_current_user
from models.database_models import User
from controllers.alert_controller import (
    get_notifications,
    mark_alert_read,
    dismiss_alert,
    mark_all_alerts_read,
    get_pending_suggestions,
    respond_to_suggestion
)
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


class SuggestionResponseRequest(BaseModel):
    approved: bool
    reason: Optional[str] = None


@router.get("")
async def get_notifications_route(
    limit: int = Query(default=20, ge=1, le=100),
    include_dismissed: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Fetches notification alerts for the user."""
    return get_notifications(
        user_uuid=current_user.user_uuid,
        limit=limit,
        include_dismissed=include_dismissed
    )


@router.patch("/{alert_id}/read", status_code=status.HTTP_200_OK)
async def mark_alert_read_route(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """Marks a single alert as read."""
    return mark_alert_read(user_uuid=current_user.user_uuid, alert_id=alert_id)


@router.post("/{alert_id}/dismiss", status_code=status.HTTP_200_OK)
async def dismiss_alert_route(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """Dismisses an alert."""
    return dismiss_alert(user_uuid=current_user.user_uuid, alert_id=alert_id)


@router.post("/read-all", status_code=status.HTTP_200_OK)
async def mark_all_alerts_read_route(
    current_user: User = Depends(get_current_user)
):
    """Marks all alerts as read."""
    return mark_all_alerts_read(user_uuid=current_user.user_uuid)


@router.get("/suggestions")
async def get_transfer_suggestions_route(
    current_user: User = Depends(get_current_user)
):
    """Lists pending Stage 4 physical bank transfer suggestions."""
    return get_pending_suggestions(user_uuid=current_user.user_uuid)


@router.post("/suggestions/{suggestion_id}/respond", status_code=status.HTTP_200_OK)
async def respond_to_suggestion_route(
    suggestion_id: str,
    request: SuggestionResponseRequest,
    current_user: User = Depends(get_current_user)
):
    """Accepts or rejects a proposed physical bank transfer suggestion."""
    return respond_to_suggestion(
        user_uuid=current_user.user_uuid,
        suggestion_id=suggestion_id,
        approved=request.approved,
        reason=request.reason
    )
