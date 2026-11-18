import json
from typing import Dict, Any
from fastapi import HTTPException
from config import SessionLocal
from services.dashboard_service import DashboardService
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

from typing import Dict, Any, Optional

dashboard_service_instance = DashboardService()

def get_dashboard_snapshot(
    user_uuid: str,
    days: Optional[int] = 30,
    time_frame: Optional[str] = None,
    page: int = 1,
    limit: int = 50,
    expense_only: bool = False,
    currency: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieves the complete consolidated financial snapshot for the user with paginated transactions and expense points.
    """
    try:
        with SessionLocal() as session:
            snapshot = dashboard_service_instance.get_full_snapshot(
                session=session,
                user_uuid=user_uuid,
                days=days,
                time_frame=time_frame,
                page=page,
                limit=limit,
                expense_only=expense_only,
                currency=currency
            )
            return snapshot
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Failed to retrieve dashboard snapshot for user {user_uuid}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Failed to synthesize dashboard snapshot.")
