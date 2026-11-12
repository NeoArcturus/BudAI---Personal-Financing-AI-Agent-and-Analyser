from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, Query
from middleware.auth_middleware import get_current_user
from models.database_models import User
from controllers.dashboard_controller import get_dashboard_snapshot
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

router = APIRouter(tags=["Dashboard"])


@router.get("/api/dashboard/snapshot")
@router.get("/api/snapshot")
def get_dashboard_snapshot_route(
    currency: Optional[str] = Query(None, description="Preferred display currency e.g. GBP, USD, EUR"),
    limit_transactions: int = Query(default=30, ge=1, le=100),
    limit_notifications: int = Query(default=15, ge=1, le=50),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Consolidated Single-Roundtrip Snapshot API.
    Synthesizes physical bank cash, virtual buckets, Stage 3 intelligence,
    lifestyle topology, and pending transfer suggestions in one single database pass.
    """
    return get_dashboard_snapshot(user_uuid=current_user.user_uuid)
