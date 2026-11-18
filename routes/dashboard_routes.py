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
    days: Optional[int] = Query(default=30, ge=0, le=3650, description="Lookback window in days: 30, 60, 90, 180, or 0 for all-time"),
    time_frame: Optional[str] = Query(default=None, description="Timeframe alias: '30d', '60d', '90d', '180d', 'all'"),
    page: int = Query(default=1, ge=1, description="Page number for paginated transactions"),
    limit: int = Query(default=50, ge=1, le=250, description="Page size for transactions"),
    expense_only: bool = Query(default=False, description="Filter transactions to expenses/debits only"),
    currency: Optional[str] = Query(None, description="Preferred display currency e.g. GBP, USD, EUR"),
    limit_transactions: Optional[int] = Query(None, description="Deprecated alias for limit"),
    limit_notifications: int = Query(default=15, ge=1, le=50),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Consolidated Single-Roundtrip Snapshot API.
    Synthesizes physical bank cash, virtual buckets, Stage 3 intelligence,
    lifestyle topology, paginated transactions, daily expense points, and pending suggestions.
    """
    effective_limit = limit if limit_transactions is None else limit_transactions
    return get_dashboard_snapshot(
        user_uuid=current_user.user_uuid,
        days=days,
        time_frame=time_frame,
        page=page,
        limit=effective_limit,
        expense_only=expense_only,
        currency=currency
    )
