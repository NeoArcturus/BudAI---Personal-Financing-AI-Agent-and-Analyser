from fastapi import APIRouter, Depends, Query, BackgroundTasks, status
from fastapi_cache.decorator import cache
from middleware.auth_middleware import get_current_user
from models.database_models import User
from schemas.api_schema import TransactionLabelCorrectionRequest
from utils.cache_utils import user_cache_key_builder
from services.logger_setup import get_core_logger

from controllers.categorizer_controller import get_review_candidates, save_manual_label

logger = get_core_logger(__name__)
router = APIRouter(prefix="/api/categorizer", tags=["categorizer"])


@router.get("/review-candidates")
@cache(expire=300, namespace="categorizer", key_builder=user_cache_key_builder)
async def get_review_candidates_route(
    account_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    current_user: User = Depends(get_current_user)
):
    return await get_review_candidates(account_id, limit, current_user)


@router.post("/labels", status_code=status.HTTP_202_ACCEPTED)
async def save_manual_label_route(
    payload: TransactionLabelCorrectionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    return await save_manual_label(payload, background_tasks, current_user)
