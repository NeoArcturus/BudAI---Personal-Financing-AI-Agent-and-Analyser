from fastapi import APIRouter, Depends, Query, BackgroundTasks, status
from fastapi_cache.decorator import cache
from middleware.auth_middleware import get_current_user
from models.database_models import User
from schemas.api_schema import TransactionLabelCorrectionRequest, RetrainCategorizerRequest
from utils.cache_utils import user_cache_key_builder
from services.logger_setup import get_core_logger

from controllers.categorizer.get import get_task_status, get_review_candidates
from controllers.categorizer.post import save_manual_label, retrain_categorizer

logger = get_core_logger(__name__)
router = APIRouter(prefix="/api/categorizer", tags=["categorizer"])

@router.get("/task-status/{task_id}")
async def get_task_status_route(task_id: str, current_user: User = Depends(get_current_user)):
    return await get_task_status(task_id, current_user)

@router.get("/review-candidates")
@cache(expire=300, namespace="categorizer", key_builder=user_cache_key_builder)
async def get_review_candidates_route(
    account_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    current_user: User = Depends(get_current_user)
):
    return await get_review_candidates(account_id, limit, current_user)

@router.post("/labels", status_code=status.HTTP_202_ACCEPTED)
async def save_manual_label_route(payload: TransactionLabelCorrectionRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    return await save_manual_label(payload, background_tasks, current_user)

@router.post("/retrain", status_code=status.HTTP_202_ACCEPTED)
async def retrain_categorizer_route(payload: RetrainCategorizerRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    return await retrain_categorizer(payload, background_tasks, current_user)
