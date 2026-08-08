from fastapi import APIRouter, Depends, BackgroundTasks
from middleware.auth_middleware import get_current_user
from models.database_models import User
from controllers.advisor.get import get_summarize_status
from controllers.advisor.post import summarize_data, SummarizeRequest

router = APIRouter(prefix="/api/advisor", tags=["advisor"])

@router.post("/summarize")
async def summarize_data_route(request: SummarizeRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    return await summarize_data(request, background_tasks, current_user)

@router.get("/status/{job_id}")
async def get_summarize_status_route(job_id: str):
    return await get_summarize_status(job_id)
