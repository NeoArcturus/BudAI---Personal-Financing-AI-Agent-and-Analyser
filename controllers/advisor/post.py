import uuid
from typing import Any
from fastapi import BackgroundTasks
from pydantic import BaseModel
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class SummarizeRequest(BaseModel):
    widget_id: str
    context_data: Any

async def summarize_data(request: SummarizeRequest, background_tasks: BackgroundTasks, current_user: Any):
    return {"status": "deprecated", "message": "Advisor summary endpoint is deprecated in favor of Agentic Generative UI"}
