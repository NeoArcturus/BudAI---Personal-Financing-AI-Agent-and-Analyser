from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uuid
import time
from middleware.auth_middleware import get_current_user
from models.database_models import User
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from services.profile_builder import ProfileBuilder
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

advisor_router = APIRouter(prefix="/api/advisor", tags=["advisor"])

job_store: Dict[str, Dict[str, Any]] = {}
mrfp_cache: Dict[str, Dict[str, Any]] = {}


class SummarizeRequest(BaseModel):
    widget_id: str
    context_data: Any


async def run_advisor_task(job_id: str, user_uuid: str, widget_id: str, context_data: Any):
    try:
        now = time.time()
        cached = mrfp_cache.get(user_uuid)
        if cached and (now - cached["timestamp"] < 3600):
            mrfp = cached["data"]
        else:
            profile_builder = ProfileBuilder(user_uuid)
            mrfp = await profile_builder.build_profile()
            mrfp_cache[user_uuid] = {"data": mrfp, "timestamp": now}

        import os
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model="qwen2.5:1.5b", base_url=base_url, temperature=0,
                         keep_alive=300, verbose=False)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are BudAI, a precise senior financial advisor. 
Below is the user's exact, mathematically verified Multi-Resolution Financial Profile.

<USER_PROFILE>
{mrfp}
</USER_PROFILE>

CRITICAL RULES FOR HALLUCINATION PREVENTION:
1. You may ONLY quote numbers, merchants, and dates that appear exactly inside the <USER_PROFILE> block or the specific data provided.
2. Provide exactly 2-3 sentences of actionable insight. Use GBP currency symbols (£). Emojis are strictly forbidden.
3. Do not round numbers. Use exact GBP amounts.
4. If predicting a future shortage, cite the specific upcoming bill from Tier 2 that will cause it.
"""),
            ("human", "Widget: {widget_id}\nData: {data}")
        ])
        chain = prompt | llm
        result = await chain.ainvoke({"widget_id": widget_id, "data": str(context_data)})
        job_store[job_id] = {"status": "completed",
                             "insight": result.content.strip()}
    except Exception as e:
        logger.error(f"Async advisor task failed for job {job_id}: {e}")
        job_store[job_id] = {"status": "failed", "error": str(e)}


@advisor_router.post("/summarize")
async def summarize_data(request: SummarizeRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "pending"}
    background_tasks.add_task(
        run_advisor_task, job_id, current_user.user_uuid, request.widget_id, request.context_data)
    return {"job_id": job_id}


@advisor_router.get("/status/{job_id}")
async def get_summarize_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
