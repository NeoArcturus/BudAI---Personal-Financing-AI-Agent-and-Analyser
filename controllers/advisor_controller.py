from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uuid
import time
import json
import hashlib
import os
from middleware.auth_middleware import get_current_user
from models.database_models import User, AdvisorSummary
from config import SessionLocal
from langchain_openai import ChatOpenAI
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

def _get_data_hash(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

async def run_advisor_task(job_id: str, user_uuid: str, widget_id: str, context_data: Any):
    try:
        data_hash = _get_data_hash(context_data)
        
        with SessionLocal() as session:
            existing = session.query(AdvisorSummary).filter_by(
                user_uuid=user_uuid,
                widget_id=widget_id,
                data_hash=data_hash
            ).first()
            if existing:
                job_store[job_id] = {"status": "completed", "insight": existing.summary_text}
                return

        now = time.time()
        cached_mrfp = mrfp_cache.get(user_uuid)
        if cached_mrfp and (now - cached_mrfp["timestamp"] < 3600):
            mrfp = cached_mrfp["data"]
        else:
            profile_builder = ProfileBuilder(user_uuid)
            mrfp = await profile_builder.build_profile()
            mrfp_cache[user_uuid] = {"data": mrfp, "timestamp": now}

        base_url = os.getenv("VLLM_SUMMARY_URL", "http://host.docker.internal:8000/v1")
        llm = ChatOpenAI(
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            base_url=base_url,
            api_key="budai-local",
            temperature=0
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are BudAI, a precise senior financial data analyst. Your ONLY goal is to provide a 2-3 sentence technical insight based on the provided data.

            <USER_PROFILE>
            {mrfp}
            </USER_PROFILE>

            CRITICAL GROUNDING RULES:
            1. STRICT DATA BOUNDARY: Use ONLY the numerical values, merchants, and dates found exactly inside the <USER_PROFILE> or the provided context data.
            2. ZERO EXTRAPOLATION: If a metric is not present, state "I do not have data on that."
            3. NO ROUNDING: Use exact GBP (£) amounts as provided.
            4. NO EMOJIS: Emojis are strictly forbidden.
            5. NO FILLER: Provide a direct, professional, and mathematically grounded observation.
            """),
            ("human", "Widget ID: {widget_id}\nContext Data: {data}")
        ])
        
        chain = prompt | llm
        result = await chain.ainvoke({"widget_id": widget_id, "data": str(context_data)})
        insight = result.content.strip()
        
        with SessionLocal() as session:
            new_summary = AdvisorSummary(
                summary_uuid=str(uuid.uuid4()),
                user_uuid=user_uuid,
                widget_id=widget_id,
                data_hash=data_hash,
                summary_text=insight
            )
            session.add(new_summary)
            session.commit()

        job_store[job_id] = {"status": "completed", "insight": insight}
    except Exception as e:
        logger.error(f"Async advisor task failed for job {job_id}: {e}")
        job_store[job_id] = {"status": "failed", "error": str(e)}

@advisor_router.post("/summarize")
async def summarize_data(request: SummarizeRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    data_hash = _get_data_hash(request.context_data)
    
    with SessionLocal() as session:
        existing = session.query(AdvisorSummary).filter_by(
            user_uuid=current_user.user_uuid,
            widget_id=request.widget_id,
            data_hash=data_hash
        ).first()
        if existing:
            job_id = str(uuid.uuid4())
            job_store[job_id] = {"status": "completed", "insight": existing.summary_text}
            return {"job_id": job_id}

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
