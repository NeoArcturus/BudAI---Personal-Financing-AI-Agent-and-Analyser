import uuid
import time
import json
import hashlib
import os
from typing import Any
from fastapi import BackgroundTasks
from models.database_models import User, AdvisorSummary
from config import SessionLocal, redis_client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from services.profile_builder import ProfileBuilder
from services.logger_setup import get_core_logger
from pydantic import BaseModel

logger = get_core_logger(__name__)

class SummarizeRequest(BaseModel):
    widget_id: str
    context_data: Any

def _get_data_hash(data: Any) -> str:
    """
    Computes a SHA-256 hash of the provided context data to ensure idempotency.
    
    Args:
        data (Any): The payload to be hashed.
        
    Returns:
        str: The resulting hexadecimal SHA-256 hash string.
    """
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

async def run_advisor_task(job_id: str, user_uuid: str, widget_id: str, context_data: Any):
    """
    Executes the asynchronous advisor task to generate a financial insight.
    Builds the user's financial profile, queries the LLM, saves the result to the DB,
    and updates the Redis job status.
    
    Args:
        job_id (str): The UUID of the background job.
        user_uuid (str): The unique identifier of the requesting user.
        widget_id (str): The ID of the widget requesting the summary.
        context_data (Any): The specific numerical/transaction data the widget is displaying.
    """
    try:
        data_hash = _get_data_hash(context_data)
        
        with SessionLocal() as session:
            existing = session.query(AdvisorSummary).filter_by(
                user_uuid=user_uuid,
                widget_id=widget_id,
                data_hash=data_hash
            ).first()
            if existing:
                redis_client.set(f"job:{job_id}", json.dumps({"status": "completed", "insight": existing.summary_text}), ex=3600)
                return

        now = time.time()
        cached_mrfp_raw = redis_client.get(f"mrfp:{user_uuid}")
        if cached_mrfp_raw:
            cached_mrfp = json.loads(cached_mrfp_raw)
            mrfp = cached_mrfp["data"]
        else:
            profile_builder = ProfileBuilder(user_uuid)
            mrfp = await profile_builder.build_profile()
            redis_client.set(f"mrfp:{user_uuid}", json.dumps({"data": mrfp, "timestamp": now}), ex=3600)

        base_url = os.getenv("VLLM_SUMMARY_URL", "http://host.docker.internal:8000/v1")
        llm = ChatOpenAI(
            model="lmstudio-community/Qwen3.5-9B-GGUF", # Mac: model="mlx-community/Qwen3.5-4B-4bit",
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
        
        def _sync_llm():
            return chain.invoke({"widget_id": widget_id, "data": str(context_data)})
            
        import asyncio
        result = await asyncio.to_thread(_sync_llm)
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

        redis_client.set(f"job:{job_id}", json.dumps({"status": "completed", "insight": insight}), ex=3600)
    except Exception as e:
        logger.error(json.dumps({"message": f"Async advisor task failed for job {job_id}: {e}", "status_code": 500}))
        redis_client.set(f"job:{job_id}", json.dumps({"status": "failed", "error": str(e)}), ex=3600)
    finally:
        from services.llm_manager import GlobalLLMManager
        GlobalLLMManager.release()

async def summarize_data(request: SummarizeRequest, background_tasks: BackgroundTasks, current_user: User):
    """
    Initiates a background task to generate a 2-3 sentence technical financial insight
    for a specific dashboard widget. Returns a job ID to poll for completion.
    
    Args:
        request (SummarizeRequest): The request payload containing widget ID and context data.
        background_tasks (BackgroundTasks): FastAPI background tasks dependency.
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: A dictionary containing the generated job_id.
    """
    from services.llm_manager import GlobalLLMManager
    GlobalLLMManager.try_acquire()
    
    try:
        data_hash = _get_data_hash(request.context_data)
        
        with SessionLocal() as session:
            existing = session.query(AdvisorSummary).filter_by(
                user_uuid=current_user.user_uuid,
                widget_id=request.widget_id,
                data_hash=data_hash
            ).first()
            if existing:
                job_id = str(uuid.uuid4())
                redis_client.set(f"job:{job_id}", json.dumps({"status": "completed", "insight": existing.summary_text}), ex=3600)
                GlobalLLMManager.release()
                return {"job_id": job_id}

        job_id = str(uuid.uuid4())
        redis_client.set(f"job:{job_id}", json.dumps({"status": "pending"}), ex=3600)
        background_tasks.add_task(
            run_advisor_task, job_id, current_user.user_uuid, request.widget_id, request.context_data)
        return {"job_id": job_id}
    except Exception:
        GlobalLLMManager.release()
        raise
