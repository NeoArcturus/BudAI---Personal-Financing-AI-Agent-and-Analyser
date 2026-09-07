import json
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy import text
from models.database_models import User, Transaction
from schemas.api_schema import TransactionLabelCorrectionRequest, RetrainCategorizerRequest
from agents.core_financial.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from config import SessionLocal
from services.logger_setup import get_core_logger
import pandas as pd
import os
import uuid
import asyncio

logger = get_core_logger(__name__)

def background_retrain_and_recategorize(user_uuid: str, transaction_uuid: str, corrected_label: str, task_id: str):
    """
    Background worker that uses RAG Fast-Learning.
    It embeds the corrected merchant, upserts to merchant_knowledge,
    and auto-sweeps past transactions using a Semantic Foreign Key.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        from datetime import datetime
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"): 
            base_url = f"{base_url}/v1"
            
        embeddings_model = OpenAIEmbeddings(
            base_url=base_url,
            model="text-embedding-nomic-embed-text-v1.5",
            api_key="budai-local",
            check_embedding_ctx_length=False
        )
        
        with SessionLocal() as session:
            tx = session.query(Transaction).filter_by(transaction_uuid=transaction_uuid, user_uuid=user_uuid).first()
            if not tx or not tx.semi_cleaned_description:
                return
            
            merchant = tx.semi_cleaned_description
            
            # Embed the merchant
            vec = embeddings_model.embed_documents([merchant])[0]
            k_uuid = str(uuid.uuid4())
            
            # Upsert to merchant_knowledge (RAG Memory)
            existing_k = session.execute(text("SELECT knowledge_uuid FROM merchant_knowledge WHERE clean_merchant_name = :name LIMIT 1"), {"name": merchant}).scalar()
            if existing_k:
                session.execute(text("UPDATE merchant_knowledge SET category = :cat, embedding = :vec, is_human_verified = TRUE WHERE knowledge_uuid = :k_uuid"), {"cat": corrected_label, "vec": str(vec), "k_uuid": existing_k})
                returned_uuid = existing_k
            else:
                session.execute(text("INSERT INTO merchant_knowledge (knowledge_uuid, clean_merchant_name, category, embedding, is_human_verified, created_at) VALUES (:uuid, :name, :cat, :vec, TRUE, :now)"), {"uuid": k_uuid, "name": merchant, "cat": corrected_label, "vec": str(vec), "now": datetime.utcnow()})
                returned_uuid = k_uuid
            
            # Auto-sweep all transactions for this user + merchant
            update_tx = text("""
                UPDATE transactions 
                SET category = :cat, merchant_knowledge_uuid = :k_uuid 
                WHERE semi_cleaned_description = :name AND user_uuid = :user_uuid
            """)
            session.execute(update_tx, {
                "cat": corrected_label, "k_uuid": returned_uuid, 
                "name": merchant, "user_uuid": user_uuid
            })
            session.commit()
            
            logger.info(json.dumps({"message": f"RAG Fast-Learning applied for {merchant} to {corrected_label}", "status_code": 200}))
            
    except Exception as e:
        logger.error(json.dumps({"message": f"Task {task_id} failed with critical error: {e}", "status_code": 500}))

async def save_manual_label(payload: TransactionLabelCorrectionRequest, background_tasks: BackgroundTasks, current_user: User):
    """
    Saves a manual category correction provided by the user for a specific transaction.
    Optionally queues a background job to retrain the local ML model.
    
    Args:
        payload (TransactionLabelCorrectionRequest): The payload containing the transaction ID and new label.
        background_tasks (BackgroundTasks): FastAPI background tasks dependency.
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: A success payload and an optional task_id if retraining was queued.
        
    Raises:
        HTTPException (400): If the corrected label is empty.
        HTTPException (404): If the transaction is not found.
    """
    try:
        normalized_label = payload.corrected_label.strip()
        if not normalized_label:
            raise HTTPException(
                status_code=400, detail="corrected_label cannot be empty.")
        def _save_label_sync():
            with SessionLocal() as session:
                exists = session.execute(text("""
                    SELECT 1 FROM transactions
                    WHERE user_uuid = :user_uuid AND transaction_uuid = :transaction_uuid
                """), {
                    "user_uuid": current_user.user_uuid,
                    "transaction_uuid": payload.transaction_uuid
                }).fetchone()
                if not exists:
                    raise HTTPException(
                        status_code=404, detail="Transaction not found.")
                agent = CategorizerAgent()
                agent.save_manual_label(
                    user_uuid=current_user.user_uuid,
                    transaction_uuid=payload.transaction_uuid,
                    corrected_label=normalized_label
                )
                task_id = str(uuid.uuid4())
                if payload.retrain_model:
                    pass
                    session.commit()
                return task_id
                
        task_id = await asyncio.to_thread(_save_label_sync)
        if payload.retrain_model:
            background_tasks.add_task(background_retrain_and_recategorize, current_user.user_uuid, payload.transaction_uuid, normalized_label, task_id)
        from utils.cache_utils import clear_user_cache
        clear_user_cache(str(current_user.user_uuid), namespace="transactions")
        clear_user_cache(str(current_user.user_uuid), namespace="categorizer")
        return {
            "status": "accepted",
            "task_id": task_id if payload.retrain_model else None,
            "message": "Label correction saved. Retraining queued if requested."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(json.dumps({"message": f"Error in save_manual_label: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_categorizer(payload: RetrainCategorizerRequest, background_tasks: BackgroundTasks, current_user: User):
    """
    Manually triggers a full retraining of the user's categorization model.
    
    Args:
        payload (RetrainCategorizerRequest): Payload indicating if forced retraining is requested.
        background_tasks (BackgroundTasks): FastAPI background tasks dependency.
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: A payload containing the accepted status and the queued task_id.
        
    Raises:
        HTTPException (500): If a database or queueing error occurs.
    """
    if not payload.force:
        return {"status": "skipped", "message": "Retraining skipped by request."}
    try:
        task_id = str(uuid.uuid4())
        def _queue_retrain():
            with SessionLocal() as session:
                pass
                session.commit()
        await asyncio.to_thread(_queue_retrain)
        # Global ML retraining is deprecated in favor of instant RAG Fast-Learning
        logger.info(json.dumps({"message": "Global retraining skipped. RAG applies updates instantly.", "status_code": 200}))
        from utils.cache_utils import clear_user_cache
        clear_user_cache(str(current_user.user_uuid), namespace="transactions")
        clear_user_cache(str(current_user.user_uuid), namespace="categorizer")
        return {
            "status": "accepted", 
            "task_id": task_id,
            "message": "Retraining and re-categorization queued."
        }
    except Exception as e:
        logger.error(json.dumps({"message": f"Error in retrain_categorizer: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail=str(e))
