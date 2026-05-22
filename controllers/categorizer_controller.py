from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status
from sqlalchemy import text
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from middleware.auth_middleware import get_current_user
from models.database_models import User, Transaction, BackgroundTask
from schemas.api_schema import TransactionLabelCorrectionRequest, RetrainCategorizerRequest
from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from config import SessionLocal
from utils.cache_utils import user_cache_key_builder
from services.logger_setup import get_core_logger
import pandas as pd
import os
from datetime import datetime
import uuid

logger = get_core_logger(__name__)

categorizer_router = APIRouter(prefix="/api/categorizer", tags=["categorizer"])

def background_retrain_and_recategorize(user_uuid: str, task_id: str):
    try:
        with SessionLocal() as session:
            task = session.query(BackgroundTask).filter_by(task_id=task_id).first()
            if task:
                task.status = "processing"
                session.commit()
        agent = CategorizerAgent()
        retrain_res = agent.retrain_from_feedback(user_uuid)
        with SessionLocal() as session:
            txs = session.query(Transaction).filter_by(user_uuid=user_uuid).all()
            if not txs:
                task = session.query(BackgroundTask).filter_by(task_id=task_id).first()
                if task:
                    task.status = "completed"
                    session.commit()
                return
            df = pd.DataFrame([{
                "transaction_uuid": t.transaction_uuid,
                "description": t.description,
                "amount": t.amount,
                "date": t.date
            } for t in txs])
            from services.Categorizer_Agent.categorizer.preprocessor import Preprocessor
            proc = Preprocessor(df, agent.local_st_path)
            xgb_model_path = os.path.join(agent.model_dir, "gbm_model.joblib")
            enc_path = os.path.join(agent.enc_dir, "label_encoder.joblib")
            if os.path.exists(xgb_model_path) and os.path.exists(enc_path):
                clean_df, embeddings = proc.preprocess_for_inference()
                final_df = agent.categorizer.predict(clean_df, embeddings, xgb_model_path, enc_path)
                category_map = final_df.set_index("transaction_uuid")["Category"].to_dict()
            else:
                logger.warning("Model files not found, skipping prediction")
                category_map = {}
            feedback_rows = session.execute(text("""
                SELECT transaction_uuid, corrected_label
                FROM transaction_label_feedback
                WHERE user_uuid = :user_uuid
            """), {"user_uuid": user_uuid}).fetchall()
            feedback_map = {row[0]: row[1] for row in feedback_rows}
            for t in txs:
                new_cat = feedback_map.get(t.transaction_uuid) or category_map.get(t.transaction_uuid, t.category)
                t.category = new_cat
            session.commit()
            from services.memory_service import MemoryService
            try:
                mem = MemoryService()
                mem.index_transactions([{
                    "transaction_uuid": t.transaction_uuid,
                    "description": t.description,
                    "category": t.category,
                    "amount": t.amount,
                    "date": t.date
                } for t in txs], user_uuid)
            except Exception as e:
                logger.error(f"Failed to update memory index: {e}")
            try:
                from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
                forecaster = ForecasterAgent()
                forecaster.generate_dynamic_parameters(user_uuid)
            except Exception as e:
                logger.error(f"Failed to regenerate dynamic parameters: {e}")
            task = session.query(BackgroundTask).filter_by(task_id=task_id).first()
            if task:
                task.status = "completed"
                session.commit()
    except Exception as e:
        logger.error(f"Task {task_id} failed with critical error: {e}")
        with SessionLocal() as session:
            task = session.query(BackgroundTask).filter_by(task_id=task_id).first()
            if task:
                task.status = "failed"
                session.commit()

@categorizer_router.get("/task-status/{task_id}")
async def get_task_status(task_id: str, current_user: User = Depends(get_current_user)):
    with SessionLocal() as session:
        task = session.query(BackgroundTask).filter_by(task_id=task_id, user_uuid=current_user.user_uuid).first()
        if not task:
            return {"task_id": task_id, "status": "not_found"}
        return {"task_id": task_id, "status": task.status}

@categorizer_router.post("/labels", status_code=status.HTTP_202_ACCEPTED)
async def save_manual_label(payload: TransactionLabelCorrectionRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    try:
        normalized_label = payload.corrected_label.strip()
        if not normalized_label:
            raise HTTPException(
                status_code=400, detail="corrected_label cannot be empty.")
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
                new_task = BackgroundTask(
                    task_id=task_id,
                    user_uuid=current_user.user_uuid,
                    type="retrain_recategorize"
                )
                session.add(new_task)
                session.commit()
                background_tasks.add_task(background_retrain_and_recategorize, current_user.user_uuid, task_id)
        await FastAPICache.clear(namespace="transactions", key=str(current_user.user_uuid))
        await FastAPICache.clear(namespace="categorizer", key=str(current_user.user_uuid))
        return {
            "status": "accepted",
            "task_id": task_id if payload.retrain_model else None,
            "message": "Label correction saved. Retraining queued if requested."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in save_manual_label: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@categorizer_router.post("/retrain", status_code=status.HTTP_202_ACCEPTED)
async def retrain_categorizer(payload: RetrainCategorizerRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    if not payload.force:
        return {"status": "skipped", "message": "Retraining skipped by request."}
    try:
        task_id = str(uuid.uuid4())
        with SessionLocal() as session:
            new_task = BackgroundTask(
                task_id=task_id,
                user_uuid=current_user.user_uuid,
                type="full_retrain"
            )
            session.add(new_task)
            session.commit()
        background_tasks.add_task(background_retrain_and_recategorize, current_user.user_uuid, task_id)
        await FastAPICache.clear(namespace="transactions", key=str(current_user.user_uuid))
        await FastAPICache.clear(namespace="categorizer", key=str(current_user.user_uuid))
        return {
            "status": "accepted", 
            "task_id": task_id,
            "message": "Retraining and re-categorization queued."
        }
    except Exception as e:
        logger.error(f"Error in retrain_categorizer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@categorizer_router.get("/review-candidates")
@cache(expire=300, namespace="categorizer", key_builder=user_cache_key_builder)
async def get_review_candidates(
    account_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    current_user: User = Depends(get_current_user)
):
    try:
        with SessionLocal() as session:
            base_query = """
                SELECT transaction_uuid, account_id, date, amount, description, category
                FROM transactions
                WHERE user_uuid = :user_uuid AND lower(category) = 'needs review'
            """
            params = {"user_uuid": current_user.user_uuid, "limit": limit}
            if account_id:
                base_query += " AND account_id = :account_id"
                params["account_id"] = account_id
            base_query += " ORDER BY date DESC LIMIT :limit"
            rows = session.execute(text(base_query), params).fetchall()
        data = []
        for row in rows:
            data.append({
                "transaction_uuid": row[0],
                "account_id": row[1],
                "date": row[2].isoformat() if hasattr(row[2], "isoformat") else str(row[2]),
                "amount": float(row[3] or 0),
                "description": row[4] or "",
                "predicted_category": row[5] or "Needs Review"
            })
        return {"status": "success", "count": len(data), "items": data}
    except Exception as e:
        logger.error(f"Error in get_review_candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))
