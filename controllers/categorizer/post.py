import json
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy import text
from models.database_models import User, BackgroundTask, Transaction
from schemas.api_schema import TransactionLabelCorrectionRequest, RetrainCategorizerRequest
from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from config import SessionLocal
from services.logger_setup import get_core_logger
import pandas as pd
import os
import uuid
import asyncio

logger = get_core_logger(__name__)

def background_retrain_and_recategorize(user_uuid: str, task_id: str):
    """
    Background worker function that retrains the user's specific XGBoost categorization model
    based on manual feedback, then recategorizes all historical transactions, updates memory indexes,
    and refreshes dynamic forecasting parameters.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        task_id (str): The UUID of the background task tracking this execution.
    """
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
                logger.warning(json.dumps({"message": f"Model files not found, skipping prediction", "status_code": 400}))
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
                logger.error(json.dumps({"message": f"Failed to update memory index: {e}", "status_code": 500}))
            try:
                from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
                forecaster = ForecasterAgent()
                accounts = session.execute(text("SELECT account_id FROM accounts WHERE user_uuid = :user_uuid"), {"user_uuid": user_uuid}).fetchall()
                for (acc_id,) in accounts:
                    forecaster.generate_dynamic_parameters(user_uuid, acc_id)
            except Exception as e:
                logger.error(json.dumps({"message": f"Failed to regenerate dynamic parameters: {e}", "status_code": 500}))
            task = session.query(BackgroundTask).filter_by(task_id=task_id).first()
            if task:
                task.status = "completed"
                session.commit()
    except Exception as e:
        logger.error(json.dumps({"message": f"Task {task_id} failed with critical error: {e}", "status_code": 500}))
        with SessionLocal() as session:
            task = session.query(BackgroundTask).filter_by(task_id=task_id).first()
            if task:
                task.status = "failed"
                session.commit()

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
                    new_task = BackgroundTask(
                        task_id=task_id,
                        user_uuid=current_user.user_uuid,
                        type="retrain_recategorize"
                    )
                    session.add(new_task)
                    session.commit()
                return task_id
                
        task_id = await asyncio.to_thread(_save_label_sync)
        if payload.retrain_model:
            background_tasks.add_task(background_retrain_and_recategorize, current_user.user_uuid, task_id)
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
                new_task = BackgroundTask(
                    task_id=task_id,
                    user_uuid=current_user.user_uuid,
                    type="full_retrain"
                )
                session.add(new_task)
                session.commit()
        await asyncio.to_thread(_queue_retrain)
        background_tasks.add_task(background_retrain_and_recategorize, current_user.user_uuid, task_id)
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
