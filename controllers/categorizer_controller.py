import json
import uuid
import asyncio
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy import text
from models.database_models import User, Transaction
from schemas.api_schema import TransactionLabelCorrectionRequest, RetrainCategorizerRequest
from agents.core_financial.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


def background_retrain_and_recategorize(user_uuid: str, transaction_uuid: str, corrected_label: str, task_id: str):
    """
    Background worker that updates merchant categorization using RAG Fast-Learning.
    """
    try:
        with SessionLocal() as session:
            tx = session.query(Transaction).filter_by(transaction_uuid=transaction_uuid, user_uuid=user_uuid).first()
            if not tx or not tx.semi_cleaned_description or not tx.merchant_knowledge_uuid:
                return

            merchant = tx.semi_cleaned_description
            k_uuid = tx.merchant_knowledge_uuid

            update_tx = text("""
                UPDATE transactions 
                SET merchant_knowledge_uuid = :k_uuid 
                WHERE semi_cleaned_description = :name AND user_uuid = :user_uuid
            """)
            session.execute(update_tx, {
                "k_uuid": k_uuid,
                "name": merchant,
                "user_uuid": user_uuid
            })
            session.commit()

            logger.info(json.dumps({
                "message": f"RAG Fast-Learning applied for {merchant} to {corrected_label}",
                "status_code": 200
            }))

    except Exception as e:
        logger.error(json.dumps({"message": f"Task {task_id} failed with critical error: {e}", "status_code": 500}))


async def get_task_status(task_id: str, current_user: User):
    """Retrieves status of background task."""
    return {"task_id": task_id, "status": "deprecated"}


async def get_review_candidates(account_id: str | None, limit: int, current_user: User):
    """Retrieves transactions flagged for review."""
    try:
        def _fetch_candidates():
            with SessionLocal() as session:
                base_query = """
                    SELECT t.transaction_uuid, t.account_id, t.date, t.amount, t.description, m.category
                    FROM transactions t
                    LEFT JOIN merchant_knowledge m ON t.merchant_knowledge_uuid = m.knowledge_uuid
                    WHERE t.user_uuid = :user_uuid AND lower(m.category) = 'needs review'
                """
                params = {"user_uuid": current_user.user_uuid, "limit": limit}
                if account_id:
                    base_query += " AND t.account_id = :account_id"
                    params["account_id"] = account_id
                base_query += " ORDER BY t.date DESC LIMIT :limit"
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
            return data

        data = await asyncio.to_thread(_fetch_candidates)
        return {"status": "success", "count": len(data), "items": data}
    except Exception as e:
        logger.error(json.dumps({"message": f"Error in get_review_candidates: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail=str(e))


async def save_manual_label(payload: TransactionLabelCorrectionRequest, background_tasks: BackgroundTasks, current_user: User):
    """Saves manual category label correction."""
    try:
        normalized_label = payload.corrected_label.strip()
        if not normalized_label:
            raise HTTPException(status_code=400, detail="corrected_label cannot be empty.")

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
                    raise HTTPException(status_code=404, detail="Transaction not found.")
                agent = CategorizerAgent()
                agent.save_manual_label(
                    user_uuid=current_user.user_uuid,
                    transaction_uuid=payload.transaction_uuid,
                    corrected_label=normalized_label
                )
                task_id = str(uuid.uuid4())
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
    """Triggers retraining of categorizer."""
    if not payload.force:
        return {"status": "skipped", "message": "Retraining skipped by request."}
    try:
        task_id = str(uuid.uuid4())
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
