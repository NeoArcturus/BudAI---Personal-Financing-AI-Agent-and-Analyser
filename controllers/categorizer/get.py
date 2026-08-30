import json
from fastapi import HTTPException, Query
from sqlalchemy import text
from models.database_models import User
from config import SessionLocal
from services.logger_setup import get_core_logger
import asyncio

logger = get_core_logger(__name__)

async def get_task_status(task_id: str, current_user: User):
    """
    Retrieves the status of an asynchronous background task (e.g., retraining).
    
    Args:
        task_id (str): The UUID of the background task.
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: The status payload for the specified task.
    """
    return {"task_id": task_id, "status": "deprecated"}

async def get_review_candidates(account_id: str | None, limit: int, current_user: User):
    """
    Retrieves transactions that have been flagged as 'Needs Review' by the categorizer.
    
    Args:
        account_id (str | None): Optional account ID to filter by.
        limit (int): Maximum number of transactions to return.
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: A payload containing the count and list of transaction items.
        
    Raises:
        HTTPException (500): If a database error occurs.
    """
    try:
        def _fetch_candidates():
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
            return data
            
        data = await asyncio.to_thread(_fetch_candidates)
        return {"status": "success", "count": len(data), "items": data}
    except Exception as e:
        logger.error(json.dumps({"message": f"Error in get_review_candidates: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail=str(e))
