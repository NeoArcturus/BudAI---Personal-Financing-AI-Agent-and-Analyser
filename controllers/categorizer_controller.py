from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text

from middleware.auth_middleware import get_current_user
from models.database_models import User
from schemas.api_schema import TransactionLabelCorrectionRequest, RetrainCategorizerRequest
from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from config import SessionLocal


categorizer_router = APIRouter(prefix="/api/categorizer", tags=["categorizer"])


@categorizer_router.post("/labels")
def save_manual_label(payload: TransactionLabelCorrectionRequest, current_user: User = Depends(get_current_user)):
    try:
        normalized_label = payload.corrected_label.strip()
        if not normalized_label:
            raise HTTPException(status_code=400, detail="corrected_label cannot be empty.")

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

        retrain_result = None
        if payload.retrain_model:
            retrain_result = agent.retrain_from_feedback(current_user.user_uuid)

        return {
            "status": "success",
            "message": "Label correction saved.",
            "retrain": retrain_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@categorizer_router.post("/retrain")
def retrain_categorizer(payload: RetrainCategorizerRequest, current_user: User = Depends(get_current_user)):
    if not payload.force:
        return {"status": "skipped", "message": "Retraining skipped by request."}
    try:
        agent = CategorizerAgent()
        result = agent.retrain_from_feedback(current_user.user_uuid)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@categorizer_router.get("/review-candidates")
def get_review_candidates(
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
        raise HTTPException(status_code=500, detail=str(e))
