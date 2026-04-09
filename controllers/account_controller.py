from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from middleware.auth_middleware import get_current_user
from models.database_models import User
from services.api_integrator.get_account_detail import UserAccounts
import traceback

account_router = APIRouter(prefix="/api/accounts", tags=["accounts"])


@account_router.get("/")
def get_accounts(current_user: User = Depends(get_current_user)):
    try:
        user_acc = UserAccounts(user_id=current_user.user_uuid)
        all_accounts = user_acc.get_all_accounts()
        return {"accounts": all_accounts}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch accounts.")


@account_router.get("/{account_id}/transactions")
def get_transactions(account_id: str, current_user: User = Depends(get_current_user)):
    try:
        user_acc = UserAccounts(user_id=current_user.user_uuid)
        transactions = user_acc.get_transactions_by_account(account_id)
        return {"transactions": transactions}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch transactions.")


@account_router.delete("/{provider_id}")
def revoke_connection(provider_id: str, current_user: User = Depends(get_current_user)):
    try:
        user_acc = UserAccounts(user_id=current_user.user_uuid)
        success = user_acc.revoke_provider_connection(provider_id)
        if success:
            return {"status": "success"}
        raise HTTPException(
            status_code=500, detail="Failed to revoke connection")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to revoke provider connection: {str(e)}")
