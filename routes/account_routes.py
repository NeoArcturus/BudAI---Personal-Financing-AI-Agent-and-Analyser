from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
import traceback
from middleware.auth_middleware import get_current_user
from services.api_integrator.get_account_detail import UserAccounts

account_router = APIRouter(prefix="/api/accounts", tags=["accounts"])


@account_router.get('/')
def get_accounts(current_user=Depends(get_current_user)):
    try:
        user_acc = UserAccounts(user_id=current_user.user_uuid)
        all_accounts = user_acc.get_all_accounts()
        return {"accounts": all_accounts}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@account_router.get('/{account_id}/transactions')
def get_transactions(account_id: str,
                     from_date: Optional[str] = Query(None, alias="from"),
                     to_date: Optional[str] = Query(None, alias="to"),
                     current_user=Depends(get_current_user)
                     ):
    try:
        user_acc = UserAccounts(user_id=current_user.user_uuid)
        transactions = user_acc.get_bank_transactions(
            account_id, from_date=from_date, to_date=to_date, user_uuid=current_user.user_uuid)
        return {"transactions": transactions}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@account_router.delete('/{provider_id}')
def revoke_connection(provider_id: str, current_user=Depends(get_current_user)):
    try:
        user_acc = UserAccounts(user_id=current_user.user_uuid)
        success = user_acc.revoke_provider_connection(provider_id)
        if success:
            return {"status": "success"}
        raise HTTPException(
            status_code=500, detail="Failed to revoke connection")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
