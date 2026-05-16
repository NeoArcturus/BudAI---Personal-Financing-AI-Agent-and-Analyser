from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from models.database_models import User
from middleware.auth_middleware import get_current_user
from services.api_integrator.get_account_detail import UserAccounts
from utils.cache_utils import user_cache_key_builder
import pandas as pd
import logging

logger = logging.getLogger("uvicorn.error")

account_router = APIRouter(prefix="/api/accounts", tags=["accounts"])


@account_router.get("")
@cache(expire=300, namespace="accounts", key_builder=user_cache_key_builder)
def get_accounts(current_user: User = Depends(get_current_user)):
    try:
        user_acc = UserAccounts(user_id=current_user.user_uuid)
        all_accounts = user_acc.get_all_accounts()
        return {"accounts": all_accounts}
    except Exception:
        raise HTTPException(
            status_code=500, detail="Failed to fetch accounts.")


@account_router.get("/{account_id}/transactions")
@cache(expire=300, namespace="transactions", key_builder=user_cache_key_builder)
def get_transactions(account_id: str, from_date: str = Query(None, alias="from"), to_date: str = Query(None, alias="to"), current_user: User = Depends(get_current_user)):
    try:
        user_acc = UserAccounts(user_id=current_user.user_uuid)
        df = user_acc.get_bank_transactions(
            account_id, current_user.user_uuid, from_date, to_date)
            
        if df is None or df.empty:
            return {"transactions": []}

        # Standardize and clean for frontend consumption
        # Ensure we use 'date' as the primary key
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            df = df.sort_values(by='date', ascending=False)
            # Remove rows where date could not be parsed (NaT)
            df = df.dropna(subset=['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)

        # Standardize category key
        if 'Category' in df.columns and 'category' not in df.columns:
            df['category'] = df['Category']
            
        df = df.fillna("")
        txs = df.to_dict('records')
        return {"transactions": txs}
    except Exception as e:
        logger.error(f"Failed to fetch transactions: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to fetch transactions.")


@account_router.delete("/{provider_id}")
async def revoke_connection(provider_id: str, current_user: User = Depends(get_current_user)):
    try:
        user_acc = UserAccounts(user_id=current_user.user_uuid)
        success = user_acc.revoke_provider_connection(provider_id)
        if success:
            await FastAPICache.clear(namespace="accounts", key=str(current_user.user_uuid))
            await FastAPICache.clear(namespace="transactions", key=str(current_user.user_uuid))
            return {"status": "success"}
        raise HTTPException(
            status_code=500, detail="Failed to revoke connection")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
