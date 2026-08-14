from fastapi import APIRouter, Depends, Query, BackgroundTasks
from models.database_models import User
from middleware.auth_middleware import get_current_user
from fastapi_cache.decorator import cache
from utils.cache_utils import user_cache_key_builder

from controllers.accounts.get import fetch_user_accounts, fetch_transactions, search_user_transactions
from controllers.accounts.post import sync_accounts, extend_bank_connection
from controllers.accounts.delete import revoke_connection

router = APIRouter(prefix="/api/accounts", tags=["accounts"])

@router.get("")
async def get_accounts_route(current_user: User = Depends(get_current_user)):
    return await fetch_user_accounts(current_user.user_uuid)

@router.get("/transactions/search")
async def search_transactions_route(q: str = Query(...), current_user: User = Depends(get_current_user)):
    return await search_user_transactions(current_user.user_uuid, q)

@router.get("/{account_id}/transactions")
@cache(expire=300, namespace="transactions", key_builder=user_cache_key_builder)
async def get_transactions_route(
    account_id: str,
    background_tasks: BackgroundTasks,
    from_date: str = Query(None, alias="from"),
    to_date: str = Query(None, alias="to"),
    current_user: User = Depends(get_current_user)
):
    from services.api_integrator.truelayer_sync import TrueLayerSync
    sync_service = TrueLayerSync(user_id=current_user.user_uuid)
    # Background task to sync with TrueLayer
    background_tasks.add_task(
        sync_service.trigger_sync,
        account_id=account_id,
        user_uuid=current_user.user_uuid,
        from_date=from_date,
        to_date=to_date
    )
    return await fetch_transactions(current_user.user_uuid, account_id, from_date, to_date)

@router.post("/sync")
async def sync_accounts_route(background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    return await sync_accounts(current_user.user_uuid, background_tasks)

@router.post("/banks/{bank_uuid}/extend")
async def extend_bank_connection_route(bank_uuid: str, current_user: User = Depends(get_current_user)):
    return await extend_bank_connection(bank_uuid, current_user.user_uuid)

@router.delete("/{provider_id}")
async def revoke_connection_route(provider_id: str, current_user: User = Depends(get_current_user)):
    return await revoke_connection(current_user.user_uuid, provider_id)
