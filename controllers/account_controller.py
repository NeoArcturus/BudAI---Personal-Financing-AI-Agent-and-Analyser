import json
import asyncio
import pandas as pd
from fastapi import HTTPException, BackgroundTasks
from services.api_integrator.account_reader import AccountReader
from services.api_integrator.truelayer_sync import TrueLayerSync
from services.api_integrator.access_token_generator import AccessTokenGenerator
from models.database_models import Bank
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


async def fetch_user_accounts(user_uuid: str):
    """
    Retrieves all bank accounts linked to the user's profile.
    """
    try:
        user_acc = AccountReader(user_id=user_uuid)
        all_accounts = await asyncio.to_thread(user_acc.get_all_accounts)

        # Clamp negative physical account balances to 0 for display
        for acc in all_accounts:
            if acc.get("balance") is not None and acc["balance"] < 0:
                acc["balance"] = 0.0

        return {"accounts": all_accounts}
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Failed to fetch accounts for user {user_uuid}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Failed to fetch accounts.")


async def fetch_transactions(user_uuid: str, account_id: str, from_date: str = None, to_date: str = None):
    """
    Retrieves and formats transaction history for a specific account.
    """
    try:
        user_acc = AccountReader(user_id=user_uuid)
        df = await asyncio.to_thread(
            user_acc.get_transactions,
            account_id, user_uuid, from_date, to_date
        )
        if df is None or df.empty:
            return {"transactions": []}

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='ISO8601', errors='coerce', utc=True)
            df = df.sort_values(by='date', ascending=False)
            df = df.dropna(subset=['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        if 'Category' in df.columns and 'category' not in df.columns:
            df['category'] = df['Category']
        df = df.fillna("")

        txs = df.to_dict('records')
        return {"transactions": txs}
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Failed to fetch transactions for user {user_uuid}, account {account_id}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Failed to fetch transactions.")


async def search_user_transactions(user_uuid: str, query: str):
    """
    Searches across all a user's transactions based on a text query string.
    """
    try:
        user_acc = AccountReader(user_id=user_uuid)
        df = await asyncio.to_thread(
            user_acc.get_transactions, None, user_uuid, None, None
        )
        if df is None or df.empty:
            return {"transactions": []}

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='ISO8601', errors='coerce', utc=True)
            df = df.sort_values(by='date', ascending=False)
            df = df.dropna(subset=['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        if 'Category' in df.columns and 'category' not in df.columns:
            df['category'] = df['Category']
        df = df.fillna("")

        query_lower = query.lower()
        mask = pd.Series(False, index=df.index)
        if 'description' in df.columns:
            mask = mask | df['description'].astype(str).str.lower().str.contains(query_lower, regex=False, na=False)
        if 'category' in df.columns:
            mask = mask | df['category'].astype(str).str.lower().str.contains(query_lower, regex=False, na=False)

        df_filtered = df[mask].head(20)
        txs = df_filtered.to_dict('records')
        return {"transactions": txs}
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Failed to search transactions for user {user_uuid}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail="Failed to search transactions.")


def background_sync_all_accounts(user_uuid: str):
    """
    Background worker that executes sync for all user banks.
    """
    sync = TrueLayerSync(user_id=user_uuid)
    with SessionLocal() as session:
        banks = session.query(Bank).filter_by(user_uuid=user_uuid).all()
        for bank in banks:
            try:
                sync.initialise_accounts(bank.bank_uuid, user_uuid)
            except Exception as e:
                logger.error(json.dumps({
                    "message": f"Sync failed for bank {bank.bank_uuid}: {str(e)}",
                    "status_code": 500
                }))


async def sync_accounts(user_uuid: str, background_tasks: BackgroundTasks):
    """
    Initiates asynchronous account sync.
    """
    background_tasks.add_task(background_sync_all_accounts, user_uuid)
    return {"status": "Accepted", "message": "Account sync initiated in background."}


async def extend_bank_connection(bank_uuid: str, user_uuid: str):
    """
    Extends Open Banking consent for a specific bank connection.
    """
    token_gen = AccessTokenGenerator()
    result = await asyncio.to_thread(token_gen.extend_connection, bank_uuid, user_uuid)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "Internal server error during extension"))

    return result


async def revoke_connection(user_uuid: str, provider_id: str):
    """
    Revokes access for a bank connection and clears cache.
    """
    try:
        token_gen = AccessTokenGenerator()
        results = await asyncio.to_thread(token_gen.revoke_provider, provider_id, user_uuid)

        from utils.cache_utils import clear_user_cache
        clear_user_cache(str(user_uuid), namespace="accounts")
        clear_user_cache(str(user_uuid), namespace="transactions")

        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(json.dumps({
            "message": f"Error revoking connection for user {user_uuid}, provider {provider_id}: {str(e)}",
            "status_code": 500
        }))
        raise HTTPException(status_code=500, detail=str(e))
