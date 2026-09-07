import json
import asyncio
import pandas as pd
from fastapi import HTTPException
from services.api_integrator.account_reader import AccountReader
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

async def fetch_user_accounts(user_uuid: str):
    """
    Retrieves a list of all bank accounts linked to the user's profile.
    
    Args:
        user_uuid (str): The unique identifier of the requesting user.
        
    Returns:
        dict: A dictionary containing the list of account objects.
        
    Raises:
        HTTPException (500): If fetching the accounts fails.
    """
    try:
        user_acc = AccountReader(user_id=user_uuid)
        all_accounts = await asyncio.to_thread(user_acc.get_all_accounts)
        
        # Clamp negative physical account balances to 0 for the frontend.
        # This prevents the frontend's totalWealth calculation from evaluating to negative,
        # ensuring that the DEFAULT bucket displays 0 as per user requirements.
        for acc in all_accounts:
            if acc.get("balance") is not None and acc["balance"] < 0:
                acc["balance"] = 0.0
                
        return {"accounts": all_accounts}
    except Exception as e:
        logger.error(json.dumps({"message": f"Failed to fetch accounts for user {user_uuid}: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail="Failed to fetch accounts.")

async def fetch_transactions(user_uuid: str, account_id: str, from_date: str = None, to_date: str = None):
    """
    Retrieves and formats transaction history for a specific account.
    Parses dates, handles null values, and sorts chronologically descending.
    
    Args:
        user_uuid (str): The unique identifier of the requesting user.
        account_id (str): The ID of the specific account to query, or 'ALL'.
        from_date (str, optional): Start date string (YYYY-MM-DD).
        to_date (str, optional): End date string (YYYY-MM-DD).
        
    Returns:
        dict: A payload containing the formatted transaction list.
        
    Raises:
        HTTPException (500): If parsing or data retrieval fails.
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
        logger.error(json.dumps({"message": f"Failed to fetch transactions for user {user_uuid}, account {account_id}: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail="Failed to fetch transactions.")

async def search_user_transactions(user_uuid: str, query: str):
    """
    Searches across all a user's transactions based on a text query string.
    Filters by checking for substrings in descriptions and categories.
    
    Args:
        user_uuid (str): The unique identifier of the requesting user.
        query (str): The search term to match against transactions.
        
    Returns:
        dict: A payload containing up to 20 matched transactions.
        
    Raises:
        HTTPException (500): If the search operation fails.
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
        logger.error(json.dumps({"message": f"Failed to search transactions for user {user_uuid}: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail="Failed to search transactions.")
