import asyncio
from fastapi import HTTPException, BackgroundTasks
from services.api_integrator.truelayer_sync import TrueLayerSync
from models.database_models import Bank
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def background_sync_all_accounts(user_uuid: str):
    """
    Background task that triggers a data sync for all active bank connections
    associated with a specific user.
    
    Args:
        user_uuid (str): The unique identifier of the user to sync.
    """
    sync = TrueLayerSync(user_id=user_uuid)
    with SessionLocal() as session:
        banks = session.query(Bank).filter_by(user_uuid=user_uuid).all()
        for bank in banks:
            try:
                sync.initialise_accounts(bank.bank_uuid, user_uuid)
            except Exception as e:
                logger.error(f"Sync failed for bank {bank.bank_uuid}: {e}")

async def sync_accounts(user_uuid: str, background_tasks: BackgroundTasks):
    """
    Endpoint handler to trigger an asynchronous sync of all linked accounts.
    
    Args:
        user_uuid (str): The user requesting the sync.
        background_tasks (BackgroundTasks): FastAPI background tasks dependency.
        
    Returns:
        dict: An accepted status payload indicating the sync has started.
    """
    background_tasks.add_task(background_sync_all_accounts, user_uuid)
    return {"status": "Accepted", "message": "Account sync initiated in background."}
