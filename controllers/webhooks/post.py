import json
from fastapi import BackgroundTasks
import requests
from services.logger_setup import get_core_logger
from services.api_integrator.truelayer_sync import TrueLayerSync
from config import SessionLocal, ENCRYPTION_KEY
from models.database_models import Bank, Account
from cryptography.fernet import Fernet

logger = get_core_logger(__name__)
cipher_suite = Fernet(ENCRYPTION_KEY)

def background_fetch_async_results(results_uri: str, user_uuid: str, bank_uuid: str, acc_id: str):
    """
    Background task triggered by a successful TrueLayer webhook. It fetches the completed
    asynchronous transaction sync results using the provided results URI, then stores them in the DB.
    
    Args:
        results_uri (str): The TrueLayer URL where the async results can be downloaded.
        user_uuid (str): The UUID of the user associated with the account.
        bank_uuid (str): The UUID of the bank connection.
        acc_id (str): The specific account ID being synced.
    """
    logger.info(json.dumps({"message": f"Background fetching async results from {results_uri} for account {acc_id}", "status_code": 200}))
    try:
        with SessionLocal() as session:
            bank = session.query(Bank).filter_by(bank_uuid=bank_uuid, user_uuid=user_uuid).first()
            if not bank:
                logger.warning(json.dumps({"message": f"Bank not found for UUID: {bank_uuid}", "status_code": 400}))
                return
            
            enc_token = bank.access_token
            access_token = cipher_suite.decrypt(bytes(enc_token)).decode()
            
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            res = requests.get(results_uri, headers=headers)
            if res.status_code == 200:
                tx_data = res.json().get("results", [])
                logger.info(json.dumps({"message": f"Successfully pulled {len(tx_data)} transactions from async results.", "status_code": 200}))
                
                sync_service = TrueLayerSync(user_id=user_uuid)
                sync_service.process_and_store_transactions(session, tx_data, user_uuid, bank_uuid, acc_id)
                
            else:
                logger.error(json.dumps({"message": f"Failed to fetch results from {results_uri}. Status: {res.status_code}, Response: {res.text}", "status_code": 500}))
                
    except Exception as e:
        logger.error(json.dumps({"message": f"Error in background async fetch: {e}", "status_code": 500}), exc_info=True)

async def handle_truelayer_webhook(payload: dict, background_tasks: BackgroundTasks, user_uuid: str, bank_uuid: str, acc_id: str):
    """
    Webhook endpoint to receive status updates for asynchronous TrueLayer data syncs.
    If the sync succeeded, it queues a background task to retrieve and process the actual data.
    
    Args:
        payload (dict): The webhook JSON payload from TrueLayer containing status and results_uri.
        background_tasks (BackgroundTasks): FastAPI background tasks dependency.
        user_uuid (str): The user UUID passed via query params during webhook registration.
        bank_uuid (str): The bank UUID passed via query params.
        acc_id (str): The account ID passed via query params.
        
    Returns:
        dict: A standard 200 OK acknowledgment payload.
    """
    status = payload.get("status")
    task_id = payload.get("task_id")
    
    logger.info(json.dumps({"message": f"Received TrueLayer Async Webhook for task {task_id} with status: {status}", "status_code": 200}))
    
    if status == "Succeeded":
        results_uri = payload.get("results_uri")
        if results_uri and user_uuid and bank_uuid and acc_id:
            background_tasks.add_task(
                background_fetch_async_results, 
                results_uri, 
                user_uuid, 
                bank_uuid, 
                acc_id
            )
            logger.info(json.dumps({"message": f"Offloaded results fetching for task {task_id} to background tasks.", "status_code": 200}))
        else:
            logger.warning(json.dumps({"message": f"Missing required query params (user_uuid, bank_uuid, acc_id) or results_uri.", "status_code": 400}))
            
    elif status == "Failed":
        error_desc = payload.get("error_description", "Unknown error")
        logger.error(json.dumps({"message": f"TrueLayer async task {task_id} failed: {error_desc}", "status_code": 500}))
        
        # Self-Healing Fallback for strict SCA banks (e.g. Revolut)
        if "sca" in error_desc.lower() or "psu authentication" in error_desc.lower():
            from datetime import datetime, timedelta
            fallback_from = (datetime.utcnow() - timedelta(days=89)).strftime("%Y-%m-%d")
            logger.info(json.dumps({"message": f"SCA exemption expired for {acc_id}. Triggering self-healing fallback sync from {fallback_from}.", "status_code": 200}))
            
            def fallback_sync_task():
                try:
                    sync_service = TrueLayerSync(user_id=user_uuid)
                    sync_service.trigger_sync(account_id=acc_id, user_uuid=user_uuid, from_date=fallback_from)
                except Exception as e:
                    logger.error(json.dumps({"message": f"Fallback sync failed: {e}", "status_code": 500}))
                    
            background_tasks.add_task(fallback_sync_task)
            
    return {"status": "ok", "message": "Webhook processed"}
