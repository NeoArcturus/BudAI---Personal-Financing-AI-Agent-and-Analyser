import json
from fastapi import BackgroundTasks
import requests
from services.logger_setup import get_core_logger
from services.api_integrator.truelayer_sync import TrueLayerSync
from config import SessionLocal, ENCRYPTION_KEY
from models.database_models import Bank, Account
from models.status_codes import OpenBankingStatus
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
                
                # Transition state to AI Categorization Processing
                from utils.state_manager import set_account_state
                set_account_state(acc_id, "400-102")
                
                sync_service = TrueLayerSync(user_id=user_uuid)
                sync_service.process_and_store_transactions(session, tx_data, user_uuid, bank_uuid, acc_id)
                
                # PHASE 2: Trigger Event Bus Flow AFTER data is stored
                from services.orchestration.prefect_flows import flow_event_bus_sync
                flow_event_bus_sync(user_uuid)
                logger.info(json.dumps({"message": f"Triggered Phase 2 Event Bus Flow for {user_uuid} post-sync.", "status_code": 200}))

                
            else:
                logger.error(json.dumps({"message": f"Failed to fetch results from {results_uri}. Status: {res.status_code}, Response: {res.text}", "status_code": 500}))
                
    except Exception as e:
        logger.error(json.dumps({"message": f"Error in background async fetch: {e}", "status_code": 500}), exc_info=True)

async def handle_truelayer_webhook(payload: dict, background_tasks: BackgroundTasks, user_uuid: str, bank_uuid: str, acc_id: str):
    """
    Webhook endpoint to receive status updates for asynchronous TrueLayer data syncs.
    Implements Phase 2 Idempotency and triggers the Prefect Event Bus Sync flow.
    """
    status = payload.get("status")
    task_id = payload.get("task_id")
    
    logger.info(json.dumps({"message": f"Received TrueLayer Async Webhook for task {task_id} with status: {status}", "status_code": 200}))
    
    # 1. Idempotency Lock
    if task_id:
        from config import redis_client
        lock_key = f"webhook_lock:{task_id}"
        if redis_client.get(lock_key):
            logger.info(json.dumps({"message": f"Idempotency hit: Task {task_id} already processed. Dropping payload.", "status_code": 200}))
            return {"status": "ok", "message": "Already processed"}
        # Set lock for 24 hours
        redis_client.setex(lock_key, 86400, "1")
    
    if status == "Succeeded":
        results_uri = payload.get("results_uri")
        if results_uri and user_uuid and bank_uuid and acc_id:
            # Traditional background task for fetching raw data
            background_tasks.add_task(
                background_fetch_async_results, 
                results_uri, 
                user_uuid, 
                bank_uuid, 
                acc_id
            )
            
            # PHASE 2: Trigger the Prefect Event Bus Flow to align buckets
            # (Note: In a production setup, the Prefect flow should strictly run AFTER the fetch is complete, 
            # so we could trigger it at the end of background_fetch_async_results, but triggering it here 
            # illustrates the architecture handoff).
            logger.info(json.dumps({"message": f"Background task for {user_uuid} queued.", "status_code": 200}))
        else:
            logger.warning(json.dumps({"message": f"Missing required query params (user_uuid, bank_uuid, acc_id) or results_uri.", "status_code": 400}))
            
    elif status == "Failed":
        error_desc = payload.get("error_description", "Unknown error")
        logger.error(json.dumps({"message": f"TrueLayer async task {task_id} failed: {error_desc}", "status_code": 500}))
        
        # SCA Lock detected from TrueLayer
        if "sca" in error_desc.lower() or "psu authentication" in error_desc.lower():
            logger.info(json.dumps({"message": f"SCA exemption expired for {acc_id}. Changing consent_status to CONNECTION_EXPIRED immediately.", "status_code": 200}))
            
            with SessionLocal() as session:
                bank = session.query(Bank).filter_by(bank_uuid=bank_uuid, user_uuid=user_uuid).first()
                if bank:
                    bank.consent_status = OpenBankingStatus.CONNECTION_EXPIRED.value
                    session.commit()
                    logger.info(json.dumps({"message": f"Consent status updated to CONNECTION_EXPIRED for bank {bank_uuid}", "status_code": 200}))
            
            # The fallback sync can still be attempted for the last 89 days (Open Banking rules),
            # but the UI will now properly reflect the REAUTH_REQUIRED state.
            from datetime import datetime, timedelta
            fallback_from = (datetime.utcnow() - timedelta(days=89)).strftime("%Y-%m-%d")
            
            def fallback_sync_task():
                try:
                    sync_service = TrueLayerSync(user_id=user_uuid)
                    sync_service.trigger_sync(account_id=acc_id, user_uuid=user_uuid, from_date=fallback_from)
                except Exception as e:
                    logger.error(json.dumps({"message": f"Fallback sync failed: {e}", "status_code": 500}))
                    
            background_tasks.add_task(fallback_sync_task)
            
    return {"status": "ok", "message": "Webhook processed"}
