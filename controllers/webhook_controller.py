import json
from datetime import datetime, timedelta, timezone
from fastapi import BackgroundTasks
import requests
from services.logger_setup import get_core_logger
from services.api_integrator.truelayer_sync import TrueLayerSync
from config import SessionLocal, ENCRYPTION_KEY, redis_client
from models.database_models import Bank, Account
from models.status_codes import OpenBankingStatus
from cryptography.fernet import Fernet

logger = get_core_logger(__name__)
cipher_suite = Fernet(ENCRYPTION_KEY)


def background_fetch_async_results(results_uri: str, user_uuid: str, bank_uuid: str, acc_id: str):
    """
    Background worker that fetches asynchronous transaction sync results from TrueLayer.
    """
    logger.info(json.dumps({
        "message": f"Background fetching async results from {results_uri} for account {acc_id}",
        "status_code": 200
    }))
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
                logger.info(json.dumps({
                    "message": f"Successfully pulled {len(tx_data)} transactions from async results.",
                    "status_code": 200
                }))

                from utils.state_manager import set_account_state
                set_account_state(acc_id, "400-102")

                sync_service = TrueLayerSync(user_id=user_uuid)
                sync_service.process_and_store_transactions(session, tx_data, user_uuid, bank_uuid, acc_id)

                try:
                    from services.orchestration.prefect_flows import flow_event_bus_sync
                    flow_event_bus_sync(user_uuid)
                    logger.info(json.dumps({
                        "message": f"Triggered Phase 2 Event Bus Flow for {user_uuid} post-sync.",
                        "status_code": 200
                    }))
                except Exception as prefect_err:
                    logger.error(json.dumps({
                        "message": f"Failed to trigger Prefect Event Bus Flow: {prefect_err}",
                        "status_code": 500
                    }), exc_info=True)
            else:
                logger.error(json.dumps({
                    "message": f"Failed to fetch results from {results_uri}. Status: {res.status_code}, Response: {res.text}",
                    "status_code": 500
                }))

    except Exception as e:
        logger.error(json.dumps({"message": f"Error in background async fetch: {e}", "status_code": 500}), exc_info=True)


async def handle_truelayer_webhook(payload: dict, background_tasks: BackgroundTasks, user_uuid: str = None, bank_uuid: str = None, acc_id: str = None):
    """
    Receives and processes webhook events from TrueLayer.
    """
    status = payload.get("status")
    task_id = payload.get("task_id")
    event_id = payload.get("event_id")
    event_type = payload.get("event_type")
    acc_id = acc_id or payload.get("account_id")

    dedup_key = task_id or event_id or f"{acc_id}_{event_type}"
    logger.info(json.dumps({
        "message": f"Received TrueLayer Webhook: event={event_type}, task={task_id}, status={status}, acc_id={acc_id}",
        "status_code": 200
    }))

    if dedup_key:
        try:
            lock_key = f"webhook_lock:{dedup_key}"
            if redis_client.get(lock_key):
                logger.info(json.dumps({
                    "message": f"Idempotency hit: Webhook {dedup_key} already processed. Dropping payload.",
                    "status_code": 200
                }))
                return {"status": "ok", "message": "Already processed"}
            redis_client.setex(lock_key, 86400, "1")
        except Exception as redis_err:
            logger.warning(json.dumps({
                "message": f"Redis idempotency check warning: {redis_err}. Proceeding with sync.",
                "status_code": 200
            }))

    if acc_id and (not user_uuid or not bank_uuid):
        try:
            with SessionLocal() as session:
                acc = session.query(Account).filter_by(account_id=acc_id).first()
                if acc:
                    user_uuid = user_uuid or acc.user_uuid
                    bank_uuid = bank_uuid or acc.bank_uuid
        except Exception as db_err:
            logger.error(json.dumps({
                "message": f"Error resolving account metadata for {acc_id}: {db_err}",
                "status_code": 500
            }))

    if status and status.lower() == "succeeded":
        results_uri = payload.get("results_uri")
        if results_uri and user_uuid and bank_uuid and acc_id:
            background_tasks.add_task(
                background_fetch_async_results,
                results_uri,
                user_uuid,
                bank_uuid,
                acc_id
            )
            logger.info(json.dumps({"message": f"Background task for {user_uuid} queued via results_uri.", "status_code": 200}))
        elif acc_id and user_uuid:
            def background_delta_sync():
                try:
                    sync_service = TrueLayerSync(user_id=user_uuid)
                    sync_service.trigger_sync(account_id=acc_id, user_uuid=user_uuid)
                except Exception as e:
                    logger.error(json.dumps({"message": f"Background delta sync failed for {acc_id}: {e}", "status_code": 500}))

            background_tasks.add_task(background_delta_sync)
            logger.info(json.dumps({"message": f"Direct delta sync queued for {acc_id}.", "status_code": 200}))
        else:
            logger.warning(json.dumps({"message": "Missing account_id or unable to resolve user_uuid from webhook.", "status_code": 400}))

    elif status == "Failed":
        error_desc = payload.get("error_description", "Unknown error")
        logger.error(json.dumps({"message": f"TrueLayer async task {task_id} failed: {error_desc}", "status_code": 500}))

        if "sca" in error_desc.lower() or "psu authentication" in error_desc.lower():
            logger.info(json.dumps({
                "message": f"SCA exemption expired for {acc_id}. Changing consent_status to CONNECTION_EXPIRED immediately.",
                "status_code": 200
            }))

            with SessionLocal() as session:
                bank = session.query(Bank).filter_by(bank_uuid=bank_uuid, user_uuid=user_uuid).first()
                if bank:
                    bank.consent_status = OpenBankingStatus.CONNECTION_EXPIRED.value
                    session.commit()

            fallback_from = (datetime.now(timezone.utc) - timedelta(days=89)).strftime("%Y-%m-%d")

            def fallback_sync_task():
                try:
                    sync_service = TrueLayerSync(user_id=user_uuid)
                    sync_service.trigger_sync(account_id=acc_id, user_uuid=user_uuid, from_date=fallback_from)
                except Exception as e:
                    logger.error(json.dumps({"message": f"Fallback sync failed: {e}", "status_code": 500}))

            background_tasks.add_task(fallback_sync_task)

    return {"status": "ok", "message": "Webhook processed"}
