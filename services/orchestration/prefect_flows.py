import json
from prefect import flow, task
from config import SessionLocal
from services.orchestration.delta_sync import perform_physical_delta_sync
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

@task(retries=3, retry_delay_seconds=5)
def task_physical_delta_sync(user_uuid: str):
    with SessionLocal() as session:
        return perform_physical_delta_sync(session, user_uuid)

@flow(name="Agent_Evaluation_Loop")
def flow_agent_evaluation_loop(sync_result: dict, user_uuid: str):
    with SessionLocal() as session:
        # Phase 3: Gather and Mask
        from services.orchestration.data_masker import DataMasker
        from services.analytics.detection_pipelines import run_detection_pipelines
        from services.orchestration.context_builder import build_evaluation_context
        from services.orchestration.llm_executor import execute_llm_chain
        
        masker = DataMasker()
        masked_data = masker.mask_user_state(session, user_uuid)
        discoveries = run_detection_pipelines(session, user_uuid)
        
        log_pipeline_data(
            logger=logger,
            pipeline="agent_evaluation_loop",
            stage="context_prepared",
            data={
                "user_uuid": user_uuid,
                "sync_result": sync_result,
                "masked_buckets_count": len(masked_data.get("masked_state", [])),
                "discoveries": {
                    "subscriptions_count": len(discoveries.get("new_subscriptions", [])),
                    "utilities_count": len(discoveries.get("new_utilities", []))
                }
            },
            message=f"Agent Evaluation Loop context prepared with {len(masked_data.get('masked_state', []))} buckets"
        )
        
        brief = build_evaluation_context(
            session=session,
            user_uuid=user_uuid,
            masked_state=masked_data["masked_state"],
            discoveries=discoveries,
            trigger_event=sync_result
        )
        
        # Phase 4: Execute
        execute_llm_chain(
            user_uuid=user_uuid,
            compiled_brief=brief,
            reverse_map=masked_data["reverse_map"]
        )

@flow(name="Phase2_Event_Bus_Flow")
def flow_event_bus_sync(user_uuid: str):
    # 1. Delta Sync (The Algorithm)
    sync_result = task_physical_delta_sync(user_uuid)
    
    # 2. Handoff to Agentic Loop (Ringing the Alarm)
    if sync_result and sync_result.get("status") in ["synced", "aligned"]:
        flow_agent_evaluation_loop(sync_result, user_uuid)


@task(retries=3, retry_delay_seconds=2)
def task_fetch_and_store_transactions_1yr(
    account_id: str,
    user_uuid: str,
    bank_uuid: str,
    from_date: str,
    to_date: str
) -> int:
    """
    Synchronously fetches and stores 1 full year of transactions directly
    from TrueLayer at once without sliding windows or webhooks.
    """
    from models.database_models import Bank
    from services.api_integrator.truelayer_sync import TrueLayerSync
    from cryptography.fernet import Fernet
    from config import ENCRYPTION_KEY

    cipher_suite = Fernet(ENCRYPTION_KEY)
    syncer = TrueLayerSync(user_id=user_uuid)

    with SessionLocal() as session:
        bank = session.query(Bank).filter_by(bank_uuid=bank_uuid, user_uuid=user_uuid).first()
        if not bank:
            logger.warning(f"Bank {bank_uuid} not found for account {account_id}")
            return 0

        enc_token = bank.access_token
        if isinstance(enc_token, memoryview):
            enc_token = enc_token.tobytes()
        access_token = cipher_suite.decrypt(bytes(enc_token)).decode()
        provider_id = bank.truelayer_provider_id

        url = f"{syncer.base_url}/{account_id}/transactions"
        params = {"from": from_date, "to": to_date}

        res = syncer._make_request(url, access_token, provider_id, params=params)
        if res is None:
            logger.warning(f"No response from TrueLayer for {account_id} 1-year pull ({from_date} to {to_date})")
            return 0

        if res.status_code == 200:
            tx_data = res.json().get("results", [])
            if tx_data:
                syncer.process_and_store_transactions(session, tx_data, user_uuid, bank_uuid, account_id)
                try:
                    from config import redis_client
                    payload = {
                        "event_type": "TRANSACTIONS_UPDATED",
                        "account_id": account_id,
                        "count": len(tx_data)
                    }
                    redis_client.publish(f"budai_events_{user_uuid}", json.dumps(payload))
                except Exception as re:
                    logger.warning(f"Redis publish notice failed: {re}")
                return len(tx_data)
            return 0

        if res.status_code in [400, 422]:
            logger.info(f"Open Banking boundary or account opening date reached at {from_date} for {account_id}: {res.status_code}")
            return 0

        logger.warning(f"Unexpected status code {res.status_code} fetching transactions for {account_id}: {res.text}")
        return 0


@flow(name="Prefect_Ingest_Account_Transactions_1Yr")
def flow_ingest_account_transactions_1yr(account_id: str, user_uuid: str, bank_uuid: str) -> dict:
    """
    Prefect OSS Flow to pull 1 full year (365 days) of transactions directly at once upon account creation.
    Performs a single request directly to TrueLayer without webhooks or window chunking.
    """
    from datetime import datetime, timedelta

    logger.info(f"Starting Prefect OSS 1-year transaction pull for account {account_id} (user {user_uuid})")

    now = datetime.utcnow()
    from_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    to_date = now.strftime("%Y-%m-%d")

    total_ingested = 0
    try:
        total_ingested = task_fetch_and_store_transactions_1yr(
            account_id=account_id,
            user_uuid=user_uuid,
            bank_uuid=bank_uuid,
            from_date=from_date,
            to_date=to_date
        )
    except Exception as fetch_err:
        logger.warning(f"Error fetching 1-year transactions ({from_date} to {to_date}) for account {account_id}: {fetch_err}")

    # Post-ingestion reconciliation and delta sync
    try:
        sync_result = task_physical_delta_sync(user_uuid)

        from config import redis_client
        try:
            redis_client.publish(f"budai_events_{user_uuid}", json.dumps({
                "event_type": "TRANSACTIONS_UPDATED",
                "account_id": account_id,
                "total_ingested": total_ingested
            }))
            redis_client.publish(f"budai_events_{user_uuid}", json.dumps({
                "event_type": "BUCKET_BALANCES_UPDATED",
                "user_uuid": user_uuid
            }))
        except Exception:
            pass

        if sync_result and sync_result.get("status") in ["synced", "aligned"]:
            flow_agent_evaluation_loop(sync_result, user_uuid)

    except Exception as post_err:
        logger.error(f"Post-ingestion delta sync failed for user {user_uuid}: {post_err}")

    log_pipeline_data(
        logger=logger,
        pipeline="prefect_ingest_1yr",
        stage="completed",
        data={
            "account_id": account_id,
            "user_uuid": user_uuid,
            "total_transactions_ingested": total_ingested
        },
        message=f"Completed Prefect OSS 1-year transaction pull for account {account_id}. Ingested {total_ingested} transactions."
    )
    return {"status": "success", "account_id": account_id, "total_ingested": total_ingested}



