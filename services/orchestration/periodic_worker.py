from prefect import flow, task
from datetime import datetime
import json
from config import SessionLocal
from models.database_models import User, Transaction, Bucket, SystemAlert
from services.logger_setup import get_core_logger
from services.orchestration.prefect_flows import flow_agent_evaluation_loop

logger = get_core_logger(__name__)

@task
def evaluate_user(user_uuid: str, last_evaluated: datetime):
    try:
        with SessionLocal() as session:
            # Check for new data since last_evaluated
            # Because buckets and alerts might not have a created_at column explicitly, 
            # we will rely heavily on new transactions to trigger a state evaluation.
            
            new_txs = session.query(Transaction).filter(
                Transaction.user_uuid == user_uuid,
                Transaction.date > last_evaluated.date() # Simplified to date if timestamp unavailable
            ).all()
            
            new_alerts = session.query(SystemAlert).filter(
                SystemAlert.user_id == user_uuid,
                SystemAlert.timestamp > last_evaluated
            ).all()
            
            new_buckets = session.query(Bucket).filter(
                Bucket.user_id == user_uuid,
                Bucket.updated_at > last_evaluated
            ).all()

            # If there's no new physical delta or system alerts, skip to save GPU.
            if not new_txs and not new_alerts and not new_buckets:
                logger.info(json.dumps({"message": f"Periodic Worker for {user_uuid}: No new events. Sleeping.", "status_code": 200}))
                return

            logger.info(json.dumps({"message": f"Periodic Worker for {user_uuid}: Found {len(new_txs)} txs, {len(new_alerts)} alerts, {len(new_buckets)} bucket changes. Waking AI.", "status_code": 200}))
            
            # Wake the LLM. The flow_agent_evaluation_loop({"status": "synced", "source": "10_min_cron"}, user_uuid) implicitly uses context_builder
            # which will fetch the latest state of buckets and transactions.
            flow_agent_evaluation_loop({"status": "synced", "source": "10_min_cron"}, user_uuid)
            
            # Update the High Water Mark
            user = session.query(User).filter_by(user_uuid=user_uuid).first()
            if user:
                user.last_evaluated_at = datetime.utcnow()
                session.commit()
                
    except Exception as e:
        logger.error(json.dumps({"message": f"Periodic Worker failed for user {user_uuid}: {e}", "status_code": 500}))

@flow(name="continuous_evaluation_loop", log_prints=True)
def run_periodic_evaluation():
    logger.info(json.dumps({"message": "Starting 10-minute System Periodic Worker", "status_code": 200}))
    try:
        with SessionLocal() as session:
            users = session.query(User).all()
            for user in users:
                # If never evaluated, set a baseline
                if not user.last_evaluated_at:
                    user.last_evaluated_at = datetime.utcnow()
                    session.commit()
                    # Optionally wake them on first run
                    logger.info(json.dumps({"message": f"Baseline periodic_worker set for {user.user_uuid}", "status_code": 200}))
                    flow_agent_evaluation_loop({"status": "synced", "source": "10_min_cron"}, user.user_uuid)
                else:
                    evaluate_user(user.user_uuid, user.last_evaluated_at)
                    
    except Exception as e:
        logger.error(json.dumps({"message": f"Global periodic_worker failed: {e}", "status_code": 500}))

if __name__ == "__main__":
    run_periodic_evaluation()
