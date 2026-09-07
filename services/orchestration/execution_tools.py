import json
import uuid
from langchain_core.tools import tool
from sqlmodel import Session
from models.database_models import SystemAlert, VirtualTransfer
from config import SessionLocal, redis_client
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

# We use factory functions to generate tools tightly bound to the current execution context
# This prevents the LLM from ever seeing or needing to provide the raw UUIDs.

def get_sleep_tool():
    @tool
    def sleep_execution(reason: str) -> str:
        """Use this tool to take no action and terminate the evaluation if the event is routine."""
        logger.info(json.dumps({"message": f"Decision Engine terminating: {reason}", "status_code": 200}))
        return "TERMINATE"
    return sleep_execution

def get_alert_tool(user_uuid: str):
    @tool
    def insert_system_alert(priority: int, message: str) -> str:
        """Use this tool to alert the user of important events or overdrafts."""
        try:
            with SessionLocal() as session:
                alert = SystemAlert(
                    id=str(uuid.uuid4()),
                    user_id=user_uuid,
                    event_type="LLM_GENERATED_ALERT",
                    message=message,
                    urgency_level=priority
                )
                session.add(alert)
                session.commit()
                
                # Push to WebSocket
                try:
                    payload = {
                        "event_type": "NEW_ALERT",
                        "message": message,
                        "priority": priority
                    }
                    redis_client.publish(f"budai_events_{user_uuid}", json.dumps(payload))
                except Exception as ex:
                    logger.error(f"Redis publish failed: {ex}")

                logger.info(json.dumps({"message": "System alert inserted successfully", "status_code": 200}))
                return "ALERT_CREATED"
        except Exception as e:
            logger.error(json.dumps({"message": f"Failed to insert alert: {e}", "status_code": 500}))
            return "ERROR"
    return insert_system_alert

def get_transfer_tool(reverse_map: dict, user_uuid: str):
    @tool
    def execute_virtual_transfer(source_alias: str, target_alias: str, amount: float) -> str:
        """Use this tool to move virtual money between buckets to cover expenses or allocate funds."""
        source_id = reverse_map.get(source_alias)
        target_id = reverse_map.get(target_alias)
        
        if not source_id or not target_id:
            return "ERROR: Invalid bucket alias."
            
        try:
            with SessionLocal() as session:
                from services.bucket_service import execute_bucket_transfer
                execute_bucket_transfer(
                    session=session,
                    user_uuid=user_uuid,
                    source_bucket_id=source_id,
                    target_bucket_id=target_id,
                    amount=amount
                )
                session.commit()
                
                # Push to WebSocket
                try:
                    payload = {
                        "event_type": "VIRTUAL_TRANSFER",
                        "amount": amount
                    }
                    redis_client.publish(f"budai_events_{user_uuid}", json.dumps(payload))
                except Exception as ex:
                    logger.error(f"Redis publish failed: {ex}")

                logger.info(json.dumps({"message": f"Virtual transfer of £{amount} executed.", "status_code": 200}))
                return "TRANSFER_EXECUTED"
        except Exception as e:
            logger.error(json.dumps({"message": f"Transfer failed: {e}", "status_code": 500}))
            return "ERROR: Transfer failed due to constraints."
    return execute_virtual_transfer

def get_etl_tool():
    @tool
    def trigger_etl_pipeline(pipeline_name: str, parameters: dict) -> str:
        """Use this tool to trigger a heavy background compute pipeline (e.g., 'deep_categorization')."""
        logger.info(json.dumps({"message": f"Triggering ETL pipeline: {pipeline_name}", "status_code": 200}))
        # Trigger Prefect Deployment here
        return "ETL_TRIGGERED"
    return trigger_etl_pipeline

def get_create_bucket_tool(user_uuid: str):
    @tool
    def create_virtual_bucket(bucket_type: str, name: str, target_amount: float = 0.0, target_date: str = "", priority_index: float = 0.0) -> str:
        """Use this tool to autonomously create a new virtual bucket (e.g. TARGET_DATE, ACCUMULATING) to track goals or liabilities."""
        from services.bucket_service import create_new_bucket
        from datetime import datetime
        try:
            parsed_date = None
            if target_date:
                parsed_date = datetime.fromisoformat(target_date)
            
            with SessionLocal() as session:
                new_bucket = create_new_bucket(
                    session=session,
                    user_uuid=user_uuid,
                    bucket_type=bucket_type,
                    name=name,
                    target_amount=target_amount if target_amount > 0 else None,
                    target_date=parsed_date,
                    priority_index=priority_index
                )
                session.commit()
                logger.info(json.dumps({"message": f"AI created new bucket '{name}' autonomously.", "status_code": 200}))
                return "BUCKET_CREATED"
        except Exception as e:
            logger.error(json.dumps({"message": f"Failed to create bucket: {e}", "status_code": 500}))
            return f"ERROR: {e}"
    return create_virtual_bucket

def get_update_bucket_tool(reverse_map: dict, user_uuid: str):
    @tool
    def update_virtual_bucket(bucket_alias: str, name: str = None, target_amount: float = None, target_date: str = None, priority_index: float = None) -> str:
        """Use this tool to update an existing bucket's goals, name, or priority."""
        from services.bucket_service import update_bucket_details
        from datetime import datetime
        bucket_id = reverse_map.get(bucket_alias)
        if not bucket_id:
            return "ERROR: Invalid bucket alias."
            
        try:
            parsed_date = None
            if target_date:
                parsed_date = datetime.fromisoformat(target_date)
            
            with SessionLocal() as session:
                update_bucket_details(
                    session=session,
                    user_uuid=user_uuid,
                    bucket_id=bucket_id,
                    name=name,
                    target_amount=target_amount,
                    target_date=parsed_date,
                    priority_index=priority_index
                )
                session.commit()
                
                # Push to WebSocket
                try:
                    from config import redis_client
                    payload = {"event_type": "BUCKET_UPDATED", "bucket_alias": bucket_alias}
                    redis_client.publish(f"budai_events_{user_uuid}", json.dumps(payload))
                except:
                    pass
                    
                logger.info(json.dumps({"message": f"AI updated bucket '{bucket_alias}' autonomously.", "status_code": 200}))
                return "BUCKET_UPDATED"
        except Exception as e:
            logger.error(json.dumps({"message": f"Failed to update bucket: {e}", "status_code": 500}))
            return f"ERROR: {e}"
    return update_virtual_bucket

def get_delete_bucket_tool(reverse_map: dict, user_uuid: str):
    @tool
    def delete_virtual_bucket_tool(bucket_alias: str) -> str:
        """Use this tool to completely remove/delete an active virtual bucket. Any remaining funds will be automatically moved to DEFAULT."""
        from services.bucket_service import delete_virtual_bucket
        bucket_id = reverse_map.get(bucket_alias)
        if not bucket_id:
            return "ERROR: Invalid bucket alias."
            
        try:
            with SessionLocal() as session:
                delete_virtual_bucket(session, user_uuid, bucket_id)
                session.commit()
                
                # Push to WebSocket
                try:
                    from config import redis_client
                    import json
                    payload = {"event_type": "BUCKET_DELETED", "bucket_alias": bucket_alias}
                    redis_client.publish(f"budai_events_{user_uuid}", json.dumps(payload))
                except:
                    pass
                    
                logger.info(json.dumps({"message": f"AI deleted bucket '{bucket_alias}' autonomously.", "status_code": 200}))
                return "BUCKET_DELETED"
        except Exception as e:
            logger.error(json.dumps({"message": f"Failed to delete bucket: {e}", "status_code": 500}))
            return f"ERROR: {e}"
    return delete_virtual_bucket_tool
