import os
import json
from config import redis_client
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def set_account_state(account_id: str, state_code: str, expire_seconds: int = 300):
    """
    Sets the background processing state of an account.
    Defaults to 5 minutes expiration as a failsafe lock.
    """
    try:
        redis_client.setex(f"account_state:{account_id}", expire_seconds, state_code)
        logger.debug(json.dumps({"message": f"Set state {state_code} for account {account_id}", "status_code": 100}))
    except Exception as e:
        logger.error(json.dumps({"message": f"Redis error setting state: {e}", "status_code": 500}))

def get_account_state(account_id: str) -> str:
    """
    Retrieves the current background processing state of an account.
    Returns None if no state is active.
    """
    try:
        val = redis_client.get(f"account_state:{account_id}")
        return val
    except Exception as e:
        logger.error(json.dumps({"message": f"Redis error getting state: {e}", "status_code": 500}))
        return None

def clear_account_state(account_id: str):
    """
    Removes the background processing state lock for an account.
    """
    try:
        redis_client.delete(f"account_state:{account_id}")
        logger.debug(json.dumps({"message": f"Cleared state for account {account_id}", "status_code": 100}))
    except Exception as e:
        logger.error(json.dumps({"message": f"Redis error clearing state: {e}", "status_code": 500}))
