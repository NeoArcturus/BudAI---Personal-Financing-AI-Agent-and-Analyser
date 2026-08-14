import hashlib
from fastapi import Request, Response
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

def user_cache_key_builder(
    func,
    namespace: str = "",
    request: Request = None,
    response: Response = None,
    *args,
    **kwargs,
):
    """
    Constructs a unique cache key incorporating user UUID, endpoint, and specific query parameters.
    """
    logger.info({"message": f"Entering user_cache_key_builder", "status_code": 200})

    endpoint_kwargs = kwargs.get("kwargs", {})
    user = endpoint_kwargs.get("current_user")
    
    user_uuid = user.user_uuid if hasattr(user, "user_uuid") else "anonymous"
    req_url = str(request.url.path)
    if request.url.query:
        req_url += f"?{request.url.query}"
    url_hash = hashlib.md5(req_url.encode()).hexdigest()
    return f"{namespace}:{user_uuid}:{func.__name__}:{url_hash}"

def global_cache_key_builder(
    func,
    namespace: str = "",
    request: Request = None,
    response: Response = None,
    *args,
    **kwargs,
):
    """
    Constructs a global cache key, ignoring user-specific data to create a shared cache.
    """
    logger.info({"message": f"Entering global_cache_key_builder", "status_code": 200})
    req_url = str(request.url.path)
    if request.url.query:
        req_url += f"?{request.url.query}"
    url_hash = hashlib.md5(req_url.encode()).hexdigest()
    return f"{namespace}:global:{func.__name__}:{url_hash}"

def clear_user_cache(user_uuid: str, namespace: str = None):
    from config import redis_client
    try:
        pattern = f"fastapi-cache:{namespace if namespace else '*'}:{user_uuid}:*"
        keys_to_delete = []
        for key in redis_client.scan_iter(match=pattern):
            keys_to_delete.append(key)
        
        if keys_to_delete:
            redis_client.delete(*keys_to_delete)
            logger.info({"message": f"Cleared {len(keys_to_delete)} cache keys for user {user_uuid} (namespace: {namespace})", "status_code": 200})
    except Exception as e:
        logger.error({"message": f"Failed to clear Redis cache for user {user_uuid}: {e}", "status_code": 500})

