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
    logger.info("Entering user_cache_key_builder")

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
    logger.info("Entering global_cache_key_builder")
    req_url = str(request.url.path)
    if request.url.query:
        req_url += f"?{request.url.query}"
    url_hash = hashlib.md5(req_url.encode()).hexdigest()
    return f"{namespace}:global:{func.__name__}:{url_hash}"

