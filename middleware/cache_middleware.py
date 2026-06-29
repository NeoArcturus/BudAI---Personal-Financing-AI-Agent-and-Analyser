from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class StripCacheControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware to strip Cache-Control and Pragma headers from incoming requests.
    This prevents fastapi-cache from bypassing the cache when a user performs a hard refresh.
    """
    async def dispatch(self, request: Request, call_next):
        logger.debug(f"Intercepting request: {request.url.path}")
        

        headers = request.scope.get("headers", [])
        new_headers = [
            (k, v) for k, v in headers 
            if k.lower() not in (b"cache-control", b"pragma")
        ]
        
        if len(new_headers) < len(headers):
            logger.info(f"Stripped cache-bypass headers from request to {request.url.path}")
            request.scope["headers"] = new_headers
            
        response = await call_next(request)
        return response
