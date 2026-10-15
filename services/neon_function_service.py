import os
import requests
from typing import Dict, Any, Optional
from config import NEON_FUNCTION_HELLO_BASE_URL
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class NeonFunctionService:
    def __init__(self):
        self.hello_url = NEON_FUNCTION_HELLO_BASE_URL
        if self.hello_url:
            self.hello_url = self.hello_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.hello_url)

    def invoke_hello(self) -> Dict[str, Any]:
        if not self.hello_url:
            raise RuntimeError("NEON_FUNCTION_HELLO_BASE_URL is not configured.")
        
        try:
            response = requests.get(self.hello_url, timeout=15)
            response.raise_for_status()
            return {
                "status_code": response.status_code,
                "url": self.hello_url,
                "response": response.text
            }
        except Exception as e:
            logger.error(f"Failed to invoke Neon hello function: {e}")
            raise RuntimeError(f"Error invoking Neon function: {e}")

    def invoke_function(
        self,
        base_url: str,
        path: str = "/",
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        req_headers = headers or {}
        try:
            if method.upper() == "POST":
                resp = requests.post(url, json=payload, headers=req_headers, timeout=30)
            else:
                resp = requests.get(url, params=payload, headers=req_headers, timeout=30)
            resp.raise_for_status()
            
            try:
                body = resp.json()
            except Exception:
                body = resp.text

            return {
                "status_code": resp.status_code,
                "url": url,
                "data": body
            }
        except Exception as e:
            logger.error(f"Failed to invoke Neon function at {url}: {e}")
            raise RuntimeError(f"Neon function invocation error: {e}")

neon_function_service = NeonFunctionService()
