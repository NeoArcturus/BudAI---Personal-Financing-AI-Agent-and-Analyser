from fastapi import APIRouter, WebSocket, status, Query
from config import SessionLocal
from middleware.auth_middleware import resolve_token_user_info
from services.websocket_manager import manager
from services.logger_setup import get_core_logger
import json

logger = get_core_logger(__name__)

router = APIRouter(tags=["WebSockets"])

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    try:
        with SessionLocal() as db:
            user_uuid, _ = resolve_token_user_info(token, db)

        if not user_uuid:
            logger.warning(json.dumps({"message": "WebSocket connection rejected: No user_uuid resolved", "status_code": 403}))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await manager.connect_and_listen(websocket, user_uuid)

    except Exception as e:
        logger.warning(json.dumps({"message": f"WebSocket connection rejected: {e}", "status_code": 403}))
        try:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        except Exception:
            pass
