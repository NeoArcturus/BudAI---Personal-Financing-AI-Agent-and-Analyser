from fastapi import APIRouter, WebSocket, status, Query
from jose import jwt, JWTError
from config import SECRET_KEY, ALGORITHM
from services.websocket_manager import manager
from services.logger_setup import get_core_logger
import json

logger = get_core_logger(__name__)

router = APIRouter(tags=["WebSockets"])

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_uuid = payload.get("sub")
        if not user_uuid:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
            
        await manager.connect_and_listen(websocket, user_uuid)
        
    except JWTError:
        logger.warning(json.dumps({"message": "WebSocket connection rejected: Invalid JWT Token", "status_code": 403}))
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    except Exception as e:
        logger.error(json.dumps({"message": f"WebSocket handler failed: {e}", "status_code": 500}))
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass
