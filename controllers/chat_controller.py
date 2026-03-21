from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from middleware.auth_middleware import get_current_user
from models.database_models import User
from schemas.api_schema import ChatRequest
from services.budai_chat_service import execute_chat_stream
import asyncio

chat_router = APIRouter(prefix="/api/chat", tags=["chat"])


@chat_router.post("/")
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    q = execute_chat_stream(request.input, current_user.user_uuid,
                            current_user.name, request.active_account_id)

    async def generate():
        while True:
            await asyncio.sleep(0.01)
            if not q.empty():
                token = q.get()
                if token is None:
                    break
                yield token

    return StreamingResponse(generate(), media_type="text/plain")
