from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from middleware.auth_middleware import get_current_user
from models.database_models import User
from schemas.api_schema import ChatRequest
from services.orchestrator_graph import execute_chat_graph
from langchain_core.messages import HumanMessage, AIMessage
from config import SessionLocal
from models.database_models import ChatHistory
import asyncio

chat_router = APIRouter(prefix="/api/chat", tags=["chat"])


def get_session_history(user_uuid: str):
    history = []
    with SessionLocal() as session:
        records = session.query(ChatHistory).filter(
            ChatHistory.user_uuid == user_uuid).order_by(ChatHistory.timestamp.asc()).all()
        for r in records:
            if r.role == "user":
                history.append(HumanMessage(content=r.content))
            else:
                history.append(AIMessage(content=r.content))
    return history


@chat_router.post("/")
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    initial_state = {
        "user_uuid": current_user.user_uuid,
        "active_account_id": request.active_account_id,
        "user_input": request.input,
        "chat_history": get_session_history(current_user.user_uuid),
        "selected_worker": None,
        "worker_summary": None,
        "cache_id": None,
        "chart_type": None,
        "ui_trigger_tag": None,
        "final_response": ""
    }

    q = execute_chat_graph(initial_state)

    async def generate():
        while True:
            await asyncio.sleep(0.01)
            if not q.empty():
                token = q.get()
                if token is None:
                    break
                yield token

    return StreamingResponse(generate(), media_type="text/event-stream")
