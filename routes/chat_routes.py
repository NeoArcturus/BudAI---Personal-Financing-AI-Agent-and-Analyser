from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.orchestrator_graph import execute_chat_graph
from langchain_core.messages import HumanMessage, AIMessage
from config import SessionLocal
from models.database_models import ChatHistory

router = APIRouter()


class ChatRequest(BaseModel):
    input: str
    active_account_id: str
    user_id: str
    chat_history: list = []


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


@router.post("/api/chat/")
async def chat_endpoint(request: ChatRequest):
    initial_state = {
        "user_uuid": request.user_id,
        "active_account_id": request.active_account_id,
        "user_input": request.input,
        "chat_history": get_session_history(request.user_id),
        "selected_worker": None,
        "worker_summary": None,
        "cache_id": None,
        "chart_type": None,
        "ui_trigger_tag": None,
        "final_response": ""
    }
    q = execute_chat_graph(initial_state)

    def stream_generator():
        while True:
            token = q.get()
            if token is None:
                break
            yield token

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
