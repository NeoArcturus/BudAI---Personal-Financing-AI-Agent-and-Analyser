from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
from middleware.auth_middleware import get_current_user
from models.database_models import User, ChatSession
from schemas.api_schema import ChatRequest, ExplanationRequest, ChatSessionResponse, ChatMessageResponse
from services.orchestrator_graph import execute_chat_graph, get_session_history
from config import SessionLocal
import asyncio
from services.logger_setup import get_core_logger
from services.celery_app import execute_langgraph_workflow
from celery.result import AsyncResult

logger = get_core_logger(__name__)

chat_router = APIRouter(prefix="/api/chat", tags=["chat"])

@chat_router.post("/async")
async def async_chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    initial_state = {
        "user_uuid": current_user.user_uuid,
        "session_id": request.session_id,
        "active_account_id": request.active_account_id or "ALL",
        "user_input": request.input,
        "chat_history": get_session_history(current_user.user_uuid, request.session_id),
        "selected_worker": None,
        "worker_summary": None,
        "cache_id": None,
        "chart_type": None,
        "ui_trigger_tag": None,
        "final_response": "",
        "raw_data": None,
        "is_explanation": False
    }
    task = execute_langgraph_workflow.delay(initial_state)
    return {"task_id": task.id, "status": "queued"}

@chat_router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=execute_langgraph_workflow.app)
    if task_result.state == 'PENDING':
        return {"status": "pending"}
    elif task_result.state != 'FAILURE':
        return {
            "status": "completed",
            "result": task_result.result
        }
    else:
        return {"status": "failed", "error": str(task_result.info)}


@chat_router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions(current_user: User = Depends(get_current_user)):
    try:
        with SessionLocal() as session:
            sessions = session.query(ChatSession).filter_by(
                user_uuid=current_user.user_uuid).order_by(ChatSession.last_updated.desc()).all()
            result = [
                ChatSessionResponse(
                    session_id=s.session_id,
                    title=s.title or "New Conversation",
                    last_updated=s.last_updated,
                    context_data=s.context_data
                ) for s in sessions
            ]
            return result
    except Exception as e:
        logger.error(
            f"Failed to list chat sessions for user {current_user.user_uuid}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve chat history.")


@chat_router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id, user_uuid=current_user.user_uuid).first()
            if not chat_session:
                raise HTTPException(
                    status_code=404, detail="Session not found.")
            messages = [
                ChatMessageResponse(
                    role=m.role,
                    content=m.content,
                    timestamp=m.timestamp
                ) for m in chat_session.messages
            ]
            return ChatSessionResponse(
                session_id=chat_session.session_id,
                title=chat_session.title or "New Conversation",
                last_updated=chat_session.last_updated,
                context_data=chat_session.context_data,
                messages=messages
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get chat session {session_id} for user {current_user.user_uuid}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve session details.")


@chat_router.post("")
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    initial_state = {
        "user_uuid": current_user.user_uuid,
        "session_id": request.session_id,
        "active_account_id": request.active_account_id or "ALL",
        "user_input": request.input,
        "chat_history": get_session_history(current_user.user_uuid, request.session_id),
        "selected_worker": None,
        "worker_summary": None,
        "cache_id": None,
        "chart_type": None,
        "ui_trigger_tag": None,
        "final_response": "",
        "raw_data": None,
        "is_explanation": False
    }
    q = execute_chat_graph(initial_state)

    async def generate():
        while initial_state.get('session_id') is None:
            await asyncio.sleep(0.1)
        yield f"[SESSION_ID:{initial_state['session_id']}]"
        while True:
            await asyncio.sleep(0.01)
            if not q.empty():
                token = q.get()
                if token is None:
                    break
                yield token
    return StreamingResponse(generate(), media_type="text/event-stream")


@chat_router.post("/explain")
async def explain_ui_data(request: ExplanationRequest, current_user: User = Depends(get_current_user)):
    initial_state = {
        "user_uuid": current_user.user_uuid,
        "session_id": None,
        "active_account_id": "ALL",
        "user_input": f"Please explain the {request.chart_type} data I am looking at.",
        "chat_history": get_session_history(current_user.user_uuid),
        "selected_worker": "explainer",
        "worker_summary": None,
        "cache_id": None,
        "chart_type": request.chart_type,
        "ui_trigger_tag": None,
        "final_response": "",
        "raw_data": request.raw_data,
        "is_explanation": True
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
