from models.database_models import ChatHistory
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List
from middleware.auth_middleware import get_current_user
from models.database_models import User, ChatSession
from schemas.api_schema import ChatRequest, ExplanationRequest, ChatSessionResponse, ChatMessageResponse, StreamChatRequest
from services.orchestrator_graph import execute_chat_graph_async, get_session_history
from config import SessionLocal
import asyncio
import uuid
import json
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

chat_router = APIRouter(prefix="/api/chat", tags=["chat"])

_async_tasks = {}

async def run_graph_task(task_id: str, state_input: dict):
    try:
        q = asyncio.Queue()
        asyncio.create_task(execute_chat_graph_async(state_input, q))
        full_response = []
        while True:
            token = await q.get()
            if token is None:
                break
            if not token.startswith("__TOOL_CALL__") and not token.startswith("__DATA__"):
                full_response.append(token)
            
        _async_tasks[task_id] = {
            "status": "completed",
            "result": "".join(full_response)
        }
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        _async_tasks[task_id] = {"status": "failed", "error": str(e)}

@chat_router.post("/async")
async def async_chat(request: ChatRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
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
    task_id = str(uuid.uuid4())
    _async_tasks[task_id] = {"status": "pending"}
    background_tasks.add_task(run_graph_task, task_id, initial_state)
    return {"task_id": task_id, "status": "queued"}

@chat_router.post("/stream")
async def stream_chat(request: StreamChatRequest, current_user: User = Depends(get_current_user)):
    user_input = ""
    if request.messages:
        last_msg = request.messages[-1]
        if hasattr(last_msg, "content") and last_msg.content:
            user_input = last_msg.content
        elif hasattr(last_msg, "parts") and last_msg.parts:
            user_input = "".join([p.get("text", "") for p in last_msg.parts if p.get("type") == "text"])
        elif isinstance(last_msg, dict):
            user_input = last_msg.get("content", "")
            if not user_input and "parts" in last_msg:
                user_input = "".join([p.get("text", "") for p in last_msg["parts"] if p.get("type") == "text"])
    
    logger.info(f"Stream request: '{user_input[:50]}...', Session: {request.session_id}")
    
    initial_state = {
        "user_uuid": current_user.user_uuid,
        "session_id": request.session_id,
        "active_account_id": request.active_account_id or "ALL",
        "user_input": user_input,
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
    
    q = asyncio.Queue()
    asyncio.create_task(execute_chat_graph_async(initial_state, q))

    async def generate_response():
        logger.info(f"Starting stream generator for session {request.session_id}")
        while True:
            token = await q.get()
            if token is None:
                logger.info(f"Stream queue empty. Closing connection for {request.session_id}")
                break
            
            if token.startswith("__TOOL_CALL__"):
                try:
                    payload = json.loads(token.replace("__TOOL_CALL__", ""))
                    tool_call = {
                        "toolCallId": f"call_{str(uuid.uuid4())[:8]}",
                        "toolName": payload["toolName"],
                        "args": payload["args"]
                    }
                    chunk = f'9:{json.dumps(tool_call)}\n'
                    logger.debug(f"Yielding TOOL chunk: {chunk.strip()}")
                    yield chunk
                except Exception as e:
                    logger.error(f"Tool payload error: {e}")
            elif token.startswith("__DATA__"):
                try:
                    data_payload = json.loads(token.replace("__DATA__", ""))
                    chunk = f'8:[{json.dumps(data_payload)}]\n'
                    logger.debug(f"Yielding DATA chunk: {chunk.strip()}")
                    yield chunk
                except Exception as e:
                    logger.error(f"Data payload error: {e}")
            else:
                chunk = f'0:{json.dumps(token)}\n'
                # Optional: Log small chunks for debugging
                if len(token) < 5: logger.debug(f"Yielding TEXT: {token}")
                yield chunk

    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "X-Accel-Buffering": "no",
            "X-Vercel-AI-Data-Stream": "v1",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@chat_router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    task_info = _async_tasks.get(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_info

@chat_router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions(current_user: User = Depends(get_current_user)):
    try:
        with SessionLocal() as session:
            sessions = session.query(ChatSession).filter_by(user_uuid=current_user.user_uuid).order_by(ChatSession.last_updated.desc()).all()
            return [ChatSessionResponse(session_id=s.session_id, title=s.title or "New Conversation", last_updated=s.last_updated, context_data=s.context_data) for s in sessions]
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")

@chat_router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(session_id=session_id, user_uuid=current_user.user_uuid).first()
            if not chat_session: raise HTTPException(status_code=404, detail="Session not found.")
            messages = [ChatMessageResponse(role=m.role, content=m.content, timestamp=m.timestamp) for m in chat_session.messages]
            return ChatSessionResponse(session_id=chat_session.session_id, title=chat_session.title or "New Conversation", last_updated=chat_session.last_updated, context_data=chat_session.context_data, messages=messages)
    except HTTPException: raise
    except Exception: raise HTTPException(status_code=500, detail="Failed to retrieve session details.")

@chat_router.post("")
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    initial_state = {
        "user_uuid": current_user.user_uuid, "session_id": request.session_id, "active_account_id": request.active_account_id or "ALL",
        "user_input": request.input, "chat_history": get_session_history(current_user.user_uuid, request.session_id),
        "selected_worker": None, "worker_summary": None, "cache_id": None, "chart_type": None, "ui_trigger_tag": None, "final_response": "", "raw_data": None, "is_explanation": False
    }
    q = asyncio.Queue()
    asyncio.create_task(execute_chat_graph_async(initial_state, q))
    async def generate():
        while True:
            token = await q.get()
            if token is None: break
            yield token
    return StreamingResponse(generate(), media_type="text/event-stream")

@chat_router.post("/explain")
async def explain_ui_data(request: ExplanationRequest, current_user: User = Depends(get_current_user)):
    initial_state = {
        "user_uuid": current_user.user_uuid, "session_id": None, "active_account_id": "ALL",
        "user_input": f"Please explain the {request.chart_type} data I am looking at.",
        "chat_history": get_session_history(current_user.user_uuid),
        "selected_worker": "explainer", "worker_summary": None, "cache_id": None, "chart_type": request.chart_type, "ui_trigger_tag": None, "final_response": "", "raw_data": request.raw_data, "is_explanation": True
    }
    q = asyncio.Queue()
    asyncio.create_task(execute_chat_graph_async(initial_state, q))
    async def generate():
        while True:
            token = await q.get()
            if token is None: break
            yield token
    return StreamingResponse(generate(), media_type="text/event-stream")

@chat_router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(session_id=session_id, user_uuid=current_user.user_uuid).first()
            if not chat_session:
                return {"status": "success", "message": "Session not found but considered deleted"}
            session.query(ChatHistory).filter_by(session_id=session_id).delete()
            session.delete(chat_session)
            session.commit()
            return {"status": "success", "message": "Session deleted"}
    except Exception: raise HTTPException(status_code=500, detail="Failed to delete session.")

