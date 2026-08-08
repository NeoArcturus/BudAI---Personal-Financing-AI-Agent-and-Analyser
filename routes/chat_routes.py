from fastapi import APIRouter, Depends, BackgroundTasks
from typing import List
from middleware.auth_middleware import get_current_user
from models.database_models import User
from schemas.api_schema import ChatRequest, StreamChatRequest, ChatSessionResponse, ChatSessionRenameRequest

from controllers.chat.get import get_task_status, list_chat_sessions, get_chat_session
from controllers.chat.post import async_chat, stream_chat, create_chat_session, chat
from controllers.chat.patch import rename_chat_session
from controllers.chat.delete import delete_chat_session

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/async")
async def async_chat_route(request: ChatRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    return await async_chat(request, background_tasks, current_user)

@router.post("/stream")
async def stream_chat_route(request: StreamChatRequest, current_user: User = Depends(get_current_user)):
    return await stream_chat(request, current_user)

@router.get("/status/{task_id}")
async def get_task_status_route(task_id: str):
    return await get_task_status(task_id)

@router.post("/sessions")
async def create_chat_session_route(current_user: User = Depends(get_current_user)):
    return await create_chat_session(current_user)

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions_route(current_user: User = Depends(get_current_user)):
    return await list_chat_sessions(current_user)

@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session_route(session_id: str, current_user: User = Depends(get_current_user)):
    return await get_chat_session(session_id, current_user)

@router.patch("/sessions/{session_id}")
async def rename_chat_session_route(session_id: str, request: ChatSessionRenameRequest, current_user: User = Depends(get_current_user)):
    return await rename_chat_session(session_id, request, current_user)

@router.delete("/sessions/{session_id}")
async def delete_chat_session_route(session_id: str, current_user: User = Depends(get_current_user)):
    return await delete_chat_session(session_id, current_user)

@router.post("")
async def chat_route(request: ChatRequest, current_user: User = Depends(get_current_user)):
    return await chat(request, current_user)
