from fastapi import HTTPException, Depends
from typing import List
import json
from models.database_models import User, ChatSession, ChatHistory
from schemas.api_schema import ChatSessionResponse, ChatMessageResponse
from config import SessionLocal, redis_client
from middleware.auth_middleware import get_current_user

async def get_task_status(task_id: str):
    """
    Retrieves the status of an asynchronous chat task from Redis.
    
    Args:
        task_id (str): The unique identifier of the chat task.
        
    Returns:
        dict: The job status payload including the streaming result if completed.
        
    Raises:
        HTTPException (404): If the task ID does not exist in Redis.
    """
    task_info_raw = redis_client.get(f"task:{task_id}")
    if not task_info_raw:
        raise HTTPException(status_code=404, detail="Task not found")
    return json.loads(task_info_raw)

async def list_chat_sessions(current_user: User):
    """
    Lists all chat sessions for the current user, ordered by most recently updated.
    
    Args:
        current_user (User): The authenticated user making the request.
        
    Returns:
        list[ChatSessionResponse]: A list of chat session metadata.
        
    Raises:
        HTTPException (500): If there is a database retrieval error.
    """
    try:
        with SessionLocal() as session:
            sessions = session.query(ChatSession).filter_by(
                user_uuid=current_user.user_uuid).order_by(ChatSession.last_updated.desc()).all()
            return [ChatSessionResponse(session_id=s.session_id, title=s.title or "New Conversation", last_updated=s.last_updated, context_data=s.context_data) for s in sessions]
    except Exception:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve chat history.")

async def get_chat_session(session_id: str, current_user: User):
    """
    Retrieves a specific chat session and its chronological message history.
    
    Args:
        session_id (str): The UUID of the requested chat session.
        current_user (User): The authenticated user making the request.
        
    Returns:
        ChatSessionResponse: The session metadata and its full message history.
        
    Raises:
        HTTPException (404): If the chat session is not found.
        HTTPException (500): If there is a database retrieval error.
    """
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id, user_uuid=current_user.user_uuid).first()
            if not chat_session:
                raise HTTPException(
                    status_code=404, detail="Session not found.")
            messages = session.query(ChatHistory).filter_by(
                session_id=session_id).order_by(ChatHistory.timestamp.asc()).all()

            message_responses = [ChatMessageResponse(
                role=m.role, content=m.content, reasoning_content=m.reasoning_content, timestamp=m.timestamp) for m in messages]

            return ChatSessionResponse(
                session_id=chat_session.session_id,
                title=chat_session.title or "New Conversation",
                last_updated=chat_session.last_updated,
                context_data=chat_session.context_data,
                messages=message_responses
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve session details.")
