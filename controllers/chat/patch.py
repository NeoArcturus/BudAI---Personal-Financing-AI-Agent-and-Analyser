import json
from fastapi import HTTPException
from models.database_models import User, ChatSession
from schemas.api_schema import ChatSessionRenameRequest
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

async def rename_chat_session(session_id: str, request: ChatSessionRenameRequest, current_user: User):
    """
    Renames the title of a specific chat session.
    
    Args:
        session_id (str): The UUID of the chat session to rename.
        request (ChatSessionRenameRequest): The payload containing the new title.
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: A success payload including the new title.
        
    Raises:
        HTTPException (404): If the session is not found.
        HTTPException (500): If a database error occurs.
    """
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id, user_uuid=current_user.user_uuid).first()
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            chat_session.title = request.title
            session.commit()
            return {"status": "success", "title": request.title}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(json.dumps({"message": f"Error renaming session {session_id}: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail="Internal server error")
